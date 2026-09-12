from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Optional

from flwr.client import Client
from flwr.clientapp import ClientApp
from flwr.common import Context
from flwr.server import ServerAppComponents, ServerConfig
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation

from mixfedmoe_fl.client import MixFedMoEClient
from mixfedmoe_fl.config import MixFedMoEConfig, parse_config
from mixfedmoe_fl.data import MixFedMoEDataManager
from mixfedmoe_fl.logging_utils import (
    RoundMetricsLogger,
    build_timestamped_output_dir,
    write_run_config,
)
from mixfedmoe_fl.server import build_initial_parameters, make_central_evaluate_fn
from mixfedmoe_fl.strategy import MixFedMoEStrategy

_CLIENT_DATA_MANAGER: Optional[MixFedMoEDataManager] = None


def _silence_hf_model_loading_logs() -> None:
    # from_pretrained has no dedicated "quiet" argument. Silence via logger levels.
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass

    try:
        from datasets.utils import logging as ds_logging

        ds_logging.set_verbosity_error()
    except Exception:
        pass

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _get_client_data_manager(runtime_config: MixFedMoEConfig) -> MixFedMoEDataManager:
    global _CLIENT_DATA_MANAGER
    if _CLIENT_DATA_MANAGER is None:
        _CLIENT_DATA_MANAGER = MixFedMoEDataManager.from_config(runtime_config)
    return _CLIENT_DATA_MANAGER


def _extract_partition_id(context: Context) -> int:
    if "partition-id" in context.node_config:
        return int(context.node_config["partition-id"])
    if "partition_id" in context.node_config:
        return int(context.node_config["partition_id"])
    raise KeyError("Client context.node_config must contain 'partition-id'.")


def _build_client_app(runtime_config: MixFedMoEConfig) -> ClientApp:
    def client_fn(context: Context) -> Client:
        partition_id = _extract_partition_id(context)
        dm = _get_client_data_manager(runtime_config)
        client = MixFedMoEClient(
            client_id=partition_id,
            runtime_config=runtime_config,
            data_manager=dm,
            is_malicious=(partition_id in runtime_config.malicious_clients),
        )
        return client.to_client()

    return ClientApp(client_fn=client_fn)


def _build_server_app(runtime_config: MixFedMoEConfig) -> ServerApp:
    server_data_manager = MixFedMoEDataManager.from_config(runtime_config)
    metrics_logger = RoundMetricsLogger(runtime_config.output_dir)
    parameter_names, initial_parameters = build_initial_parameters(
        runtime_config=runtime_config,
        data_manager=server_data_manager,
    )
    evaluate_fn = make_central_evaluate_fn(
        runtime_config=runtime_config,
        data_manager=server_data_manager,
        parameter_names=parameter_names,
        metrics_logger=metrics_logger,
    )
    strategy = MixFedMoEStrategy(
        runtime_config=runtime_config,
        parameter_names=parameter_names,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
    )

    def server_fn(context: Context) -> ServerAppComponents:
        _ = context
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=runtime_config.num_rounds),
        )

    return ServerApp(server_fn=server_fn)


def main() -> None:
    _silence_hf_model_loading_logs()
    runtime_config = parse_config()
    if runtime_config.fraction_evaluate != 0.0:
        raise ValueError(
            "MixFedMoE uses centralized server evaluation only. "
            "Set --fraction_evaluate=0.0."
        )

    runtime_config = replace(
        runtime_config,
        output_dir=build_timestamped_output_dir(runtime_config.output_dir),
    )
    os.makedirs(runtime_config.output_dir, exist_ok=True)
    config_path = write_run_config(runtime_config, runtime_config.output_dir)
    print(f"Run output directory: {runtime_config.output_dir}")
    print(f"Saved run config to {config_path}")

    server_app = _build_server_app(runtime_config)
    client_app = _build_client_app(runtime_config)

    backend_config = {
        "init_args": {"ignore_reinit_error": True, "include_dashboard": False},
        "client_resources": {
            "num_cpus": float(runtime_config.num_cpus_per_client),
            "num_gpus": float(runtime_config.num_gpus_per_client),
        },
    }

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=runtime_config.num_clients,
        backend_name="ray",
        backend_config=backend_config,
        verbose_logging=False,
    )


if __name__ == "__main__":
    main()
