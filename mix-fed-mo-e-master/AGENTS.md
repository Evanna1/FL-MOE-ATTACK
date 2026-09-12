# MixFedMoE

This is our repo of MixFedMoE

## Models & Datasets

model: Switch-base-8 (enc-dec architecture with interleaved expert layer.)
datasets: we support multiple text classification tasks, including:

```python
# in proof_of_concept\poc_moe.py
SUPPORTED_DATASET_NAME_TO_HF: Dict[str, str] = {
    "ag_news": "ag_news",
    "imdb": "imdb",
    # Hardcoded alias for your 20 News setup.
    "20news": "SetFit/20_newsgroups",
    "sst2": "SetFit/sst2",
    "yelp_polarity": "yelp_polarity",
    "emotion": "emotion",
}
```

FL Framework: Flower
Summary: we are performing Federated Learning with a Mixture of Expert model in Text Classification Tasks.

## Experiment Plan

### Core role

One parameter server and many clients (the number is defined by hyper-parameters)

### Data Partitioning

- Using Dirichlet distribution to create Non-IID subsets for each client. Leave the $\alpha$ (heterogeneity factor) as a hyper-parameter.
- Implementation Hint: You can use `flwr_datasets.partitioner.DirichletPartitioner` if available, or write a simple numpy-based index splitter.
- Each client must hold a distinct subset of data to provoke different Expert Activations.

### Client Simulation

Client support 3 methods, which is in the POC file @proof_of_concept\poc_moe.py.

1. Full-precision training: Client receives Full precision data. All clients train on full FP32 precision
2. Mixed-precision training: Client receives full FP32 model from server, then locally casts cold experts to BF16 (frozen) while hot experts stay FP32 and trainable.
3. Drop training: Client receives full FP32 model from server, then locally drops unassigned experts and trains assigned experts in FP32.

For the server to orchestrate the FL process, all clients should report:

1. Expert Activation Map in the training process, we can maually specify `output_router_logits=True` in the model forward to capture the router logits, but in this case, you need to write your own training code instead of using the Hugginface Trainer. Another way is to perform a short calibration after local training to capture the feature map, it depends on the implementation compexity, choose the one with minimum affort.
2. Metrics: training loss, number of samples trained etc, you can add metrics.

Hyper-params, including but not limited to:

1. Local training rounds, we use rounds to control the training behavior instead of gradient steps.
2. Local traiing parameters, such as weight decay, learning rate, learning scheduler, optmizer is hard coded to AdamW.

### Parameter Server

Role: assign the weights to each client according to the client's training record last round, aggregate the parameters.

Aggregate Rule:

- **Weighted FedAvg** for the Router and shared layers.
- **Sparse Aggregation** for Experts: Only average the weights of Expert K from clients who actually trained Expert K (Hot). the rule is also weighted.

Expert Allocation:

- Based on the uploaded profiles from client, decide which experts are "Hot" for each client for the next round.
- Constraint: Ensure every expert is covered by at least 1 client globally.
- The allocated/pruned experts number is controlled by a hyper-parameter `K`. We temporally assign same `K` for every client. (Data heterogeinity will make training speed different because of different data sizes)

### Main Experiment Loop

- Run for `R` rounds (e.g., 20 rounds).
- **Process:**
  1. Server distributes global model.
  2. Spawns 8 Processes (Clients) using Flower framework.
  3. Clients train in parallel.
  4. Server collects results.
  5. Evaluate on a held-out global Test Set.

Evaluation policy: centralized server-side evaluation only. Do not run client-side federated evaluation (`fraction_evaluate=0.0`).

### Record

1. Step to eval Accuracy (Record the original data)
2. Time to eval Accuracy (Record the original data)
3. Step/Time to Train loss (if convinient, otherwise drop this)

### Flower Architecture

@flower_hf_example\hf folder:

- shows an example of training an FL model via hugginface transformer lib. this may help you because we load model/data via hugginface lib

@flower_hf_example\simulation\tutorial.ipynb

- shows how to perform simulation, we need to perform simulation.
- CRITICAL IMPLEMENTATION DETAIL: Use `from flwr.simulation import run_simulation` to run this experiment on a single machine with multi-GPU support. Do not attempt to use manual socket programming.
- for more details, you can refer to: https://flower.ai/docs/framework/how-to-run-simulations.html

@https://flower.ai/docs/framework/tutorial-series-build-a-strategy-from-scratch-pytorch.html

- this website shows how to modiy a strategy, if you can not fetch, I will convert this web to markdown.
- You MUST implement a Custom Strategy class inheriting from `flwr.server.strategy.FedAvg`.
- You need to modify configure fit, aggregate_fit and other functions to meet my request.

### Code Standards (Simplicity First)

- Hardcoding is OK: Since we only use switch-base-8, you can hardcode layer names (e.g., encoder.block...mlp.router) to avoid complex graph traversal.
- Type Hinting: Use Dict[str, Any], List[float], etc.
- Error Handling: Fail fast. Do not add complex try-catch blocks. If a shape mismatches, let it crash so we can fix the logic.
- Imports: Keep imports clean.

### Critical System Constraints for Simulation

**1. Tracking "Wall-clock Time" accurately in Simulation:**
Because we are _simulating_ FL, true wall-clock time might be skewed by Ray's overhead. To accurately plot "Time vs Accuracy" reflecting the Straggler Effect:

- Each Client's `fit()` method MUST measure its own execution time (e.g., `start = time.time() ... end = time.time()`) and return `local_training_time = end - start` inside the `metrics` dictionary.
- In the Strategy's `aggregate_fit()`, calculate the `round_time = max(client_times)`. Accumulate this `round_time` globally. This is the only fair way to prove MixFedMoE is faster than Full-FedAvg in a synchronous FL setup.

**2. StateDict to Flower Parameters Conversion:**
HuggingFace uses PyTorch `state_dict` (OrderedDict of Tensors), but Flower communicates using a list of NumPy ndarrays (`List[np.ndarray]`). It defined the parameter object by itself. you must follow the Flower's Standard which is provided in the examples.

- For simplicity in this simulation, convert and transmit the **full FP32 model parameters** each round.
- Apply `mix`/`drop` behavior locally on client after loading full parameters (freeze BF16 experts in mix, prune experts in drop).

**3. HuggingFace Trainer Memory Leak Prevention:**
If using `Trainer` inside the Client's `fit` method, be careful. The Client function is called repeatedly. Ensure that the model and Trainer are properly garbage-collected or moved back to CPU after `fit` completes to prevent CUDA OOM across rounds. `torch.cuda.empty_cache()` at the end of `fit` might be helpful. You can Also use custom training function, because we do not use many advanced features such as LR_Scheduler, grad accumlation etc, we wish the training process is as simple as @flower_hf_example\hf\huggingface_example\client_app.py.

**4. Use official implementation.**
When trying to implement something, Identify if FLOWER Framework have implemented it first. for common methods like Data Partition and Diriclet Distribution, the flower framework has built-in method. so you can use the existing methods.

### Other Information

I use windows for development (Current Machine) but the code will be run in linux server. My current machine do not have cuda env. but using CPU for a short test is a valid option.

Both machine have conda env named `flwr`, you must activate this env before any tests.

Test in my development machine can be slow, so you can try to creat a small model with random weights for testing.
