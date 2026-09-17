from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_trigger_optimization":
        from mixfedmoe_fl.trigger_optimization import main as trigger_optimization_main

        trigger_optimization_main(arguments[1:])
        return
    parser = argparse.ArgumentParser(description="MixFedMoE experiment commands.")
    parser.add_argument("command", choices=["_trigger_optimization"])
    parser.parse_args(arguments)


if __name__ == "__main__":
    main()
