from __future__ import annotations

import argparse
import sys


def _run_train(args: list[str]) -> None:
    from . import train

    train.main(args)


def _run_eval(args: list[str]) -> None:
    from . import eval as eval_mod

    eval_mod.main(args)


def _run_serve(args: list[str]) -> None:
    from . import fastapi_app

    fastapi_app.main(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="causal_consistency_nn",
        description="Causal consistency neural network utilities",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train", help="Train a model")
    sub.add_parser("eval", help="Evaluate a trained model")
    sub.add_parser("serve", help="Serve a trained model via FastAPI")

    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        return

    args, extra = parser.parse_known_args(argv)

    if args.command == "train":
        _run_train(extra)
    elif args.command == "eval":
        _run_eval(extra)
    elif args.command == "serve":
        _run_serve(extra)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
