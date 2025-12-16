import os
import yaml
import argparse
from pathlib import Path

from scent import evaluate, preprocess, train


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, help="config path", default="./configs/config.yaml"
)
parser.add_argument(
    "--mode",
    type=str,
    help="script mode",
    choices=["preprocess", "train", "evaluate"],
    required=True,
)
parser.add_argument("--graph-training", type=bool, help="graph training", default=False)
parser.add_argument(
    "--eval-mode", type=str, help="eval mode", choices=["aida", "shadowlink"]
)
parser.add_argument("--eval-checkpoint", type=str, help="eval checkpoint")


def main(args, config):

    if args.mode == "preprocess":
        preprocess(args, config)

    if args.mode == "train":
        train(args, config)

    if args.mode == "evaluate":
        evaluate(args, config)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    main(args, config)
