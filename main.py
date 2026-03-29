import argparse
import json

from src.config import load_config


def build_parser():
    parser = argparse.ArgumentParser(description="Train KD-RetinaNet from a central config file.")
    parser.add_argument(
        "--config",
        default="configs/resnet101_teacher_resnet50_student.json",
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, for example --set optimizer.lr=0.01",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config and exit.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.set)

    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2))
        return

    from train import train_from_config

    train_from_config(config)


if __name__ == "__main__":
    main()
