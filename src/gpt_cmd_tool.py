import argparse
import json
import os

import torch

from model import device
from train import Trainer


def _positive_int(value, name):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be > 0")
    return ivalue


def _train_split(value):
    fvalue = float(value)
    if not (0.0 < fvalue < 1.0):
        raise argparse.ArgumentTypeError("train_split must be between 0 and 1")
    return fvalue


def build_parser():
    parser = argparse.ArgumentParser(
        description="Command-line training tool for the self-attention toy transformer"
    )

    parser.add_argument("--input", default="input.txt", help="Path to training text file")
    parser.add_argument(
        "--encoder-type",
        default="char",
        help="Encoder type (e.g. char or a tiktoken encoding name)",
    )

    parser.add_argument("--max-len", type=lambda v: _positive_int(v, "max_len"), default=256)
    parser.add_argument("--d-embed", type=lambda v: _positive_int(v, "d_embed"), default=384)
    parser.add_argument("--n-heads", type=lambda v: _positive_int(v, "n_heads"), default=6)
    parser.add_argument("--d-hidden", type=lambda v: _positive_int(v, "d_hidden"), default=1536)
    parser.add_argument("--n-blocks", type=lambda v: _positive_int(v, "n_blocks"), default=6)
    parser.add_argument("--base", type=lambda v: _positive_int(v, "base"), default=10000)

    parser.add_argument("--batch-size", type=lambda v: _positive_int(v, "batch_size"), default=64)
    parser.add_argument("--block-size", type=lambda v: _positive_int(v, "block_size"), default=256)
    parser.add_argument("--epochs", type=lambda v: _positive_int(v, "epochs"), default=100)
    parser.add_argument("--train-split", type=_train_split, default=0.9)

    parser.add_argument(
        "--save-path",
        default="",
        help="Optional custom checkpoint path. If omitted, Trainer uses timestamp naming.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved model/training config before training",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    model_config = {
        "n_vocab": None,
        "max_len": args.max_len,
        "d_embed": args.d_embed,
        "n_heads": args.n_heads,
        "d_hidden": args.d_hidden,
        "n_blocks": args.n_blocks,
        "base": args.base,
    }

    training_config = {
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "epochs": args.epochs,
        "train_split": args.train_split,
    }

    if args.print_config:
        print("Resolved config:")
        print(
            json.dumps(
                {
                    "device": device,
                    "encoder_type": args.encoder_type,
                    "model_config": model_config,
                    "training_config": training_config,
                    "input": args.input,
                    "save_path": args.save_path or "<trainer default>",
                },
                indent=2,
            )
        )

    trainer = Trainer(
        encoder_type=args.encoder_type,
        text=text,
        model_config=model_config,
        training_config=training_config,
    )

    if args.save_path:
        checkpoint = trainer.train(save_checkpoint=False)
        torch.save(checkpoint, args.save_path)
        print(f"Saved checkpoint to: {args.save_path}")
    else:
        trainer.train(save_checkpoint=True)
        print("Training finished and checkpoint saved with default timestamp naming.")


if __name__ == "__main__":
    main()
