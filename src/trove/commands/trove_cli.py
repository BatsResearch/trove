#!/usr/bin/env python

# from here:
# https://github.com/huggingface/datasets/blob/a1b5a32365cd2c6631fa4ad236a881f9f62bf32b/src/datasets/commands/datasets_cli.py
from argparse import ArgumentParser

from trove.commands.env import EnvironmentCommand


def parse_unknown_args(unknown_args):
    return {
        key.lstrip("-"): value
        for key, value in zip(unknown_args[::2], unknown_args[1::2])
    }


def main():
    parser = ArgumentParser(
        "Trove CLI tool", usage="trove-cli <command> [<args>]", allow_abbrev=False
    )
    commands_parser = parser.add_subparsers(help="trove-cli command helpers")

    # Register commands
    EnvironmentCommand.register_subcommand(commands_parser)

    # Parse args
    args, unknown_args = parser.parse_known_args()
    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)
    kwargs = parse_unknown_args(unknown_args)

    # Run
    service = args.func(args, **kwargs)
    service.run()


if __name__ == "__main__":
    main()
