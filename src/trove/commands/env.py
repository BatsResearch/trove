"""From here:

https://github.com/huggingface/datasets/blob/a1b5a32365cd2c6631fa4ad236a881f9f62bf32b/src/datasets/commands/env.py
"""

import platform
from argparse import ArgumentParser

from trove import __version__ as version
from trove.commands import BaseTroveCLICommand


def info_command_factory(_):
    return EnvironmentCommand()


class EnvironmentCommand(BaseTroveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        cmd_parser = parser.add_parser(
            "env", help="Print relevant system environment info."
        )
        cmd_parser.set_defaults(func=info_command_factory)

    def run(self):
        info = {
            "`trove` version": version,
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
        }

        print("\nCopy-and-paste the text below in your GitHub issue.\n")
        print(self.format_dict(info))

        return info

    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"
