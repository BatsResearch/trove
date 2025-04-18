"""From here:

https://github.com/huggingface/datasets/blob/a1b5a32365cd2c6631fa4ad236a881f9f62bf32b/src/datasets/commands/__init__.py
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser


class BaseTroveCLICommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()
