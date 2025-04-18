"""Base loss for information retrieval."""

from typing import Optional

import torch
from torch import nn

from .model_args import ModelArguments


class RetrievalLoss(nn.Module):
    """Base class for loss functions that can automatically detect the correct subclass to
    instantiate.

    To add a new loss function, create a new class that inherits from :class:`RetrievalLoss`.
    The users can instantiate the new loss function by its name.
    For example, if you do::

        class MyLoss(RetrievalLoss):
            _alias = 'foo_loss'
            ...

    Then users can use ``MyLoss`` if ``ModelArguments.loss == "MyLoss"`` or ``ModelArguments.loss == "foo_loss"``.

    **Do not overwrite or modify** ``cls._loss_registry`` class attribute.
    """

    # An alternative name (other than class name) that others can use to refer to your loss function.
    _alias: Optional[str] = None
    _loss_registry = dict()

    def __init_subclass__(cls, **kwargs):
        """Store a pointer to new loss functions that inherit from ``RetrievalLoss``."""
        super().__init_subclass__(**kwargs)
        assert cls.__name__ not in cls._loss_registry
        cls._loss_registry[cls.__name__] = cls
        if cls._alias is not None:
            assert cls._alias not in cls._loss_registry
            cls._loss_registry[cls._alias] = cls

    @classmethod
    def available_losses(cls) -> None:
        """Prints a list of all available loss functions and their aliases."""
        all_losses = list()
        for l in cls._loss_registry.values():
            if l not in all_losses:
                all_losses.append(l)
        if len(all_losses) == 0:
            print("Did not find any loss functions.")
            return
        print('Available losses in ("name", "alias") format:')
        for l in all_losses:
            print(f'("{l.__name__}", "{l._alias}")')

    @classmethod
    def from_model_args(cls, args: ModelArguments, **kwargs):
        """Instantiate the correct subclass of BaseLoss based on ``args.loss``.

        Args:
            args (ModelArguments): ``args.loss`` is used to detect the correct loss function subclass.
                ``args`` is also passed to the constructor of the target subclass.
            **kwargs: is passed to the constructor of the target subclass.

        Returns:
            An instance of the specified loss subclass. Or ``None`` if ``args.loss is None``
        """
        if args.loss is None:
            return None
        loss_cls = cls._loss_registry[args.loss]
        return loss_cls(args=args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(
        self, logits: torch.Tensor, label: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Calculates the loss given the similarity scores between query and passages.

        The ``logits`` argument contains the similarity score between queries and
        passages (including in-batch negatives) for the entire batch.
        The shape of the ``logits`` argument is `[NUM_QUERIES, NUM_QUERIES * DOCS_PER_QUERY]`.
        The documents are organized sequentially. Basically, the related docs for the i_th query
        are ``docs[i * DOCS_PER_QUERY: (i + 1) * DOCS_PER_QUERY]`` and the rest are in-batch negatives.

        The ``label`` argument contains the groundtruth relevancy level between queries and documents for the entire
        batch. ``label`` does NOT include in-batch negatives. ``label`` tensor is of shape `[NUM_QUERIES, DOCS_PER_QUERY]`.
        We assign in-batch negatives, a relevancy level of 0 and extend the ``label`` argument accordingly.

        You can use the following snippet of code to make sure ``logits`` and ``labels`` are of the same shape and give
        a label of zero (0) to in-batch negatives.

        .. code:: python

            if label.size(1) != logits.size(1):
                label = torch.block_diag(*torch.chunk(label, label.shape[0]))

        Args:
            logits (torch.Tensor): The similarity scores between queries and passages (including in-batch negatives).
                shape: `[NUM_QUERIES, NUM_QUERIES * DOCS_PER_QUERY]`
            label (torch.Tensor): The groundtruth relevancy level between queries and passages (excluding in-batch negatives).
                shape: `[NUM_QUERIES, DOCS_PER_QUERY]`.
            **kwargs: Not used. Just to make the signature compatible with other losses.

        Returns:
            the loss value.
        """
        raise NotImplementedError
