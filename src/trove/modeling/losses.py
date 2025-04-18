import torch
import torch.nn.functional as F
from torch import nn

from ..logging_utils import get_logger_with_config
from .loss_base import RetrievalLoss
from .model_args import ModelArguments

logger, logging_conf = get_logger_with_config("trove")


class InfoNCELoss(RetrievalLoss):
    _alias = "infonce"

    def __init__(self, args: ModelArguments, **kwargs) -> None:
        """Implements InfoNCE loss."""
        super().__init__()

        temp_init_value = float(args.temperature)
        if args.temperature_learnable:
            msg = (
                "In ModelArguments, 'temperature_learnable' is set to True."
                " This feature is incomplete and the learned temperature value is not saved to checkpoints."
                " So you cannot access it later."
            )
            logger.warning(msg)
            _temp_tensor = torch.tensor(temp_init_value)
            if "dtype" in kwargs:
                _temp_tensor = _temp_tensor.to(kwargs["dtype"])
            if "device" in kwargs:
                _temp_tensor = _temp_tensor.to(kwargs["device"])
            self.temperature = nn.Parameter(_temp_tensor)
        else:
            self.temperature = temp_init_value

        reduction = kwargs.get("reduction", "mean")
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculates the loss given the similarity scores between query and passages.

        The ``logits`` argument contains the similarity scores between queries and passages for the entire batch.
        For each query in the batch, there are K corresponding passages, where 1 is positive and k-1 are negative passages.
        For each query the positive passages comes first before negative passages, e.g., ``passages_for_q1 = [pos, neg, neg, neg, neg, ..., neg]``

        To use in-batch negatives, the passages for all queries are concatenated in a list.
        I.e., ``all_passages = passages_for_q1 + passages_for_q2 + ... + passages_for_qn``
        As a result, the index of the positive passage for query `i` is `i * K` where `K` is the number of passages per query.

        Args:
            logits (torch.Tensor): The similarity scores between queries and passages. shape: `[NUM_QUERIES, NUM_PASSAGES]`
            **kwargs: Not used. Just to make the signature compatible with other losses.

        Returns:
            the InfoNCE loss value.
        """
        target = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        passages_per_query = logits.size(1) // logits.size(0)
        # index of the positive passage in the combined list of documents
        target = target * passages_per_query
        loss = self.cross_entropy(logits / self.temperature, target)
        return loss


class KLDivergenceLoss(RetrievalLoss):
    _alias = "kl"

    def __init__(self, args: ModelArguments, **kwargs) -> None:
        """Implements KL divergence loss."""
        super().__init__()

        temp_init_value = float(args.temperature)
        if args.temperature_learnable:
            msg = (
                "In ModelArguments, 'temperature_learnable' is set to True."
                " This feature is incomplete and the learned temperature value is not saved to checkpoints."
                " So you cannot access it later."
            )
            logger.warning(msg)
            _temp_tensor = torch.tensor(temp_init_value)
            if "dtype" in kwargs:
                _temp_tensor = _temp_tensor.to(kwargs["dtype"])
            if "device" in kwargs:
                _temp_tensor = _temp_tensor.to(kwargs["device"])
            self.temperature = nn.Parameter(_temp_tensor)
        else:
            self.temperature = temp_init_value

    def forward(
        self, logits: torch.Tensor, label: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Calculates the loss given the similarity scores between query and passages."""
        if label.size(1) != logits.size(1):
            label = torch.block_diag(*torch.chunk(label, label.shape[0]))

        preds = F.log_softmax(logits / self.temperature, dim=1)
        targets = F.log_softmax(label.double(), dim=1)
        loss = F.kl_div(
            input=preds, target=targets, log_target=True, reduction="batchmean"
        )
        return loss
