from typing import Dict, Optional

import torch

from . import modeling_utils
from .pretrained_retriever import PretrainedRetriever, RetrieverOutput


class BiEncoderRetriever(PretrainedRetriever):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        query: Optional[Dict[str, torch.Tensor]] = None,
        passage: Optional[Dict[str, torch.Tensor]] = None,
        label: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = True,
        **kwargs,
    ) -> RetrieverOutput:
        """Encodes query and passages and potentially calculate similarity scores and loss.

        Args:
            query (Optional[Dict[str, torch.Tensor]]): tokenized query.
            passage (Optional[Dict[str, torch.Tensor]]): tokenized passages.
            label (Optional[torch.Tensor]): Relevancy scores of the corresponding passages for each query.
                If there are k documents for each query, this is a 2D tensor of shape `[NUM_QUERIES, k]`
            return_loss (Optional[bool]): if true, calculate the loss value.
            **kwargs: unused keyword arguments are passed to the ``forward()`` method of the loss module.

        Returns:
            If possible, query and passage embeddings as well as similarity and loss scores.
        """
        # There is nothing to do if both query and passages are missing
        assert query is not None or passage is not None

        if query is not None:
            query = self.encoder.encode_query(inputs=query)

        if passage is not None:
            passage = self.encoder.encode_passage(inputs=passage)

        if query is None or passage is None:
            # if one of the passage or query is missing, we cannot do much beyond embedding them.
            return RetrieverOutput(query=query, passage=passage)

        if self.training and self.is_ddp:
            # Gather the query and passage embeddings from all the processes in a distributed environment.
            # After this if block, 'query' will contain the embedding of all the queries across all processes
            # Similarly, 'passage' will contain the embeddings of all passage from all processes
            # Also, 'label' is the value of input argument 'label' from all processes concatenated along dim 0
            query = modeling_utils.gather_tensors(
                tensor_obj=query, world_size=self.world_size, rank=self.process_rank
            )
            passage = modeling_utils.gather_tensors(
                tensor_obj=passage, world_size=self.world_size, rank=self.process_rank
            )
            label = modeling_utils.gather_tensors(
                tensor_obj=label, world_size=self.world_size, rank=self.process_rank
            )

        logits = self.similarity_fn(query=query, passage=passage)
        if not return_loss or self.loss is None:
            return RetrieverOutput(query=query, passage=passage, logits=logits)

        loss = self.loss(logits=logits, label=label, **kwargs)

        return RetrieverOutput(loss=loss, logits=logits, query=query, passage=passage)
