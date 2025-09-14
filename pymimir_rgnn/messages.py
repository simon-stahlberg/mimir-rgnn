import torch
import torch.nn as nn

from .bases import Encoder
from .configs import HyperparameterConfig
from .encoders import get_relations_from_encoders
from .modules import MLP

class PredicateMLPMessages:
    """Message function using separate MLPs for each predicate relation.

    This message function creates a separate multi-layer perceptron (MLP) for
    each relation type in the graph. This allows the model to learn different
    message computation strategies for different types of relations.
    """

    def __init__(self,
                 config: HyperparameterConfig,
                 input_spec: tuple[Encoder, ...]):
        """Initialize the predicate MLP message function.

        Args:
            config: The hyperparameter configuration containing embedding sizes.
            input_spec: The input specification to determine which relations exist.
        """
        self._relation_mlps = nn.ModuleDict()
        relations = get_relations_from_encoders(config.domain, input_spec)
        for relation_name, relation_arity in relations:
            input_size = relation_arity * config.embedding_size
            output_size = relation_arity * config.embedding_size
            if (input_size > 0) and (output_size > 0):
                self._relation_mlps[relation_name] = MLP(input_size, output_size)

    def forward(self, relation_name: str, argument_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute messages for a specific relation using its dedicated MLP.

        Args:
            relation_name: The name of the relation for which to compute messages.
            argument_embeddings: The embeddings of the arguments involved in the relation.

        Returns:
            The computed messages with the same shape as argument_embeddings.

        Raises:
            ValueError: If no MLP is found for the given relation name.
        """
        if relation_name not in self._relation_mlps:
            raise ValueError(f"No MLP found for relation '{relation_name}'")
        relation_module: MLP = self._relation_mlps[relation_name]  # type: ignore
        messages = relation_module(argument_embeddings.view(-1, relation_module.input_size))
        return messages
