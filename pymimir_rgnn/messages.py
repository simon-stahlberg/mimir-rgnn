import torch
import torch.nn as nn

from .bases import Encoder
from .configs import HyperparameterConfig
from .encoders import get_relations_from_encoders
from .modules import MLP

class PredicateMLPMessages:
    def __init__(self,
                 config: HyperparameterConfig,
                 input_spec: tuple[Encoder, ...]):
        self._relation_mlps = nn.ModuleDict()
        relations = get_relations_from_encoders(config.domain, input_spec)
        for relation_name, relation_arity in relations:
            input_size = relation_arity * config.embedding_size
            output_size = relation_arity * config.embedding_size
            if (input_size > 0) and (output_size > 0):
                self._relation_mlps[relation_name] = MLP(input_size, output_size)

    def forward(self, relation_name: str, argument_embeddings: torch.Tensor) -> torch.Tensor:
        if relation_name not in self._relation_mlps:
            raise ValueError(f"No MLP found for relation '{relation_name}'")
        relation_module: MLP = self._relation_mlps[relation_name]  # type: ignore
        messages = relation_module(argument_embeddings.view(-1, relation_module.input_size))
        return messages
