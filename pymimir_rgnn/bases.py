import pymimir as mm
import torch

from abc import ABC, abstractmethod
from typing import Any


class EncodedLists:
    def __init__(self):
        self.flattened_relations: dict[str, list[int]] = {}
        self.node_count: int = 0
        self.node_sizes: list[int] = []
        self.object_sizes: list[int] = []
        self.object_indices: list[int] = []
        self.action_sizes: list[int] = []
        self.action_indices: list[int] = []


class EncodedTensors:
    def __init__(self):
        self.flattened_relations: dict[str, torch.Tensor] = {}
        self.node_count: int = 0
        self.node_sizes: torch.Tensor = torch.LongTensor()
        self.object_sizes: torch.Tensor = torch.LongTensor()
        self.object_indices: torch.Tensor = torch.LongTensor()
        self.action_sizes: torch.Tensor = torch.LongTensor()
        self.action_indices: torch.Tensor = torch.LongTensor()


class Encoder(ABC):
    """Base class for encoders that transform PDDL structures into graph neural network inputs."""

    @abstractmethod
    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get the relations this encoder contributes to the graph encoding, expressed as a list of name-arity pairs."""
        pass

    @abstractmethod
    def encode(self, input_value: Any, encoding: 'EncodedLists', state: mm.State) -> int:
        """
        Encode the input value into the representation.

        Args:
            input_value: The input data to encode (state, goal, actions, etc.)
            encoding: The EncodedLists object to populate
            state: The current planning state for context

        Returns:
            Number of nodes added to the graph
        """
        pass


class Decoder(ABC, torch.nn.Module):
    """Base class for decoders that implement readout logic from node embeddings."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> Any:
        """
        Perform readout from node embeddings to produce output values.

        Args:
            node_embeddings: The node embeddings from the graph neural network
            encoding: The encoding information on how to interpret the node embeddings

        Returns:
            The decoded output (tensors, lists of tensors, etc.)
        """
        pass


class AggregationFunction(ABC, torch.nn.Module):
    """Base class for aggregation functions used in graph neural networks."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, node_embeddings: torch.Tensor, messages: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Aggregate messages based on the provided indices.

        Args:
            node_embeddings: The current node embeddings
            messages: The messages to aggregate
            indices: The indices indicating how to group messages for aggregation

        Returns:
            The aggregated messages
        """
        pass


class MessageFunction(ABC, torch.nn.Module):
    """Base class for message functions used in graph neural networks."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, relation_name: str, argument_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute messages for a given relation and its argument embeddings.

        Args:
            relation_name: The name of the relation for which to compute messages
            argument_embeddings: The embeddings of the arguments involved in the relation

        Returns:
            The computed messages
        """
        pass


class UpdateFunction(ABC, torch.nn.Module):
    """Base class for update functions used in graph neural networks."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, node_embeddings: torch.Tensor, aggregated_messages: torch.Tensor) -> torch.Tensor:
        """
        Update node embeddings based on the aggregated messages.

        Args:
            node_embeddings: The current node embeddings
            aggregated_messages: The aggregated messages to use for the update

        Returns:
            The updated node embeddings
        """
        pass
