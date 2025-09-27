import pymimir as mm
import torch

from abc import ABC, abstractmethod
from typing import Any


class EncodedLists:
    """Intermediate representation for storing graph encoding data as lists.

    This class holds the graph structure and node information in list format
    before conversion to tensors for use in the graph neural network.
    """

    def __init__(self):
        """Initialize empty encoded lists for graph representation."""
        self.flattened_relations: dict[str, list[int]] = {}
        self.node_count: int = 0
        self.node_sizes: list[int] = []
        self.object_sizes: list[int] = []
        self.object_indices: list[int] = []
        self.action_sizes: list[int] = []
        self.action_indices: list[int] = []


class EncodedTensors:
    """Tensor representation for graph encoding data.

    This class holds the graph structure and node information as tensors
    ready for use in the graph neural network computations.
    """

    def __init__(self):
        """Initialize empty encoded tensors for graph representation."""
        self.flattened_relations: dict[str, torch.Tensor] = {}
        self.node_count: int = 0
        self.node_sizes: torch.Tensor = torch.LongTensor()
        self.object_sizes: torch.Tensor = torch.LongTensor()
        self.object_indices: torch.Tensor = torch.LongTensor()
        self.action_sizes: torch.Tensor = torch.LongTensor()
        self.action_indices: torch.Tensor = torch.LongTensor()


class EncodingContext():
    def __init__(self, problem: mm.Problem, id_offset: int) -> None:
        self.problem = problem
        self.id_offset = id_offset

        # Map from object indices to global node IDs
        objects = problem.get_objects() + problem.get_domain().get_constants()
        self.object_index_to_id: dict[int, int] = { obj.get_index(): i + id_offset for i, obj in enumerate(objects) }
        self.action_ids: list[int] = []

    def get_object_id(self, object_index: int) -> int:
        return self.object_index_to_id[object_index]

    def new_action_id(self) -> int:
        action_id = self.id_offset + len(self.object_index_to_id) + len(self.action_ids)
        self.action_ids.append(action_id)
        return action_id

    def get_object_ids(self) -> list[int]:
        return list(self.object_index_to_id.values())

    def get_object_count(self) -> int:
        return len(self.object_index_to_id)

    def get_action_ids(self) -> list[int]:
        return self.action_ids

    def get_action_count(self) -> int:
        return len(self.action_ids)

    def get_node_count(self) -> int:
        return len(self.object_index_to_id) + len(self.action_ids)


class Encoder(ABC):
    """Base class for encoders that transform PDDL structures into graph neural network inputs.

    Encoders are responsible for converting PDDL planning structures (states, goals, actions)
    into a relational graph representation suitable for graph neural networks.
    """

    @abstractmethod
    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get the relations this encoder contributes to the graph encoding.

        Args:
            domain: The PDDL domain containing predicates and actions.

        Returns:
            List of (relation_name, arity) pairs representing the relations
            this encoder will add to the graph structure.
        """
        pass

    @abstractmethod
    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        """Encode the input value into the intermediate graph representation.

        Args:
            input_value: The input data to encode (state, goal, actions, etc.).
            state: The current planning state for context.
            encoding: The EncodedLists object to populate with graph structure.
            context: Context for tracking local additions to the encoding.
        """
        pass


class Decoder(ABC, torch.nn.Module):
    """Base class for decoders that implement readout logic from node embeddings.

    Decoders extract meaningful outputs from the node embeddings produced by
    the graph neural network, such as action values or object embeddings.
    """

    def __init__(self):
        """Initialize the decoder as a PyTorch module."""
        super().__init__()

    @abstractmethod
    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> Any:
        """Perform readout from node embeddings to produce output values.

        Args:
            node_embeddings: The node embeddings from the graph neural network.
            encoding: The encoding information on how to interpret the node embeddings.

        Returns:
            The decoded output (tensors, lists of tensors, etc.).
        """
        pass


class AggregationFunction(ABC, torch.nn.Module):
    """Base class for aggregation functions used in graph neural networks.

    Aggregation functions combine messages from multiple sources during
    the message passing phase of graph neural network computation.
    """

    def __init__(self):
        """Initialize the aggregation function as a PyTorch module."""
        super().__init__()

    @abstractmethod
    def forward(self, node_embeddings: torch.Tensor, messages: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Aggregate messages based on the provided indices.

        Args:
            node_embeddings: The current node embeddings.
            messages: The messages to aggregate.
            indices: The indices indicating how to group messages for aggregation.

        Returns:
            The aggregated messages with the same shape as node_embeddings.
        """
        pass


class MessageFunction(ABC, torch.nn.Module):
    """Base class for message functions used in relational graph neural networks.

    Message functions compute messages between nodes based on their relations
    during the message passing phase of graph neural network computation.
    """

    def __init__(self):
        """Initialize the message function as a PyTorch module."""
        super().__init__()

    def setup(self, relations: dict[str, torch.Tensor]) -> None:
        """Optional setup before message computation.

        Args:
            relations: Dictionary mapping relation names to their argument indices.
        """
        pass

    @abstractmethod
    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute messages and indices for all relations.

        Args:
            node_embeddings: The current node embeddings.
            relations: Dictionary mapping relation names to their argument indices.

        Returns:
            Tuple of (messages, indices) for aggregation.
        """
        pass

    def cleanup(self) -> None:
        """Optional cleanup after message computation."""
        pass


class UpdateFunction(ABC, torch.nn.Module):
    """Base class for update functions used in graph neural networks.

    Update functions compute new node embeddings based on the current embeddings
    and the aggregated messages received during message passing.
    """

    def __init__(self):
        """Initialize the update function as a PyTorch module."""
        super().__init__()

    @abstractmethod
    def forward(self, node_embeddings: torch.Tensor, aggregated_messages: torch.Tensor) -> torch.Tensor:
        """Update node embeddings based on the aggregated messages.

        Args:
            node_embeddings: The current node embeddings.
            aggregated_messages: The aggregated messages to use for the update.

        Returns:
            The updated node embeddings with the same shape as the input embeddings.
        """
        pass
