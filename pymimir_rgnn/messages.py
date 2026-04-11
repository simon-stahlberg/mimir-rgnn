import torch
import torch.nn as nn

from abc import abstractmethod
from typing import Any

from .bases import Encoder, MessageFunction
from .configs import HyperparameterConfig
from .encoders import get_relations_from_encoders
from .modules import MLP

class PredicateMLPMessages(MessageFunction):
    """Message function using separate MLPs for each predicate relation.

    This message function creates a separate multi-layer perceptron (MLP) for
    each relation type in the graph. This allows the model to learn different
    message computation strategies for different types of relations.
    """

    def __init__(self,
                 hparam_config: HyperparameterConfig,
                 input_spec: tuple[Encoder, ...]):
        """Initialize the predicate MLP message function.

        Args:
            config: The hyperparameter configuration containing embedding sizes.
            input_spec: The input specification to determine which relations exist.
        """
        super().__init__()
        self._embedding_size = hparam_config.embedding_size
        self._relation_mlps = nn.ModuleDict()
        self._cache: dict[str, Any] | None = None
        relations = get_relations_from_encoders(hparam_config.domain, input_spec)
        for relation_name, relation_arity in relations:
            input_size = relation_arity * hparam_config.embedding_size
            output_size = relation_arity * hparam_config.embedding_size
            if (input_size > 0) and (output_size > 0):
                self._relation_mlps[relation_name] = MLP(input_size, output_size)

    def setup(self, relations: dict[str, torch.Tensor]) -> None:
        """Cache per-forward relation metadata shared across layers.

        Args:
            relations: Dictionary mapping relation names to their argument indices.
        """
        active_relations: list[tuple[MLP, int, int]] = []
        output_indices_list: list[torch.Tensor] = []
        message_count = 0
        for relation_name, argument_indices in relations.items():
            if argument_indices.numel() == 0:
                continue
            relation_module: MLP = self._relation_mlps[relation_name]  # type: ignore
            next_message_count = message_count + argument_indices.numel()
            active_relations.append((relation_module, message_count, next_message_count))
            output_indices_list.append(argument_indices)
            message_count = next_message_count

        if len(output_indices_list) == 0:
            device = next(iter(relations.values())).device
            output_indices = torch.empty(0, dtype=torch.int, device=device)
        else:
            output_indices = torch.cat(output_indices_list, 0)

        self._cache = {
            'active_relations': tuple(active_relations),
            'message_count': message_count,
            'output_indices': output_indices,
        }

    def cleanup(self) -> None:
        """Clear cached relation metadata after the forward pass."""
        self._cache = None

    def _forward_relation(self, relation_name: str, argument_embeddings: torch.Tensor) -> torch.Tensor:
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

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute messages and indices for all relations.

        Args:
            node_embeddings: The current node embeddings.
            relations: Dictionary mapping relation names to their argument indices.

        Returns:
            Tuple of (messages, indices) for aggregation.
        """
        if self._cache is None:
            self.setup(relations)
        assert self._cache is not None, 'Relation cache must be available during forward.'
        message_count = self._cache['message_count']
        if message_count == 0:
            output_messages = node_embeddings.new_empty((0, self._embedding_size))
            output_indices: torch.Tensor = self._cache['output_indices']  # type: ignore
            return output_messages, output_indices

        output_indices: torch.Tensor = self._cache['output_indices']  # type: ignore
        gathered_embeddings = torch.index_select(node_embeddings, 0, output_indices)
        output_messages = node_embeddings.new_empty((message_count, self._embedding_size))
        for relation_module, start_index, end_index in self._cache['active_relations']:
            argument_embeddings = gathered_embeddings[start_index:end_index]
            argument_messages = relation_module(argument_embeddings.view(-1, relation_module.input_size))
            relation_output = (argument_embeddings.view_as(argument_messages) + argument_messages).view(-1, self._embedding_size)
            output_messages[start_index:end_index] = relation_output
        return output_messages, output_indices


class AttentionMessagesBase(MessageFunction):
    """Message function using TransformerEncoderLayer for parallel message computation.

    This message function uses a sequence of TransformerEncoderLayer provided by PyTorch
    to compute all messages in parallel. This approach treats each ground atom as a sequence
    where the first token represents the predicate symbol and the remaining tokens represent
    the objects.
    """

    def __init__(self,
                 hparam_config: HyperparameterConfig,
                 num_heads: int = 8,
                 num_layers: int = 2):
        """Initialize the attention message function.

        Args:
            hparam_config: The hyperparameter configuration containing embedding sizes.
            num_heads: The number of attention heads in each Transformer encoder layer.
            num_layers: The number of layers of the Transformer encoder.
        """
        super().__init__()
        self._embedding_size = hparam_config.embedding_size
        self._cache: dict[str, Any]

        # TransformerEncoderLayer for parallel message computation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hparam_config.embedding_size,
            nhead=num_heads,
            dim_feedforward=hparam_config.embedding_size,
            activation='relu',
            batch_first=True,
            dropout=0.0,
            layer_norm_eps=0.0
        )
        self._transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @abstractmethod
    def get_arity_of_relation(self, relation_name: str) -> int:
        """Get the arity of a given relation name."""
        pass

    @abstractmethod
    def get_index_of_relation(self, relation_name: str) -> int:
        """Get the unique index for a given relation name."""
        pass

    @abstractmethod
    def get_max_arity(self) -> int:
        """Get the maximum arity among all relations."""
        pass

    @abstractmethod
    def get_predicate_embeddings(self, predicate_ids: torch.Tensor) -> torch.Tensor:
        """Get the embeddings for the predicate tokens based on their IDs."""
        pass

    @abstractmethod
    def get_positional_embeddings(self, positions: torch.Tensor) -> torch.Tensor:
        """Get the positional embeddings for the given sequence length."""
        pass

    def setup(self, relations: dict[str, torch.Tensor]) -> None:
        """Pre-compute static parts that don't change across layers.

        Args:
            relations: Dictionary mapping relation names to their argument indices.
        """
        assert relations is not None, "Relations must be provided for setup."
        assert len(relations) > 0, "At least one relation must be provided for setup."

        max_sequence_length = self.get_max_arity() + 1  # One extra token for the predicate symbol

        # Determine device from the first relation tensor
        device = next(iter(relations.values())).device

        # Pre-compute all indices for batched node embedding selection
        # We always select max_arity embeddings per atom, using index 0 for padding
        all_node_indices = []
        all_predicate_ids = []
        all_padding_masks = []

        # Pre-compute indices for extracting messages after transformer
        message_indices = []
        output_indices = []

        # Track offsets for each relation to map back later
        relation_offset = 0

        for relation_name, argument_indices in relations.items():
            if argument_indices.numel() == 0:
                continue

            arity = self.get_arity_of_relation(relation_name)
            num_atoms = argument_indices.numel() // arity
            predicate_id = self.get_index_of_relation(relation_name)

            # Reshape to [num_atoms, arity]
            atom_indices = argument_indices.view(num_atoms, arity)

            # Pad to max_arity using index 0 (arbitrary object)
            if arity < max_sequence_length - 1:  # -1 because we add predicate token
                padding_size = (max_sequence_length - 1) - arity
                padding_indices = torch.zeros((num_atoms, padding_size), dtype=torch.long, device=device)
                padded_indices = torch.cat([atom_indices, padding_indices], dim=1)
            else:
                padded_indices = atom_indices

            all_node_indices.append(padded_indices)

            # Predicate IDs for this relation
            predicate_ids = torch.full((num_atoms,), predicate_id, dtype=torch.long, device=device)
            all_predicate_ids.append(predicate_ids)

            # Padding masks -- True for padding positions
            sequence_length = arity + 1  # +1 for predicate token
            padding_mask = torch.zeros(num_atoms, max_sequence_length, dtype=torch.bool, device=device)
            if sequence_length < max_sequence_length:
                padding_mask[:, sequence_length:] = True
            all_padding_masks.append(padding_mask)

            # Message extraction indices
            object_indices = torch.arange(1, sequence_length, dtype=torch.long, device=device)
            atom_offsets = torch.arange(num_atoms, dtype=torch.long, device=device) * max_sequence_length
            relation_message_indices = (object_indices.unsqueeze(0) + atom_offsets.unsqueeze(1)).reshape(-1)
            message_indices.append(relation_message_indices + relation_offset * max_sequence_length)

            # Output indices for aggregation
            output_indices.append(argument_indices)
            relation_offset += num_atoms

        # Concatenate all pre-computed tensors
        self._cache = {
            'node_indices': torch.cat(all_node_indices, dim=0),  # [total_atoms, max_arity]
            'predicate_ids': torch.cat(all_predicate_ids, dim=0),  # [total_atoms]
            'padding_masks': torch.cat(all_padding_masks, dim=0),  # [total_atoms, max_sequence_length]
            'message_indices': torch.cat(message_indices, dim=0),  # [total_messages]
            'output_indices': torch.cat(output_indices, dim=0),
            'total_atoms': relation_offset
        }

    def cleanup(self) -> None:
        """Clear cached data to free up memory.

        This is called after the message passing phase is complete.
        """
        assert hasattr(self, '_cache'), "No cache to clean up."
        del self._cache

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute messages using transformer attention for all relations in a single pass.

        Args:
            node_embeddings: The current node embeddings.
            relations: Dictionary mapping relation names to their argument indices.

        Returns:
            Tuple of (messages, indices) for aggregation.
        """
        # Assert setup was called and cache is available
        assert hasattr(self, '_cache'), "No cache found. Did you call setup()?"
        assert self._cache['total_atoms'] > 0, "No atoms in cache. Did you provide valid relations?"

        # Get all object embeddings in one go using pre-computed indices
        # The cached node_indices includes padding indices as well
        # Shape: [total_atoms, max_arity, embedding_size]
        node_indices: torch.Tensor = self._cache['node_indices']  # type: ignore
        all_object_embeddings = torch.index_select(node_embeddings, 0, node_indices.view(-1))
        all_object_embeddings = all_object_embeddings.view(self._cache['total_atoms'], self.get_max_arity(), self._embedding_size)  # type: ignore

        # Get all predicate embeddings
        # Shape: [total_atoms, 1, embedding_size]
        predicate_ids: torch.Tensor = self._cache['predicate_ids']  # type: ignore
        all_predicate_embeddings = self.get_predicate_embeddings(predicate_ids).unsqueeze(1)

        # Combine predicate and object embeddings
        # Shape: [total_atoms, max_sequence_length, embedding_size]
        sequence_embeddings = torch.cat([all_predicate_embeddings, all_object_embeddings], dim=1)

        # TODO: This can be moved to setup/cache.
        # Add positional embeddings -- use full max_sequence_length
        positions = torch.arange(self.get_max_arity() + 1, device=sequence_embeddings.device)
        positional_embeddings = self.get_positional_embeddings(positions).unsqueeze(0)  # [1, max_sequence_length, embedding_size]
        sequence_embeddings = sequence_embeddings + positional_embeddings

        # Apply transformer to all sequences in one pass
        padding_masks: torch.Tensor = self._cache['padding_masks']  # type: ignore
        transformed_embeddings = self._transformer.forward(sequence_embeddings, src_key_padding_mask=padding_masks).view(-1, self._embedding_size)

        # Extract object messages using pre-computed indices
        message_indices: torch.Tensor = self._cache['message_indices']  # type: ignore
        output_messages = transformed_embeddings.index_select(0, message_indices)

        output_indices: torch.Tensor = self._cache['output_indices']  # type: ignore
        return output_messages, output_indices


class AttentionMessages(AttentionMessagesBase):
    """Message function using TransformerEncoderLayer for parallel message computation.

    This message function uses a sequence of TransformerEncoderLayer provided by PyTorch
    to compute all messages in parallel. This approach treats each ground atom as a sequence
    where the first token represents the predicate symbol and the remaining tokens represent
    the objects.
    """

    def __init__(self,
                 hparam_config: HyperparameterConfig,
                 input_spec: tuple[Encoder, ...],
                 num_heads: int = 8,
                 num_layers: int = 2):
        """Initialize the attention message function.

        Args:
            hparam_config: The hyperparameter configuration containing embedding sizes.
            input_spec: The input specification to determine which relations exist.
            num_heads: The number of attention heads in each Transformer encoder layer.
            num_layers: The number of layers of the Transformer encoder.
        """
        super().__init__(hparam_config, num_heads, num_layers)
        self._relation_arities = dict(get_relations_from_encoders(hparam_config.domain, input_spec))
        assert len(self._relation_arities) > 0, "No relations found in input specification."
        self._max_arity = max(self._relation_arities.values())
        self._predicate_to_idx = {name: idx for idx, name in enumerate(sorted(self._relation_arities.keys()))}
        self._num_predicates = len(self._predicate_to_idx)
        self._predicate_embeddings = nn.Embedding(self._num_predicates, hparam_config.embedding_size)
        self._positional_embeddings = nn.Embedding(self._max_arity + 1, hparam_config.embedding_size)

    def get_arity_of_relation(self, relation_name: str) -> int:
        return self._relation_arities[relation_name]

    def get_index_of_relation(self, relation_name: str) -> int:
        return self._predicate_to_idx[relation_name]

    def get_max_arity(self) -> int:
        return self._max_arity

    def get_predicate_embeddings(self, predicate_ids: torch.Tensor) -> torch.Tensor:
        return self._predicate_embeddings(predicate_ids)

    def get_positional_embeddings(self, positions: torch.Tensor) -> torch.Tensor:
        return self._positional_embeddings(positions)
