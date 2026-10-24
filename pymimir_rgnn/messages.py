import torch
import torch.nn as nn

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from .bases import Encoder, MessageFunction
from .configs import HyperparameterConfig
from .encoders import get_relations_from_encoders
from .modules import MLP, SparseMLP
from .utils import ternarize

@dataclass
class _RelationGroup:
    """Relations of one arity whose MLPs are computed together, each padded to the group's largest relation."""
    num_relations: int
    width: int
    input_size: int
    padded_indices: torch.Tensor
    rows: torch.Tensor
    inner_weight: torch.Tensor
    inner_bias: torch.Tensor
    outer_weight: torch.Tensor
    outer_bias: torch.Tensor


class PredicateMLPMessages(MessageFunction):
    """Message function using separate MLPs for each predicate relation.

    This message function creates a separate multi-layer perceptron (MLP) for
    each relation type in the graph. This allows the model to learn different
    message computation strategies for different types of relations.

    Relations of the same arity are computed together with batched matrix
    products, which launches a few kernels per arity instead of several per
    relation.
    """

    supports_replay = True

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
        self._ternarize_messages = hparam_config.ternarize_messages
        self._relation_mlps = nn.ModuleDict()
        relations = get_relations_from_encoders(hparam_config.domain, input_spec)
        for relation_name, relation_arity in relations:
            input_size = relation_arity * hparam_config.embedding_size
            output_size = relation_arity * hparam_config.embedding_size
            if (input_size > 0) and (output_size > 0):
                self._relation_mlps[relation_name] = MLP(
                    input_size,
                    output_size,
                    hidden_size=hparam_config.message_hidden_size,
                )
        self._grouped: tuple[list[_RelationGroup], torch.Tensor] | None = None

    def setup(self, relations: dict[str, torch.Tensor]) -> None:
        """Group the relations once for all layers.

        Args:
            relations: Dictionary mapping relation names to their argument indices.
        """
        self._grouped = self._group_relations(relations)

    def _group_relations(self, relations: dict[str, torch.Tensor]) -> tuple[list[_RelationGroup], torch.Tensor]:
        """Group the nonempty relations by arity and stack their MLPs' parameters.

        Args:
            relations: Dictionary mapping relation names to their argument indices.

        Returns:
            The groups, and the receiving node of each message in the order the groups produce them.

        Raises:
            ValueError: If no MLP is found for a nonempty relation.
        """
        groups = self._split_by_arity(relations)
        # The messages are in the order of the groups, and within a group in the order of its relations; a group's
        # argument indices are a slice of the output indices.
        output_indices = torch.cat([argument_indices for _, members in groups for _, argument_indices in members])
        rows, atoms_per_group = self._padded_rows(groups, output_indices.device)
        relation_groups: list[_RelationGroup] = []
        first_atom = 0
        first_index = 0
        for (input_size, members), num_atoms in zip(groups, atoms_per_group):
            arity = input_size // self._embedding_size
            width = members[0][1].numel() // arity
            group_rows = rows[first_atom:first_atom + num_atoms]
            argument_indices = output_indices[first_index:first_index + num_atoms * arity].view(-1, arity)
            # Padding refers to node 0; the padded rows are computed but never selected.
            padded_indices = output_indices.new_zeros((len(members) * width, arity)).index_copy_(0, group_rows, argument_indices)
            mlps: list[MLP] = [self._relation_mlps[relation_name] for relation_name, _ in members]  # type: ignore
            relation_groups.append(_RelationGroup(
                num_relations=len(members),
                width=width,
                input_size=input_size,
                padded_indices=padded_indices.view(-1),
                rows=group_rows,
                inner_weight=torch.stack([mlp._inner.weight for mlp in mlps]).transpose(1, 2),
                inner_bias=torch.stack([mlp._inner.bias for mlp in mlps]).unsqueeze(1),
                outer_weight=torch.stack([mlp._outer.weight for mlp in mlps]).transpose(1, 2),
                outer_bias=torch.stack([mlp._outer.bias for mlp in mlps]).unsqueeze(1),
            ))
            first_atom += num_atoms
            first_index += num_atoms * arity
        return relation_groups, output_indices

    def _split_by_arity(self, relations: dict[str, torch.Tensor]) -> list[tuple[int, list[tuple[str, torch.Tensor]]]]:
        """The nonempty relations in groups of one arity, as (input size, [(relation name, argument indices)]).

        Within an arity the relations go largest first, split so that padding adds at most half to the rows of a
        group.
        """
        by_input_size: dict[int, list[tuple[str, torch.Tensor]]] = {}
        for relation_name, argument_indices in relations.items():
            if argument_indices.numel() > 0:
                if relation_name not in self._relation_mlps:
                    raise ValueError(f"No MLP found for relation '{relation_name}'")
                relation_module: MLP = self._relation_mlps[relation_name]  # type: ignore
                by_input_size.setdefault(relation_module.input_size, []).append((relation_name, argument_indices))
        groups: list[tuple[int, list[tuple[str, torch.Tensor]]]] = []
        for input_size, members in by_input_size.items():
            arity = input_size // self._embedding_size
            members.sort(key=lambda member: member[1].numel(), reverse=True)
            group: list[tuple[str, torch.Tensor]] = []
            group_atoms = 0
            for member in members:
                num_atoms = member[1].numel() // arity
                if group and 2 * (len(group) + 1) * (group[0][1].numel() // arity) > 3 * (group_atoms + num_atoms):
                    groups.append((input_size, group))
                    group, group_atoms = [], 0
                group.append(member)
                group_atoms += num_atoms
            groups.append((input_size, group))
        return groups

    def _padded_rows(self, groups: list[tuple[int, list[tuple[str, torch.Tensor]]]],
                     device: torch.device) -> tuple[torch.Tensor, list[int]]:
        """The padded row of each atom, for all groups at once, and the number of atoms of each group.

        Relation r of a group starts at row r * width of the group's padded rows, rather than after the atoms before
        it, where width is the number of atoms of the group's largest relation.
        """
        shifts: list[int] = []
        counts: list[int] = []
        atoms_per_group: list[int] = []
        total_atoms = 0
        for input_size, members in groups:
            arity = input_size // self._embedding_size
            width = members[0][1].numel() // arity
            first_atom = 0
            for row, (_, argument_indices) in enumerate(members):
                shifts.append(row * width - first_atom - total_atoms)
                counts.append(argument_indices.numel() // arity)
                first_atom += counts[-1]
            atoms_per_group.append(first_atom)
            total_atoms += first_atom
        host = torch.tensor(shifts + counts, dtype=torch.long)
        if device.type == 'cuda':
            # One copy from pinned memory, which does not wait for the device.
            host = host.pin_memory()
        host = host.to(device, non_blocking=True)
        rows = torch.arange(total_atoms, device=device) \
            + torch.repeat_interleave(host[:len(shifts)], host[len(shifts):], output_size=total_atoms)
        return rows, atoms_per_group

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute messages and indices for all relations.

        Args:
            node_embeddings: The current node embeddings.
            relations: Dictionary mapping relation names to their argument indices; if `setup` was called, the same
                as given to it.

        Returns:
            Tuple of (messages, indices) for aggregation.
        """
        groups, output_indices = self._grouped if self._grouped is not None else self._group_relations(relations)
        output_messages_list: list[torch.Tensor] = []
        for group in groups:
            argument_embeddings = node_embeddings.index_select(0, group.padded_indices)
            argument_embeddings = argument_embeddings.view(group.num_relations, group.width, group.input_size)
            hidden = nn.functional.mish(torch.baddbmm(group.inner_bias, argument_embeddings, group.inner_weight))
            argument_messages = torch.baddbmm(group.outer_bias, hidden, group.outer_weight)
            output_messages = (argument_embeddings + argument_messages).view(-1, group.input_size)
            output_messages_list.append(output_messages.index_select(0, group.rows).view(-1, self._embedding_size))
        output_messages = torch.cat(output_messages_list, 0)
        if self._ternarize_messages:
            output_message_logits = output_messages
            output_messages = ternarize(output_message_logits)
            self._record_ternary_messages(output_message_logits, output_messages)
        return output_messages, output_indices

    def cleanup(self) -> None:
        """Release the groups of the latest `setup`."""
        self._grouped = None


class PredicateLinearMessages(MessageFunction):
    """Message function using separate linear maps for each predicate relation.

    This is a linear-only counterpart to PredicateMLPMessages. Each relation
    type gets its own linear layer from the concatenated argument embeddings to
    relation-position messages. No argument residual is added.
    """

    supports_replay = True

    def __init__(self,
                 hparam_config: HyperparameterConfig,
                 input_spec: tuple[Encoder, ...]):
        """Initialize the relation-specific linear message function.

        Args:
            hparam_config: Hyperparameter configuration.
            input_spec: Encoder tuple that determines which relations exist.
        """
        super().__init__()
        self._embedding_size = hparam_config.embedding_size
        self._ternarize_messages = hparam_config.ternarize_messages
        self._relation_linears = nn.ModuleDict()
        relations = get_relations_from_encoders(hparam_config.domain, input_spec)
        for relation_name, relation_arity in relations:
            input_size = relation_arity * hparam_config.embedding_size
            output_size = relation_arity * hparam_config.embedding_size
            if (input_size > 0) and (output_size > 0):
                self._relation_linears[relation_name] = nn.Linear(input_size, output_size)

    def _forward_relation(self, relation_name: str, argument_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute messages for a specific relation using its linear map."""
        if relation_name not in self._relation_linears:
            raise ValueError(f"No linear layer found for relation '{relation_name}'")
        relation_module: nn.Linear = self._relation_linears[relation_name]  # type: ignore
        return relation_module(argument_embeddings.view(-1, relation_module.in_features))

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute linear messages and indices for all relations."""
        output_messages_list: list[torch.Tensor] = []
        output_indices_list: list[torch.Tensor] = []
        for relation_name, argument_indices in relations.items():
            if argument_indices.numel() > 0:
                argument_embeddings = torch.index_select(node_embeddings, 0, argument_indices)
                argument_messages = self._forward_relation(relation_name, argument_embeddings)
                output_messages = argument_messages.view(-1, self._embedding_size)
                output_messages_list.append(output_messages)
                output_indices_list.append(argument_indices)
        output_messages = torch.cat(output_messages_list, 0)
        output_indices = torch.cat(output_indices_list, 0)
        if self._ternarize_messages:
            output_message_logits = output_messages
            output_messages = ternarize(output_message_logits)
            self._record_ternary_messages(output_message_logits, output_messages)
        return output_messages, output_indices


class SparseMLPMessages(MessageFunction):
    """PredicateMLPMessages variant with hard top-k gated linear relation maps.

    Each output feature is a linear function of at most k original input
    features. The implementation still uses dense PyTorch linear operations,
    with hard top-k masks applied to the dense weight matrices. The top-k
    margin penalty encourages stable separation between selected and rejected
    gate logits.

    Args:
        hparam_config: Hyperparameter configuration.
        input_spec: Encoder tuple that determines which relations exist.
        k: Maximum number of active input connections per output feature.
        tau: Temperature for the straight-through sigmoid surrogate.
        gate_mode: Training-time hard top-k mode. Evaluation always uses
            deterministic top-k.
    """

    def __init__(self,
                 hparam_config: HyperparameterConfig,
                 input_spec: tuple[Encoder, ...],
                 k: int,
                 tau: float = 1.0,
                 gate_mode: Literal["gumbel_topk", "deterministic_topk"] = "gumbel_topk"):
        super().__init__()
        self._embedding_size = hparam_config.embedding_size
        self._ternarize_messages = hparam_config.ternarize_messages
        self._relation_mlps = nn.ModuleDict()
        relations = get_relations_from_encoders(hparam_config.domain, input_spec)
        for relation_name, relation_arity in relations:
            input_size = relation_arity * hparam_config.embedding_size
            output_size = relation_arity * hparam_config.embedding_size
            if (input_size > 0) and (output_size > 0):
                self._relation_mlps[relation_name] = SparseMLP(input_size, output_size, k=k, tau=tau, gate_mode=gate_mode)

    def topk_margin_penalty(self, margin: float) -> torch.Tensor:
        """Mean SparseMLP top-k margin penalty across relation modules."""
        penalties = [mlp.topk_margin_penalty(margin) for mlp in self._relation_mlps.values()]  # type: ignore[attr-defined, operator]
        if penalties:
            return torch.stack(penalties).mean()
        return torch.tensor(0.0)

    def _forward_relation(self, relation_name: str, argument_embeddings: torch.Tensor) -> torch.Tensor:
        if relation_name not in self._relation_mlps:
            raise ValueError(f"No MLP found for relation '{relation_name}'")
        relation_module: SparseMLP = self._relation_mlps[relation_name]  # type: ignore
        return relation_module(argument_embeddings.view(-1, relation_module.input_size))

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        output_messages_list: list[torch.Tensor] = []
        output_indices_list: list[torch.Tensor] = []
        for relation_name, argument_indices in relations.items():
            if argument_indices.numel() > 0:
                argument_embeddings = torch.index_select(node_embeddings, 0, argument_indices)
                argument_messages = self._forward_relation(relation_name, argument_embeddings)
                output_messages = argument_messages.view(-1, self._embedding_size)
                output_messages_list.append(output_messages)
                output_indices_list.append(argument_indices)
        output_messages = torch.cat(output_messages_list, 0)
        output_indices = torch.cat(output_indices_list, 0)
        if self._ternarize_messages:
            output_message_logits = output_messages
            output_messages = ternarize(output_message_logits)
            self._record_ternary_messages(output_message_logits, output_messages)
        return output_messages, output_indices


class SenderOnlyMLPMessages(MessageFunction):
    """Message function where messages depend only on the sender arguments.

    For each relation and argument position, a dedicated MLP computes the message
    delivered to that position from the embeddings of the *other* argument
    positions only. The receiver's own embedding is neither an input to the MLP
    nor added to the message; it reaches the update function separately through
    the concatenation with the aggregated messages.

    This makes each message channel a function of the senders alone, so that
    (with ternarized messages and max aggregation) an aggregated channel has pure
    role-restriction semantics: +1 iff some related object satisfies a concept
    over its embedding bits. For binary relations the two position MLPs realize
    the role and its inverse. Unary relations carry a learnable constant message,
    marking atom membership itself.
    """

    def __init__(self,
                 hparam_config: HyperparameterConfig,
                 input_spec: tuple[Encoder, ...]):
        """Initialize the sender-only MLP message function.

        Args:
            hparam_config: The hyperparameter configuration containing embedding sizes.
            input_spec: The input specification to determine which relations exist.
        """
        super().__init__()
        self._embedding_size = hparam_config.embedding_size
        self._ternarize_messages = hparam_config.ternarize_messages
        self._relation_arities: dict[str, int] = {}
        self._position_mlps = nn.ModuleDict()
        self._unary_messages = nn.ParameterDict()
        relations = get_relations_from_encoders(hparam_config.domain, input_spec)
        for relation_name, relation_arity in relations:
            if relation_arity == 0:
                continue
            self._relation_arities[relation_name] = relation_arity
            if relation_arity == 1:
                # Std 1 so initial values straddle the ternarize threshold.
                self._unary_messages[relation_name] = nn.Parameter(torch.randn(hparam_config.embedding_size))
            else:
                for position in range(relation_arity):
                    input_size = (relation_arity - 1) * hparam_config.embedding_size
                    self._position_mlps[f'{relation_name}@{position}'] = MLP(input_size, hparam_config.embedding_size)

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sender-only messages and indices for all relations.

        Args:
            node_embeddings: The current node embeddings.
            relations: Dictionary mapping relation names to their argument indices.

        Returns:
            Tuple of (messages, indices) for aggregation.
        """
        output_messages_list: list[torch.Tensor] = []
        output_indices_list: list[torch.Tensor] = []
        for relation_name, argument_indices in relations.items():
            if argument_indices.numel() == 0:
                continue
            if relation_name not in self._relation_arities:
                raise ValueError(f"No message parameters found for relation '{relation_name}'")
            arity = self._relation_arities[relation_name]
            argument_embeddings = torch.index_select(node_embeddings, 0, argument_indices).view(-1, arity, self._embedding_size)
            num_atoms = argument_embeddings.shape[0]
            if arity == 1:
                messages = self._unary_messages[relation_name].view(1, 1, -1).expand(num_atoms, 1, -1)
            else:
                position_messages: list[torch.Tensor] = []
                for position in range(arity):
                    sender_positions = [p for p in range(arity) if p != position]
                    sender_embeddings = argument_embeddings[:, sender_positions, :].reshape(num_atoms, -1)
                    position_mlp: MLP = self._position_mlps[f'{relation_name}@{position}']  # type: ignore
                    position_messages.append(position_mlp(sender_embeddings))
                messages = torch.stack(position_messages, dim=1)
            output_messages_list.append(messages.reshape(-1, self._embedding_size))
            output_indices_list.append(argument_indices)
        output_messages = torch.cat(output_messages_list, 0)
        output_indices = torch.cat(output_indices_list, 0)
        if self._ternarize_messages:
            output_message_logits = output_messages
            output_messages = ternarize(output_message_logits)
            self._record_ternary_messages(output_message_logits, output_messages)
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
        self._ternarize_messages = hparam_config.ternarize_messages
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
        if self._ternarize_messages:
            output_message_logits = output_messages
            output_messages = ternarize(output_message_logits)
            self._record_ternary_messages(output_message_logits, output_messages)

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
