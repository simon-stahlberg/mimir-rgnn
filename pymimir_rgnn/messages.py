import torch
import torch.nn as nn

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
        relations = get_relations_from_encoders(hparam_config.domain, input_spec)
        for relation_name, relation_arity in relations:
            input_size = relation_arity * hparam_config.embedding_size
            output_size = relation_arity * hparam_config.embedding_size
            if (input_size > 0) and (output_size > 0):
                self._relation_mlps[relation_name] = MLP(input_size, output_size)

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
        output_messages_list: list[torch.Tensor] = []
        output_indices_list: list[torch.Tensor] = []
        for relation_name, argument_indices in relations.items():
            if argument_indices.numel() > 0:
                argument_embeddings = torch.index_select(node_embeddings, 0, argument_indices)
                argument_messages = self._forward_relation(relation_name, argument_embeddings)
                output_messages = (argument_embeddings.view_as(argument_messages) + argument_messages).view(-1, self._embedding_size)
                output_messages_list.append(output_messages)
                output_indices_list.append(argument_indices)
        output_messages = torch.cat(output_messages_list, 0)
        output_indices = torch.cat(output_indices_list, 0)
        return output_messages, output_indices


class AttentionMessages(MessageFunction):
    """Message function using TransformerEncoderLayer for parallel message computation.

    This message function uses a sequence of TransformerEncoderLayer provided by PyTorch
    to compute all messages in parallel. This approach treats each ground atom as a sequence
    where the first token represents the predicate symbol and the remaining tokens represent
    the objects.
    """

    def __init__(self,
                 hparam_config: HyperparameterConfig,
                 input_spec: tuple[Encoder, ...]):
        """Initialize the attention message function.

        Args:
            hparam_config: The hyperparameter configuration containing embedding sizes.
            input_spec: The input specification to determine which relations exist.
        """
        super().__init__()
        self._embedding_size = hparam_config.embedding_size
        self._relation_arities = dict(get_relations_from_encoders(hparam_config.domain, input_spec))
        assert len(self._relation_arities) > 0, "No relations found in input specification."
        self._max_sequence_length = max(self._relation_arities.values()) + 1  # One extra token for the predicate symbol
        self._predicate_to_idx = {name: idx for idx, name in enumerate(sorted(self._relation_arities.keys()))}
        self._num_predicates = len(self._predicate_to_idx)
        self._predicate_embeddings = nn.Embedding(self._num_predicates, hparam_config.embedding_size)
        self._positional_embeddings = nn.Embedding(self._max_sequence_length, hparam_config.embedding_size)

        # TransformerEncoderLayer for parallel message computation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hparam_config.embedding_size,
            nhead=8,  # Standard number of attention heads
            dim_feedforward=hparam_config.embedding_size * 4,  # Standard multiplier
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self._transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def setup(self, relations: dict[str, torch.Tensor]) -> None:
        """Pre-compute static parts that don't change across layers.
        
        Args:
            relations: Dictionary mapping relation names to their argument indices.
        """
        if not relations:
            self._cache = {}
            return
            
        device = next(iter(relations.values())).device if relations else torch.device('cpu')
        
        # Cache structure for each relation
        relation_caches = {}
        
        for relation_name, argument_indices in relations.items():
            if argument_indices.numel() == 0:
                continue
                
            arity = self._relation_arities[relation_name]
            predicate_id = self._predicate_to_idx[relation_name]
            
            # Reshape argument indices to group them by ground atoms
            num_atoms = argument_indices.shape[0] // arity
            atom_indices = argument_indices.view(num_atoms, arity)  # [num_atoms, arity]
            
            # Pre-compute predicate embeddings for each atom
            predicate_embedding = self._predicate_embeddings(torch.full((num_atoms,), predicate_id, device=device))
            predicate_embedding = predicate_embedding.unsqueeze(1)  # [num_atoms, 1, embedding_size]
            
            # Pre-compute positional embeddings for this relation's sequence length
            sequence_length = arity + 1
            positions = torch.arange(sequence_length, device=device)
            positional_embeddings = self._positional_embeddings(positions).unsqueeze(0)  # [1, arity+1, embedding_size]
            
            # Pre-compute padding masks
            padding_mask = torch.zeros(num_atoms, self._max_sequence_length, dtype=torch.bool, device=device)
            if sequence_length < self._max_sequence_length:
                padding_mask[:, sequence_length:] = True  # Mask padding positions
                
            # Pre-compute padding tokens if needed
            padding_tokens = None
            if sequence_length < self._max_sequence_length:
                padding_size = self._max_sequence_length - sequence_length
                padding_tokens = torch.zeros(num_atoms, padding_size, self._embedding_size, device=device)
            
            # Store everything for this relation
            relation_caches[relation_name] = {
                'atom_indices': atom_indices,  # [num_atoms, arity] - for selecting object embeddings
                'predicate_embedding': predicate_embedding,  # [num_atoms, 1, embedding_size]
                'positional_embeddings': positional_embeddings,  # [1, arity+1, embedding_size]  
                'padding_mask': padding_mask,  # [num_atoms, max_sequence_length]
                'padding_tokens': padding_tokens,  # [num_atoms, padding_size, embedding_size] or None
                'sequence_length': sequence_length,
                'num_atoms': num_atoms,
                'arity': arity
            }
        
        # Pre-compute message and output indices for batched processing
        message_index_list = []
        output_index_list = []
        output_offset = 0
        
        for relation_name, argument_indices in relations.items():
            if argument_indices.numel() == 0:
                continue
                
            cache = relation_caches[relation_name]
            num_atoms = cache['num_atoms']
            arity = cache['arity']
            sequence_length = cache['sequence_length']
            
            # Compute message indices for extracting object embeddings after transformer
            messages_arguments_indices = torch.arange(1, sequence_length, dtype=torch.long, device=device)
            messages_arguments_offsets = torch.arange(num_atoms, dtype=torch.long, device=device) * self._max_sequence_length
            messages_indices = (messages_arguments_indices.unsqueeze(0) + messages_arguments_offsets.unsqueeze(1)).reshape(-1)
            messages_indices += output_offset
            output_offset += num_atoms * self._max_sequence_length
            
            message_index_list.append(messages_indices)
            output_index_list.append(argument_indices)
        
        # Store global indices
        self._cache = {
            'relation_caches': relation_caches,
            'message_indices': torch.cat(message_index_list, dim=0) if message_index_list else torch.empty(0, dtype=torch.long, device=device),
            'output_indices': torch.cat(output_index_list, dim=0) if output_index_list else torch.empty(0, dtype=torch.long, device=device),
            'has_data': len(relation_caches) > 0
        }

    def cleanup(self) -> None:
        """Clear cached data to free up memory.
        
        This is called after the message passing phase is complete.
        """
        if hasattr(self, '_cache'):
            del self._cache

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute messages using transformer attention for all relations in a single pass.

        Args:
            node_embeddings: The current node embeddings.
            relations: Dictionary mapping relation names to their argument indices.

        Returns:
            Tuple of (messages, indices) for aggregation.
        """
        # Use cached data if setup was called, otherwise fall back to original logic
        if not hasattr(self, '_cache') or not self._cache.get('has_data', False):
            # Fallback to original implementation if cache is not available
            return self._forward_original(node_embeddings, relations)
            
        # Fast path using cached data
        relation_sequence_list: list[torch.Tensor] = []
        relation_mask_list: list[torch.Tensor] = []
        
        cache = self._cache
        relation_caches = cache['relation_caches']  # type: ignore
        
        # Process relations in the same order as the original implementation
        for relation_name, argument_indices in relations.items():
            if argument_indices.numel() == 0:
                continue
            
            if relation_name not in relation_caches:  # type: ignore
                # This shouldn't happen if setup was called correctly
                continue
                
            rel_cache = relation_caches[relation_name]  # type: ignore
            
            # Get object embeddings using cached indices
            atom_indices = rel_cache['atom_indices']  # [num_atoms, arity]
            object_embeddings = torch.index_select(node_embeddings, 0, atom_indices.view(-1))
            object_embeddings = object_embeddings.view(rel_cache['num_atoms'], rel_cache['arity'], self._embedding_size)
            
            # Combine with cached predicate embeddings
            sequence_embeddings = torch.cat([rel_cache['predicate_embedding'], object_embeddings], dim=1)
            
            # Add cached positional embeddings
            sequence_embeddings = sequence_embeddings + rel_cache['positional_embeddings']
            
            # Add cached padding if needed
            if rel_cache['padding_tokens'] is not None:
                sequence_embeddings = torch.cat([sequence_embeddings, rel_cache['padding_tokens']], dim=1)
            
            relation_sequence_list.append(sequence_embeddings)
            relation_mask_list.append(rel_cache['padding_mask'])
        
        if not relation_sequence_list:
            # No valid sequences found
            device = node_embeddings.device
            empty_messages = torch.empty(0, self._embedding_size, device=device)
            empty_indices = torch.empty(0, dtype=torch.long, device=device)
            return empty_messages, empty_indices
        
        # Batch all sequences together
        relation_sequences = torch.cat(relation_sequence_list, dim=0)
        src_key_padding_mask = torch.cat(relation_mask_list, dim=0)
        
        # Apply transformer to all sequences
        transformed_embeddings = self._transformer.forward(relation_sequences, src_key_padding_mask=src_key_padding_mask)
        transformed_embeddings = transformed_embeddings.view(-1, self._embedding_size)
        
        # Extract object messages using cached indices
        output_messages = transformed_embeddings.index_select(0, cache['message_indices'])  # type: ignore
        
        return output_messages, cache['output_indices']  # type: ignore
    
    def _forward_original(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Original forward implementation as fallback when cache is not available.
        
        Args:
            node_embeddings: The current node embeddings.
            relations: Dictionary mapping relation names to their argument indices.

        Returns:
            Tuple of (messages, indices) for aggregation.
        """
        assert relations is not None, "Relations dictionary must be provided."
        assert len(self._relation_arities) > 0, "No relations available for message computation."

        device = node_embeddings.device

        # Collect all sequences and their metadata - keep per relation for easier processing
        relation_sequence_list: list[torch.Tensor] = []
        relation_mask_list: list[torch.Tensor] = []
        message_index_list: list[torch.Tensor] = []
        output_index_list: list[torch.Tensor] = []
        output_offset = 0

        for relation_name, argument_indices in relations.items():
            assert relation_name in self._relation_arities, f"Messages function is not defined for relation '{relation_name}'."

            if argument_indices.numel() == 0:
                continue

            arity = self._relation_arities[relation_name]
            predicate_id = self._predicate_to_idx[relation_name]

            # Reshape argument indices to group them by ground atoms
            num_atoms = argument_indices.shape[0] // arity
            atom_indices = argument_indices.view(num_atoms, arity)  # [num_atoms, arity]

            # Get object embeddings for each atom
            object_embeddings = torch.index_select(node_embeddings, 0, atom_indices.view(-1))
            object_embeddings = object_embeddings.view(num_atoms, arity, self._embedding_size)

            # Create predicate embeddings for each atom
            predicate_embedding = self._predicate_embeddings(torch.full((num_atoms,), predicate_id, device=device))
            predicate_embedding = predicate_embedding.unsqueeze(1)  # [num_atoms, 1, embedding_size]

            # Combine predicate and object embeddings into sequences
            sequence_embeddings = torch.cat([predicate_embedding, object_embeddings], dim=1)  # [num_atoms, arity+1, embedding_size]

            # Add positional embeddings for this relation's sequence length
            positions = torch.arange(arity + 1, device=device)
            positional_embeddings = self._positional_embeddings(positions).unsqueeze(0)  # [1, arity+1, embedding_size]
            sequence_embeddings = sequence_embeddings + positional_embeddings

            # Pad to max sequence length for this batch
            sequence_length = arity + 1
            relation_mask = torch.zeros(num_atoms, self._max_sequence_length, dtype=torch.bool, device=device)

            if sequence_length < self._max_sequence_length:
                relation_mask[:, sequence_length:] = True  # Mask padding positions
                padding_size = self._max_sequence_length - sequence_length
                padding_tokens = torch.zeros(num_atoms, padding_size, self._embedding_size, device=device)
                sequence_embeddings = torch.cat([sequence_embeddings, padding_tokens], dim=1)

            # Compute message indices for each argument in the sequence
            # These indices will be used to extract the relevant output embeddings after transformer processing
            messages_arguments_indices = torch.arange(1, sequence_length, dtype=torch.long, device=device)
            messages_arguments_offsets = torch.arange(num_atoms, dtype=torch.long, device=device) * self._max_sequence_length
            messages_indices = (messages_arguments_indices.unsqueeze(0) + messages_arguments_offsets.unsqueeze(1)).reshape(-1)
            messages_indices += output_offset
            output_offset += num_atoms * self._max_sequence_length

            # Store per relation (don't concatenate yet - different arities)
            relation_sequence_list.append(sequence_embeddings)
            relation_mask_list.append(relation_mask)
            message_index_list.append(messages_indices)
            output_index_list.append(argument_indices)

        if not relation_sequence_list:
            # No valid sequences found
            empty_messages = torch.empty(0, self._embedding_size, device=device)
            empty_indices = torch.empty(0, dtype=torch.long, device=device)
            return empty_messages, empty_indices

        # Batch all sequences together (now they all have the same sequence length)
        relation_sequences = torch.cat(relation_sequence_list, dim=0)  # [total_num_atoms, max_sequence_length, embedding_size]
        messages_indices = torch.cat(message_index_list, dim=0)  # [total_num_objects]
        src_key_padding_mask = torch.cat(relation_mask_list, dim=0)  # [total_num_atoms, max_sequence_length]

        # Apply transformer to all sequences
        transformed_embeddings = self._transformer.forward(relation_sequences, src_key_padding_mask=src_key_padding_mask)
        transformed_embeddings = transformed_embeddings.view(-1, self._embedding_size)  # Flatten for easier indexing

        # Extract object messages and their indices for aggregation
        output_messages = transformed_embeddings.index_select(0, messages_indices)
        output_indices = torch.cat(output_index_list, dim=0)

        return output_messages, output_indices
