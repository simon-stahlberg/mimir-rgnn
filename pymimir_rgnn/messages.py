import torch
import torch.nn as nn

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
    to compute all messages in parallel. In contrast to PredicateMLPMessages, this approach
    treats each ground atom as a sequence where the first token represents the predicate
    symbol and the remaining tokens represent the objects.
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
        
        # Get all relations and their arities
        relations = get_relations_from_encoders(hparam_config.domain, input_spec)
        self._relation_arities = {name: arity for name, arity in relations}
        
        # Determine maximum arity for positional embeddings
        max_arity = max(self._relation_arities.values()) if self._relation_arities else 0
        self._max_sequence_length = max_arity + 1  # +1 for predicate symbol
        
        # Create embeddings for predicate symbols
        predicate_names = sorted(self._relation_arities.keys())
        self._predicate_to_id = {name: i for i, name in enumerate(predicate_names)}
        self._num_predicates = len(predicate_names)
        
        # Learned embeddings for predicate symbols
        self._predicate_embeddings = nn.Embedding(self._num_predicates, hparam_config.embedding_size)
        
        # Positional embeddings (position 0 = predicate, position 1+ = objects)
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
    
    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute messages using transformer attention for all relations in a single pass.
        
        Args:
            node_embeddings: The current node embeddings.
            relations: Dictionary mapping relation names to their argument indices.
            
        Returns:
            Tuple of (messages, indices) for aggregation.
        """
        if not relations:
            # Return empty tensors if no relations
            device = node_embeddings.device
            empty_messages = torch.empty(0, self._embedding_size, device=device)
            empty_indices = torch.empty(0, dtype=torch.long, device=device)
            return empty_messages, empty_indices
        
        device = node_embeddings.device
        
        # Collect all sequences and their metadata - keep per relation for easier processing
        relation_sequences: list[torch.Tensor] = []
        relation_original_embeddings: list[torch.Tensor] = []
        relation_indices: list[torch.Tensor] = []
        relation_arities: list[int] = []
        
        for relation_name, argument_indices in relations.items():
            if argument_indices.numel() == 0:
                continue
                
            if relation_name not in self._relation_arities:
                continue
                
            arity = self._relation_arities[relation_name]
            predicate_id = self._predicate_to_id[relation_name]
            
            # Reshape argument indices to group them by ground atoms
            num_atoms = argument_indices.shape[0] // arity
            atom_indices = argument_indices.view(num_atoms, arity)  # [num_atoms, arity]
            
            # Get object embeddings for each atom
            object_embeddings = torch.index_select(node_embeddings, 0, atom_indices.flatten())
            object_embeddings = object_embeddings.view(num_atoms, arity, self._embedding_size)
            
            # Create predicate embeddings for each atom
            predicate_emb = self._predicate_embeddings(torch.full((num_atoms,), predicate_id, device=device))
            predicate_emb = predicate_emb.unsqueeze(1)  # [num_atoms, 1, embedding_size]
            
            # Combine predicate and object embeddings into sequences
            sequence_embeddings = torch.cat([predicate_emb, object_embeddings], dim=1)  # [num_atoms, arity+1, embedding_size]
            
            # Add positional embeddings for this relation's sequence length
            positions = torch.arange(arity + 1, device=device)
            pos_embeddings = self._positional_embeddings(positions).unsqueeze(0)  # [1, arity+1, embedding_size]
            sequence_embeddings = sequence_embeddings + pos_embeddings
            
            # Pad to max sequence length for this batch
            if sequence_embeddings.shape[1] < self._max_sequence_length:
                padding_size = self._max_sequence_length - sequence_embeddings.shape[1]
                padding = torch.zeros(num_atoms, padding_size, self._embedding_size, device=device)
                sequence_embeddings = torch.cat([sequence_embeddings, padding], dim=1)
            
            # Store per relation (don't concatenate yet - different arities)
            relation_sequences.append(sequence_embeddings)
            relation_original_embeddings.append(object_embeddings)
            relation_indices.append(atom_indices.flatten())
            relation_arities.append(arity)
        
        if not relation_sequences:
            # No valid sequences found
            empty_messages = torch.empty(0, self._embedding_size, device=device)
            empty_indices = torch.empty(0, dtype=torch.long, device=device)
            return empty_messages, empty_indices
        
        # Batch all sequences together (now they all have the same sequence length)
        batched_sequences = torch.cat(relation_sequences, dim=0)  # [total_num_atoms, max_sequence_length, embedding_size]
        batched_indices = torch.cat(relation_indices, dim=0)  # [total_num_objects]
        
        # Create attention mask for variable-length sequences across different relations
        src_key_padding_mask = torch.zeros(batched_sequences.shape[0], self._max_sequence_length, 
                                         dtype=torch.bool, device=device)
        
        # Fill mask based on actual sequence lengths per relation
        seq_idx = 0
        for i, arity in enumerate(relation_arities):
            actual_seq_len = arity + 1  # +1 for predicate
            num_atoms_in_relation = relation_sequences[i].shape[0]
            
            # Mask padding positions for this relation's atoms
            if actual_seq_len < self._max_sequence_length:
                src_key_padding_mask[seq_idx:seq_idx + num_atoms_in_relation, actual_seq_len:] = True
            
            seq_idx += num_atoms_in_relation
        
        # Apply transformer to all sequences in parallel - SINGLE CALL!
        transformed_embeddings = self._transformer(batched_sequences, src_key_padding_mask=src_key_padding_mask)
        
        # Extract object messages from batched output and rebuild per-relation structure
        output_messages_list: list[torch.Tensor] = []
        
        seq_start = 0
        for i, arity in enumerate(relation_arities):
            num_atoms_in_relation = relation_sequences[i].shape[0]
            
            # Extract transformed sequences for this relation
            relation_transformed = transformed_embeddings[seq_start:seq_start + num_atoms_in_relation]
            
            # Extract object messages (skip predicate at position 0)
            object_messages = relation_transformed[:, 1:arity+1, :]  # [num_atoms, arity, embedding_size]
            object_messages = object_messages.reshape(-1, self._embedding_size)  # [num_atoms * arity, embedding_size]
            
            # Add residual connection with original object embeddings
            original_object_embeddings = relation_original_embeddings[i].reshape(-1, self._embedding_size)
            final_messages = object_messages + original_object_embeddings
            
            output_messages_list.append(final_messages)
            seq_start += num_atoms_in_relation
        
        # Concatenate all messages and indices
        output_messages = torch.cat(output_messages_list, 0)
        output_indices = batched_indices
            
        return output_messages, output_indices
