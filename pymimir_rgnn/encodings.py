import pymimir as mm
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Any, Optional

from .utils import get_action_name, get_atom_name, get_effect_name, get_effect_relation_name, get_predicate_name, relations_to_tensors


class Encoder(ABC):
    """Base class for encoders that transform PDDL structures into graph neural network inputs."""
    
    @abstractmethod
    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get the relations this encoder contributes to the graph encoding."""
        pass
    
    @abstractmethod
    def encode(self, input_value: Any, intermediate: 'ListInput', state: mm.State) -> int:
        """
        Encode the input value into the intermediate representation.
        
        Args:
            input_value: The input data to encode (state, goal, actions, etc.)
            intermediate: The intermediate ListInput object to populate
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
    def forward(self, node_embeddings: torch.Tensor, input: 'EncodedInput') -> Any:
        """
        Perform readout from node embeddings to produce output values.
        
        Args:
            node_embeddings: The node embeddings from the graph neural network
            input: The encoded input containing graph structure information
            
        Returns:
            The decoded output (tensors, lists of tensors, etc.)
        """
        pass


class StateEncoder(Encoder):
    """Encoder for planning states."""
    
    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        ignored_predicate_names = ['number']
        predicates = [predicate for predicate in domain.get_predicates() if not predicate.get_name() in ignored_predicate_names]
        return [(get_predicate_name(predicate, False, True), predicate.get_arity()) for predicate in predicates]
    
    def encode(self, input_value: Any, intermediate: 'ListInput', state: mm.State) -> int:
        """Encode a planning state into the intermediate representation."""
        assert isinstance(input_value, mm.State), f'StateEncoder expected a State, got {type(input_value)}'
        
        problem = input_value.get_problem()
        domain = problem.get_domain()
        num_objects = len(problem.get_objects()) + len(domain.get_constants())
        
        # Add atom relations for all atoms in the state
        for atom in input_value.get_atoms():
            self._add_atom_relation(atom, input_value, False, intermediate)
        
        # Track object indices
        intermediate.object_indices.extend(range(intermediate.node_count, intermediate.node_count + num_objects))
        intermediate.object_sizes.append(num_objects)
        
        return num_objects
    
    def _add_atom_relation(self, atom: mm.GroundAtom, state: mm.State, is_goal_atom: bool, intermediate: 'ListInput'):
        """Helper method to add an atom relation to the intermediate representation."""
        relation_name = get_atom_name(atom, state, is_goal_atom)
        object_indices: list[int] = [term.get_index() + intermediate.node_count for term in atom.get_terms()]
        if relation_name not in intermediate.flattened_relations:
            intermediate.flattened_relations[relation_name] = object_indices
        else:
            intermediate.flattened_relations[relation_name].extend(object_indices)


class GoalEncoder(Encoder):
    """Encoder for goal conditions."""
    
    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        ignored_predicate_names = ['number']
        predicates = [predicate for predicate in domain.get_predicates() if not predicate.get_name() in ignored_predicate_names]
        relations = []
        relations.extend([(get_predicate_name(predicate, True, False), predicate.get_arity()) for predicate in predicates])
        relations.extend([(get_predicate_name(predicate, True, True), predicate.get_arity()) for predicate in predicates])
        return relations
    
    def encode(self, input_value: Any, intermediate: 'ListInput', state: mm.State) -> int:
        """Encode a goal condition into the intermediate representation."""
        assert isinstance(input_value, mm.GroundConjunctiveCondition), f'GoalEncoder expected a GroundConjunctiveCondition, got {type(input_value)}'
        
        for literal in input_value:  # type: ignore
            assert isinstance(literal, mm.GroundLiteral), 'Goal condition should contain ground literals.'
            assert literal.get_polarity(), 'Only positive literals are supported.'
            self._add_atom_relation(literal.get_atom(), state, True, intermediate)
            
        return 0  # Goals don't add new nodes, they just add relations
    
    def _add_atom_relation(self, atom: mm.GroundAtom, state: mm.State, is_goal_atom: bool, intermediate: 'ListInput'):
        """Helper method to add an atom relation to the intermediate representation."""
        relation_name = get_atom_name(atom, state, is_goal_atom)
        object_indices: list[int] = [term.get_index() + intermediate.node_count for term in atom.get_terms()]
        if relation_name not in intermediate.flattened_relations:
            intermediate.flattened_relations[relation_name] = object_indices
        else:
            intermediate.flattened_relations[relation_name].extend(object_indices)


class GroundActionsEncoder(Encoder):
    """Encoder for ground actions."""
    
    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        return [(get_action_name(action), action.get_arity() + 1) for action in domain.get_actions()]
    
    def encode(self, input_value: Any, intermediate: 'ListInput', state: mm.State) -> int:
        """Encode ground actions into the intermediate representation."""
        assert isinstance(input_value, list), f'GroundActionsEncoder expected a list, got {type(input_value)}'
        
        actions = input_value
        num_actions = len(actions)
        
        for action_index, action in enumerate(actions):
            assert isinstance(action, mm.GroundAction), f'Expected a GroundAction in the list at position {action_index}'
            
            problem = action.get_problem()
            num_objects = len(problem.get_objects())
            relation_name = get_action_name(action)
            action_local_id = num_objects + action_index
            action_global_id = action_local_id + intermediate.node_count
            term_ids = [action_global_id] + [term.get_index() + intermediate.node_count for term in action.get_objects()]
            
            # Add to input relations
            if relation_name not in intermediate.flattened_relations:
                intermediate.flattened_relations[relation_name] = term_ids
            else:
                intermediate.flattened_relations[relation_name].extend(term_ids)
                
            # Each action adds a new node, remember the id
            intermediate.action_indices.append(action_global_id)
            
        intermediate.action_sizes.append(num_actions)
        return num_actions


class TransitionEffectsEncoder(Encoder):
    """Encoder for transition effects."""
    
    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        ignored_predicate_names = ['number']
        predicates = [predicate for predicate in domain.get_predicates() if not predicate.get_name() in ignored_predicate_names]
        relations = []
        relations.extend([(get_effect_name(predicate, True, False), predicate.get_arity() + 1) for predicate in predicates])
        relations.extend([(get_effect_name(predicate, False, False), predicate.get_arity() + 1) for predicate in predicates])
        relations.extend([(get_effect_name(predicate, True, True), predicate.get_arity() + 1) for predicate in predicates])
        relations.extend([(get_effect_name(predicate, False, True), predicate.get_arity() + 1) for predicate in predicates])
        relations.append((get_effect_relation_name(), 2))
        return relations
    
    def encode(self, input_value: Any, intermediate: 'ListInput', state: mm.State) -> int:
        """Encode transition effects into the intermediate representation."""
        assert isinstance(input_value, list), f'TransitionEffectsEncoder expected a list, got {type(input_value)}'
        
        effects_list = input_value
        problem = state.get_problem()
        goal_condition = problem.get_goal_condition()
        num_objects = len(problem.get_objects())
        num_transitions = len(effects_list)
        
        for transition_index, effects in enumerate(effects_list):
            assert isinstance(effects, list), 'Expected a list of lists of ground literals.'
            transition_local_id = num_objects + transition_index
            transition_global_id = transition_local_id + intermediate.node_count
            
            for effect in effects:
                assert isinstance(effect, mm.GroundLiteral), 'Expected a list of lists of ground literals.'
                effect_name = get_effect_name(effect.get_atom().get_predicate(), effect.get_polarity(), False)
                term_ids = [transition_global_id] + [term.get_index() + intermediate.node_count for term in effect.get_atom().get_terms()]
                
                if effect_name not in intermediate.flattened_relations:
                    intermediate.flattened_relations[effect_name] = term_ids
                else:
                    intermediate.flattened_relations[effect_name].extend(term_ids)
                    
                # Add literals stating how this transition affects the goal
                if any(x == effect.get_atom() for x in goal_condition):
                    goal_effect_name = get_effect_name(effect.get_atom().get_predicate(), effect.get_polarity(), True)
                    goal_term_ids = [transition_global_id] + [term.get_index() + intermediate.node_count for term in effect.get_atom().get_terms()]
                    
                    if goal_effect_name not in intermediate.flattened_relations:
                        intermediate.flattened_relations[goal_effect_name] = goal_term_ids
                    else:
                        intermediate.flattened_relations[goal_effect_name].extend(goal_term_ids)
                        
            # Each transition adds a new node, remember the id
            intermediate.action_indices.append(transition_global_id)
            
        intermediate.action_sizes.append(num_transitions)
        return num_transitions


class SuccessorsEncoder(Encoder):
    """Encoder for state successors."""
    
    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        # Successors not yet implemented
        return []
    
    def encode(self, input_value: Any, intermediate: 'ListInput', state: mm.State) -> int:
        """Encode state successors into the intermediate representation."""
        raise NotImplementedError('State successors are not supported yet.')


# Decoder implementations
class ActionScalarDecoder(Decoder):
    """Decoder for scalar values over actions."""
    
    def __init__(self, embedding_size: int):
        super().__init__()
        # Import modules here to avoid circular imports
        from .modules import SumReadout, MLP
        self._object_readout = SumReadout(embedding_size, embedding_size)
        self._action_value = MLP(2 * embedding_size, 1)

    def forward(self, node_embeddings: torch.Tensor, input: 'EncodedInput') -> list[torch.Tensor]:
        action_embeddings = node_embeddings.index_select(0, input.action_indices)
        object_embeddings = node_embeddings.index_select(0, input.object_indices)
        object_aggregation: torch.Tensor = self._object_readout(object_embeddings, input.object_sizes)
        object_aggregation = object_aggregation.repeat_interleave(input.action_sizes, dim=0)
        values: torch.Tensor = self._action_value(torch.cat((action_embeddings, object_aggregation), dim=1))
        return [action_values.view(-1) for action_values in values.split(input.action_sizes.tolist())]  # type: ignore


class ActionEmbeddingDecoder(Decoder):
    """Decoder for embeddings over actions."""
    
    def forward(self, node_embeddings: torch.Tensor, input: 'EncodedInput') -> torch.Tensor:
        return node_embeddings.index_select(0, input.action_indices)


class ObjectsScalarDecoder(Decoder):
    """Decoder for scalar values over objects."""
    
    def __init__(self, embedding_size: int):
        super().__init__()
        # Import modules here to avoid circular imports
        from .modules import SumReadout
        self._object_readout = SumReadout(embedding_size, 1)

    def forward(self, node_embeddings: torch.Tensor, input: 'EncodedInput') -> torch.Tensor:
        object_embeddings = node_embeddings.index_select(0, input.object_indices)
        return self._object_readout(object_embeddings, input.object_sizes).view(-1)


class ObjectsEmbeddingDecoder(Decoder):
    """Decoder for embeddings over objects."""
    
    def forward(self, node_embeddings: torch.Tensor, input: 'EncodedInput') -> torch.Tensor:
        return node_embeddings.index_select(0, input.object_indices)


class ListInput:
    def __init__(self):
        self.flattened_relations: dict[str, list[int]] = {}
        self.node_count: int = 0
        self.node_sizes: list[int] = []
        self.object_sizes: list[int] = []
        self.object_indices: list[int] = []
        self.action_sizes: list[int] = []
        self.action_indices: list[int] = []


class EncodedInput:
    def __init__(self):
        self.flattened_relations: dict[str, torch.Tensor] = {}
        self.node_count: int = 0
        self.node_sizes: torch.Tensor = torch.LongTensor()
        self.object_sizes: torch.Tensor = torch.LongTensor()
        self.object_indices: torch.Tensor = torch.LongTensor()
        self.action_sizes: torch.Tensor = torch.LongTensor()
        self.action_indices: torch.Tensor = torch.LongTensor()


def get_encoding_from_encoders(domain: mm.Domain, input_specification: tuple[Encoder, ...]) -> list[tuple[str, int]]:
    """Get relations from encoder-based input specifications."""
    relations: list[tuple[str, int]] = []
    for encoder in input_specification:
        relations.extend(encoder.get_relations(domain))
    relations.sort()  # Ensure that the output is deterministic.
    return relations


def encode_input_from_encoders(input: list[tuple], input_specification: tuple[Encoder, ...], device: torch.device) -> EncodedInput:
    """Encode input using encoder-based specifications."""
    intermediate = ListInput()
    
    # Process each input instance
    for instance in input:
        assert isinstance(instance, tuple), 'Input instance must be a tuple.'
        assert len(instance) == len(input_specification), 'Mismatch between the length of an input instance and the input specification.'
        
        # Find the state encoder to get the state
        state = None
        for i, encoder in enumerate(input_specification):
            if isinstance(encoder, StateEncoder):
                state = instance[i]
                assert isinstance(state, mm.State), f'Expected a State at position {i}, got {type(state)}'
                break
        
        assert state is not None, 'Input specification must contain a StateEncoder.'
        
        # Track nodes added for this instance
        added_nodes = 0
        
        # Process each encoder with its corresponding input value
        for encoder_index, encoder in enumerate(input_specification):
            input_value = instance[encoder_index]
            nodes_added = encoder.encode(input_value, intermediate, state)
            added_nodes += nodes_added
            
        intermediate.node_sizes.append(added_nodes)
        intermediate.node_count += added_nodes

    # Convert the lists to tensors on the correct device
    result = EncodedInput()
    result.flattened_relations = relations_to_tensors(intermediate.flattened_relations, device)
    result.node_count = intermediate.node_count
    result.node_sizes = torch.tensor(intermediate.node_sizes, dtype=torch.int, device=device, requires_grad=False)
    result.object_indices = torch.tensor(intermediate.object_indices, dtype=torch.int, device=device, requires_grad=False)
    result.object_sizes = torch.tensor(intermediate.object_sizes, dtype=torch.int, device=device, requires_grad=False)
    result.action_indices = torch.tensor(intermediate.action_indices, dtype=torch.int, device=device, requires_grad=False)
    result.action_sizes = torch.tensor(intermediate.action_sizes, dtype=torch.int, device=device, requires_grad=False)
    return result
