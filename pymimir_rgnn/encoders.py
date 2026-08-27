from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pymimir as mm
import torch
from pymimir.learning import RelationBuffer

from .bases import Encoder, EncodedLists, EncodedTensors, EncodingContext
from .utils import get_action_name, get_effect_name, get_effect_relation_name, get_predicate_name, relations_to_tensors


class EncoderRelationKind(Enum):
    """Semantic role of a relation declared by a built-in encoder."""

    STATE_PREDICATE = "state_predicate"
    STATE_TYPE = "state_type"
    STATE_CONSTANT = "state_constant"
    GOAL_FALSE = "goal_false"
    GOAL_TRUE = "goal_true"
    EFFECT_POSITIVE = "effect_positive"
    EFFECT_NEGATIVE = "effect_negative"
    EFFECT_POSITIVE_GOAL = "effect_positive_goal"
    EFFECT_NEGATIVE_GOAL = "effect_negative_goal"
    EFFECT_LINK = "effect_link"


@dataclass(frozen=True)
class EncoderRelation:
    """Semantic description of one relation declared by an encoder."""

    name: str
    arity: int
    kind: EncoderRelationKind
    source_name: str | None


def _type_relation_name(
    type_name: str,
    suffix: str,
    *,
    expressive: bool = False,
) -> str:
    prefix = "expressive_" if expressive else ""
    return f"{prefix}type_relation_{type_name}{suffix}"


def _constant_relation_name(constant_name: str, suffix: str) -> str:
    return f"constant_relation_{constant_name}{suffix}"


def _predicate_relation_arity(predicate: mm.Predicate) -> int:
    """Return the encoded arity after lifting nullary facts over objects."""
    return max(predicate.arity, 1)


def _expressive_predicate_relation_arity(predicate: mm.Predicate) -> int:
    """Return the object-pair arity, with nullary facts lifted to unary."""
    return max(predicate.arity * predicate.arity, 1)


def _relations_from_native_buffer(
    relation_buffer: RelationBuffer,
    custom_relations: dict[str, list[int]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Create relation tensors while keeping native values packed.

    Native values cross the Python/Torch boundary once through ``frombuffer``
    and, for non-CPU devices, through one packed device transfer.  Relations
    without a custom collision are views into that packed device tensor.
    """
    relations = relations_to_tensors(custom_relations, device)
    descriptors = relation_buffer.descriptors
    value_view = relation_buffer.values
    # RelationBuffer validates that its descriptors cover this packed int32
    # storage contiguously when it is constructed.
    value_count = value_view.nbytes // 4

    if value_count > 0:
        cpu_values = torch.frombuffer(
            value_view,
            dtype=torch.int32,
            count=value_count,
            requires_grad=False,
        )
        # The synchronous copy lets the temporary CPU exporter be released as
        # soon as this helper returns on accelerator paths.
        device_values = cpu_values.to(device=device, non_blocking=False)
    else:
        # torch.frombuffer rejects empty buffers.  A single empty base still
        # lets every nullary relation be represented by a view.
        device_values = torch.empty(
            0,
            dtype=torch.int32,
            device=device,
            requires_grad=False,
        )

    for descriptor in descriptors:
        native_values = device_values.narrow(
            0,
            descriptor.offset,
            descriptor.length,
        )

        custom_values = relations.get(descriptor.name)
        if custom_values is None:
            relations[descriptor.name] = native_values
        else:
            # Match the previous list merge exactly: all custom values are
            # accumulated first, followed by all native values for the batch.
            relations[descriptor.name] = torch.cat(
                (custom_values, native_values),
            )

    return relations, device_values


class StateEncoder(Encoder):
    """Encoder for planning states.

    This encoder transforms a planning state (which contains atoms/facts that are
    currently true) into nodes and relations for the graph neural network.
    Objects become nodes, atoms become relations between objects, and each domain
    constant receives its own unary identity relation.
    """

    def __init__(self, suffix: str = '') -> None:
        """Initialize the state encoder.

        Args:
            suffix: Optional suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), 'Suffix must be a string.'
        self.suffix = suffix

    def get_relation_descriptors(
        self,
        domain: mm.Domain,
    ) -> list[EncoderRelation]:
        """Describe state relations by semantic kind and source symbol.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            State predicate, type, and domain-constant relation descriptors.
            Callers should use ``kind`` and ``source_name`` rather than
            descriptor positions.
        """
        descriptors = [
            EncoderRelation(
                name=get_predicate_name(
                    predicate,
                    False,
                    True,
                    self.suffix,
                ),
                arity=_predicate_relation_arity(predicate),
                kind=EncoderRelationKind.STATE_PREDICATE,
                source_name=predicate.name,
            )
            for predicate in domain.predicates
        ]
        descriptors.extend(
            EncoderRelation(
                name=_type_relation_name(type_name, self.suffix),
                arity=1,
                kind=EncoderRelationKind.STATE_TYPE,
                source_name=type_name,
            )
            for type_name in domain.type_hierarchy
        )
        descriptors.extend(
            EncoderRelation(
                name=_constant_relation_name(constant.name, self.suffix),
                arity=1,
                kind=EncoderRelationKind.STATE_CONSTANT,
                source_name=constant.name,
            )
            for constant in domain.constants
        )
        return descriptors

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get the names and arities of relations emitted by this encoder."""
        return [
            (descriptor.name, descriptor.arity)
            for descriptor in self.get_relation_descriptors(domain)
        ]

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: EncodedLists,
        context: EncodingContext,
    ) -> None:
        assert isinstance(input_value, mm.State), f'StateEncoder expected a State, got {type(input_value)}'
        mm.learning.encode_state(
            context,
            input_value,
            suffix=self.suffix,
        )


class GoalEncoder(Encoder):
    """Encoder for goal conditions.

    This encoder transforms a goal condition (conjunctive condition of literals
    that must be satisfied) into relations for the graph neural network.
    It creates both goal-specific relations and marks which atoms are true/false
    in the current state relative to the goal.
    """

    def __init__(self, suffix: str = '') -> None:
        """Initialize the goal encoder.

        Args:
            suffix: Suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), 'Suffix must be a string.'
        self.suffix = suffix

    def get_relation_descriptors(
        self,
        domain: mm.Domain,
    ) -> list[EncoderRelation]:
        """Describe goal relations by semantic kind and predicate name.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            False and true goal-relation descriptors for every predicate.
        """
        descriptors = [
            EncoderRelation(
                name=get_predicate_name(
                    predicate,
                    True,
                    False,
                    self.suffix,
                ),
                arity=_predicate_relation_arity(predicate),
                kind=EncoderRelationKind.GOAL_FALSE,
                source_name=predicate.name,
            )
            for predicate in domain.predicates
        ]
        descriptors.extend(
            EncoderRelation(
                name=get_predicate_name(
                    predicate,
                    True,
                    True,
                    self.suffix,
                ),
                arity=_predicate_relation_arity(predicate),
                kind=EncoderRelationKind.GOAL_TRUE,
                source_name=predicate.name,
            )
            for predicate in domain.predicates
        )
        return descriptors

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get the names and arities of relations emitted by this encoder."""
        return [
            (descriptor.name, descriptor.arity)
            for descriptor in self.get_relation_descriptors(domain)
        ]

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: EncodedLists,
        context: EncodingContext,
    ) -> None:
        assert isinstance(input_value, mm.GroundConjunctiveCondition), f'GoalEncoder expected a GroundConjunctiveCondition, got {type(input_value)}'
        mm.learning.encode_goal(
            context,
            state,
            input_value,
            suffix=self.suffix,
        )



class GroundActionsEncoder(Encoder):
    """Encoder for ground actions.

    This encoder transforms a sequence of available ground actions into nodes and
    relations for the graph neural network. Each action becomes a new node,
    and relations connect actions to their parameter objects.
    """

    def __init__(self, suffix: str = '') -> None:
        """Initialize the ground actions encoder.

        Args:
            suffix: Optional suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), 'Suffix must be a string.'
        self.suffix = suffix

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for actions.

        Args:
            domain: The PDDL domain containing action definitions.

        Returns:
            List of (relation_name, arity) pairs for all actions, where arity
            is the action's parameter count plus 1 (for the action node itself).
        """
        return [(get_action_name(action, self.suffix), action.arity + 1) for action in domain.actions]

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: EncodedLists,
        context: EncodingContext,
    ) -> None:
        assert isinstance(input_value, (list, tuple)), f'GroundActionsEncoder expected a sequence, got {type(input_value)}'
        mm.learning.encode_action_list(
            context,
            state,
            input_value,
            suffix=self.suffix,
        )


class TransitionEffectsEncoder(Encoder):
    """Encoder for transition effects.

    This encoder compares an ordered sequence of successor states with the
    current state. Each successor becomes a new transition node, in input order,
    with relations connecting it to the fluent and derived atoms that changed.
    """

    def __init__(self, suffix: str = '') -> None:
        """Initialize the transition effects encoder.

        Args:
            suffix: Optional suffix to append to relation names to avoid conflicts.
        """
        super().__init__()
        assert isinstance(suffix, str), 'Suffix must be a string.'
        self.suffix = suffix

    def get_relation_descriptors(
        self,
        domain: mm.Domain,
    ) -> list[EncoderRelation]:
        """Describe transition-effect relations by semantic kind and source.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            Predicate effect descriptors and the structural effect link.
        """
        descriptors: list[EncoderRelation] = []
        variants = (
            (EncoderRelationKind.EFFECT_POSITIVE, True, False),
            (EncoderRelationKind.EFFECT_NEGATIVE, False, False),
            (EncoderRelationKind.EFFECT_POSITIVE_GOAL, True, True),
            (EncoderRelationKind.EFFECT_NEGATIVE_GOAL, False, True),
        )
        for kind, positive, affects_goal in variants:
            descriptors.extend(
                EncoderRelation(
                    name=get_effect_name(
                        predicate,
                        positive,
                        affects_goal,
                        self.suffix,
                    ),
                    arity=predicate.arity + 1,
                    kind=kind,
                    source_name=predicate.name,
                )
                for predicate in domain.predicates
            )
        descriptors.append(
            EncoderRelation(
                name=get_effect_relation_name(self.suffix),
                arity=2,
                kind=EncoderRelationKind.EFFECT_LINK,
                source_name=None,
            )
        )
        return descriptors

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get the names and arities of relations emitted by this encoder."""
        return [
            (descriptor.name, descriptor.arity)
            for descriptor in self.get_relation_descriptors(domain)
        ]

    def encode(
        self,
        input_value: Any,
        state: mm.State,
        encoding: EncodedLists,
        context: EncodingContext,
    ) -> None:
        assert isinstance(input_value, tuple) and len(input_value) == 3, (
            f'TransitionEffectsEncoder expected a 3-tuple (successors, effect_relations, goal_condition), '
            f'got {type(input_value)}'
        )
        successors, effect_relations, goal_condition = input_value
        assert isinstance(successors, Sequence) and all(
            isinstance(successor, mm.State) for successor in successors
        ), (
            'TransitionEffectsEncoder expected an ordered sequence of State '
            'values as the first element.'
        )
        assert isinstance(goal_condition, mm.GroundConjunctiveCondition), (
            f'TransitionEffectsEncoder expected a GroundConjunctiveCondition as the third element, '
            f'got {type(goal_condition)}'
        )
        mm.learning.encode_transition_effects(
            context,
            state,
            successors,
            effect_relations,
            goal_condition,
            suffix=self.suffix,
        )


class VirtualNodeEncoder(Encoder):
    """Encoder for virtual nodes.

    This encoder adds a virtual node to the representation.
    All original objects are connected to a new virtual node.
    This can help the graph neural network to better capture global context.
    """
    def __init__(self) -> None:
        """Initialize the virtual node encoder."""
        super().__init__()
        self.link_name = 'virtual_node_link'

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        return [(self.link_name, 2)]

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        mm.learning.encode_virtual_node(context)


def get_relations_from_encoders(domain: mm.Domain, input_specification: tuple[Encoder, ...]) -> list[tuple[str, int]]:
    """Get all relations from a collection of encoders.

    Args:
        domain: The PDDL domain containing predicates and actions.
        input_specification: Tuple of encoder instances.

    Returns:
        Sorted list of (relation_name, arity) pairs from all encoders.
    """
    relations_set: list[tuple[str, int]] = []
    for encoder in input_specification:
        relations_set.extend(encoder.get_relations(domain))
    relations_list = list(relations_set)
    relations_list.sort()  # Ensure that the output is deterministic.
    return relations_list


def get_input_from_encoders(input: list[tuple], input_specification: tuple[Encoder, ...], device: torch.device) -> EncodedTensors:
    """Encode input using a collection of encoders.

    This function processes a batch of input instances using the provided
    encoder specification and returns the encoded graph representation
    ready for use in the graph neural network.

    Args:
        input: List of input tuples, where each tuple contains the inputs
               corresponding to the encoder specification.
        input_specification: Tuple of encoder instances that define how to
                             process each element of the input tuples.
        device: The torch device to place the resulting tensors on.

    Returns:
        EncodedTensors object containing the graph representation.

    Raises:
        AssertionError: If input format doesn't match specification or if
                        no StateEncoder is found in the specification.
    """
    encoding_lists = EncodedLists()
    with EncodingContext() as context:
        # Process each input instance.
        for instance in input:
            assert isinstance(instance, tuple), 'Input instance must be a tuple.'
            assert len(instance) == len(input_specification), 'Mismatch between the length of an input instance and the input specification.'

            # Find the state of the input.
            state = None
            for x in instance:
                if isinstance(x, mm.State):
                    state = x
                    break

            assert state is not None, 'Input must contain a State.'

            context.begin_instance(state.problem)
            try:
                # Process each encoder with its corresponding input value.
                for encoder_index, encoder in enumerate(input_specification):
                    input_value = instance[encoder_index]
                    encoder.encode(input_value, state, encoding_lists, context)
            finally:
                context.end_instance()

        # Native encoders accumulate their relations in the shared context.
        # Custom Python encoders continue writing to EncodedLists and may use
        # the same node allocator while an instance is active.
        relation_buffer = context.to_relation_buffer()

        # Copy batch-wide metadata before closing the native context. The
        # Python-owned lists remain valid during the tensor conversion below.
        encoding_lists.node_count = context.node_count
        encoding_lists.node_sizes = context.node_sizes
        encoding_lists.object_indices = context.object_indices
        encoding_lists.object_sizes = context.object_sizes
        encoding_lists.action_indices = context.action_indices
        encoding_lists.action_sizes = context.action_sizes
        encoding_lists.virtual_indices = context.virtual_indices
        encoding_lists.virtual_sizes = context.virtual_sizes
        encoding_lists.auxiliary_indices = context.auxiliary_indices
        encoding_lists.auxiliary_sizes = context.auxiliary_sizes

    # The native context is closed before any tensor allocation or device copy.
    encoding_tensors = EncodedTensors()
    (
        encoding_tensors.flattened_relations,
        encoding_tensors._native_relation_values,
    ) = _relations_from_native_buffer(
        relation_buffer,
        encoding_lists.flattened_relations,
        device,
    )
    encoding_tensors.node_count = encoding_lists.node_count
    encoding_tensors.node_sizes = torch.tensor(encoding_lists.node_sizes, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.object_indices = torch.tensor(encoding_lists.object_indices, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.object_sizes = torch.tensor(encoding_lists.object_sizes, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.action_indices = torch.tensor(encoding_lists.action_indices, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.action_sizes = torch.tensor(encoding_lists.action_sizes, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.virtual_indices = torch.tensor(encoding_lists.virtual_indices, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.virtual_sizes = torch.tensor(encoding_lists.virtual_sizes, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.auxiliary_indices = torch.tensor(encoding_lists.auxiliary_indices, dtype=torch.int, device=device, requires_grad=False)
    encoding_tensors.auxiliary_sizes = torch.tensor(encoding_lists.auxiliary_sizes, dtype=torch.int, device=device, requires_grad=False)
    return encoding_tensors


class ExpressiveEncoderBase(Encoder):
    def __init__(self, suffix: str) -> None:
        super().__init__()
        assert isinstance(suffix, str), 'Suffix must be a string.'
        self.prefix = 'expressive_'
        self.suffix = suffix

    def get_relation_name_of_predicate(self, predicate: mm.Predicate, is_goal_predicate: bool, is_true: bool) -> str:
        return self.prefix + get_predicate_name(predicate, is_goal_predicate, is_true, self.suffix)


class ExpressiveStateEncoder(ExpressiveEncoderBase):
    """Expressive encoder for planning states.

    This encoder transforms a planning state (which contains atoms — facts that
    are currently true) into nodes and relations for the graph neural network.
    In contrast to StateEncoder, pairs of objects become nodes, and atoms become
    relations between these pairs.
    """

    def __init__(self, suffix: str = '') -> None:
        """Initialize the expressive state encoder.

        Args:
            suffix: Optional suffix to append to relation names to avoid conflicts.
        """
        super().__init__(suffix)
        self.composition_name = self.prefix + 'composition'

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for state atoms.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            Predicate relations followed by namespaced unary type relations.
            Nullary predicates are unary over the original object nodes; all
            other predicates retain the expressive object-pair arity.
        """
        relations = [
            (
                self.get_relation_name_of_predicate(predicate, False, True),
                _expressive_predicate_relation_arity(predicate),
            )
            for predicate in domain.predicates
        ]
        relations.extend(
            (_type_relation_name(type_name, self.suffix, expressive=True), 1)
            for type_name in domain.type_hierarchy
        )
        relations.append((self.composition_name, 3))
        return relations

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        assert isinstance(input_value, mm.State), f'ExpressiveStateEncoder expected a State, got {type(input_value)}'
        mm.learning.encode_expressive_state(
            context,
            input_value,
            suffix=self.suffix,
        )


class ExpressiveGoalEncoder(ExpressiveEncoderBase):
    """Expressive encoder for goal conditions.

    This encoder transforms a goal condition (a conjunctive condition of
    literals that must be satisfied) into relations for the graph neural
    network. It creates goal-specific relations and marks which atoms are true
    or false in the current state relative to the goal. Unlike GoalEncoder, the
    conjunctive goal is encoded using object pairs.
    """

    def __init__(self, suffix: str = '') -> None:
        """Initialize the goal encoder.

        Args:
            suffix: Suffix to append to relation names to avoid conflicts.
        """
        super().__init__(suffix)

    def get_relations(self, domain: mm.Domain) -> list[tuple[str, int]]:
        """Get relations that this encoder will add for goal conditions.

        Args:
            domain: The PDDL domain containing predicate definitions.

        Returns:
            List of (relation_name, arity) pairs for goal predicates,
            including both true and false variants. Nullary predicates are
            unary over the original object nodes.
        """
        relations = []
        relations.extend([
            (
                self.get_relation_name_of_predicate(predicate, True, False),
                _expressive_predicate_relation_arity(predicate),
            )
            for predicate in domain.predicates
        ])
        relations.extend([
            (
                self.get_relation_name_of_predicate(predicate, True, True),
                _expressive_predicate_relation_arity(predicate),
            )
            for predicate in domain.predicates
        ])
        return relations

    def encode(self, input_value: Any, state: mm.State, encoding: 'EncodedLists', context: 'EncodingContext') -> None:
        assert isinstance(input_value, mm.GroundConjunctiveCondition), f'GoalEncoder expected a GroundConjunctiveCondition, got {type(input_value)}'
        mm.learning.encode_expressive_goal(
            context,
            state,
            input_value,
            suffix=self.suffix,
        )
