from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from .bases import Decoder
from .configs import HyperparameterConfig
from .encoders import EncodedTensors
from .modules import MLP, PooledReadout, Pooling, SumReadout


class ActionScalarDecoder(Decoder):
    """Decoder for scalar values over actions.

    This decoder produces a scalar value for each action by combining action
    embeddings with object embeddings. Commonly used for Q-values or action
    preferences in reinforcement learning applications.
    """

    def __init__(self, hparam_config: 'HyperparameterConfig'):
        """Initialize the action scalar decoder.

        Args:
            config: The hyperparameter configuration containing embedding sizes.
        """
        super().__init__()
        assert isinstance(hparam_config, HyperparameterConfig), "hparam_config must be an instance of HyperparameterConfig."
        self._object_readout = SumReadout(hparam_config.embedding_size, hparam_config.embedding_size)
        self._action_value = MLP(2 * hparam_config.embedding_size, 1)

    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> list[torch.Tensor]:
        """Compute scalar values for each action.

        Args:
            node_embeddings: The node embeddings from the graph neural network.
            encoding: The encoding information containing action and object indices.

        Returns:
            List of tensors, where each tensor contains scalar values for the
            actions in the corresponding input instance.
        """
        action_embeddings = node_embeddings.index_select(0, encoding.action_indices)
        object_embeddings = node_embeddings.index_select(0, encoding.object_indices)
        object_aggregation = self._object_readout.forward(object_embeddings, encoding.object_sizes)
        object_aggregation = object_aggregation.repeat_interleave(
            encoding.action_sizes,
            dim=0,
            output_size=action_embeddings.shape[0],
        )
        values = self._action_value.forward(torch.cat((action_embeddings, object_aggregation), dim=1))
        if len(encoding.action_counts) == 1:
            return [values.view(-1)]
        return [action_values.view(-1) for action_values in values.split(encoding.action_counts)]  # type: ignore


class ActionEmbeddingDecoder(Decoder):
    """Decoder for embeddings over actions.

    This decoder extracts the learned embeddings for action nodes without
    any additional processing. Useful when you want to access the raw
    action representations learned by the graph neural network.
    """

    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> tuple[torch.Tensor, ...]:
        """Extract action node embeddings.

        Args:
            node_embeddings: The node embeddings from the graph neural network.
            encoding: The encoding information containing action indices.

        Returns:
            Tuple of tensors, each containing the embeddings of all action nodes.
        """
        action_embeddings = node_embeddings.index_select(0, encoding.action_indices)
        if len(encoding.action_counts) == 1:
            return (action_embeddings,)
        return action_embeddings.split_with_sizes(encoding.action_counts)


class ObjectsScalarDecoder(Decoder):
    """Decoder for scalar values over objects.

    This decoder produces a single scalar value by aggregating all object
    embeddings within each input instance. Commonly used for state value
    estimation or global state assessment.
    """

    def __init__(self, hparam_config: 'HyperparameterConfig'):
        """Initialize the objects scalar decoder.

        Args:
            config: The hyperparameter configuration containing embedding sizes.
        """
        super().__init__()
        assert isinstance(hparam_config, HyperparameterConfig), "hparam_config must be an instance of HyperparameterConfig."
        self._object_readout = SumReadout(hparam_config.embedding_size, hparam_config.embedding_size)
        self._state_value = MLP((1) * hparam_config.embedding_size, 1)

    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> torch.Tensor:
        """Compute scalar values by aggregating object embeddings.

        Args:
            node_embeddings: The node embeddings from the graph neural network.
            encoding: The encoding information containing object indices and sizes.

        Returns:
            Tensor containing one scalar value per input instance.
        """
        object_embeddings = node_embeddings.index_select(0, encoding.object_indices)
        object_aggregation = self._object_readout(object_embeddings, encoding.object_sizes)
        return self._state_value(object_aggregation).view(-1)


class ObjectsEmbeddingDecoder(Decoder):
    """Decoder for embeddings over objects.

    This decoder extracts the learned embeddings for object nodes without
    any additional processing. Useful when you want to access the raw
    object representations learned by the graph neural network.
    """

    def forward(self, node_embeddings: torch.Tensor, encoding: 'EncodedTensors') -> tuple[torch.Tensor, ...]:
        """Extract object node embeddings.

        Args:
            node_embeddings: The node embeddings from the graph neural network.
            encoding: The encoding information containing object indices.

        Returns:
            Tuple of tensors, each containing the embeddings of object nodes for an input instance.
        """
        object_embeddings = node_embeddings.index_select(0, encoding.object_indices)
        if encoding.object_sizes.numel() == 1:
            return (object_embeddings,)
        return object_embeddings.split_with_sizes(encoding.object_sizes.tolist())


def _instance_offsets(encoding: EncodedTensors) -> torch.Tensor:
    """Return the exclusive end offset of every instance's node id range."""
    return encoding.node_sizes.cumsum(0)


def _instance_of_nodes(node_ids: torch.Tensor, encoding: EncodedTensors) -> torch.Tensor:
    """Map global node ids to the index of the instance owning them.

    Node ids are contiguous per instance, so the instance of a node is the number
    of cumulative node-size boundaries at or below it.
    """
    return torch.bucketize(node_ids, _instance_offsets(encoding), right=True)


def _split_by_instance(values: torch.Tensor, instance_ids: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    """Split rows by instance, keeping the original row order within each instance."""
    if batch_size == 1:
        return [values]
    counts = torch.bincount(instance_ids, minlength=batch_size)
    order = torch.sort(instance_ids, stable=True).indices
    return list(values.index_select(0, order).split(counts.tolist()))


def _global_node_ids(per_instance: Sequence[Sequence[int] | torch.Tensor], device: torch.device) -> tuple[torch.Tensor, list[int]]:
    """Concatenate per-instance node id lists into one device tensor with the sizes."""
    tensors = [torch.as_tensor(ids, dtype=torch.long).view(-1) for ids in per_instance]
    sizes = [tensor.numel() for tensor in tensors]
    if tensors:
        return torch.cat(tensors).to(device), sizes
    return torch.empty((0,), dtype=torch.long, device=device), sizes


# The context name of the object nodes of an instance; every other context name is a key of the encoding extras.
OBJECTS_CONTEXT = 'objects'


def _context_nodes(context: str, encoding: EncodedTensors, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The global node ids of a context, instance after instance, and the number of them per instance.

    A context is a group of nodes per instance: ``'objects'`` names the object
    nodes, any other name is a key of ``encoding.extras`` holding one sequence
    (list or LongTensor) of global node ids per instance.
    """
    if context == OBJECTS_CONTEXT:
        return encoding.object_indices, encoding.object_sizes
    assert context in encoding.extras, f"The context '{context}' is neither 'objects' nor a key of the encoding extras."
    per_instance = encoding.extras[context]
    batch_size = encoding.node_sizes.numel()
    assert len(per_instance) == batch_size, (
        f"extras['{context}'] must list the nodes of every instance ({batch_size}), got {len(per_instance)}."
    )
    node_ids, sizes = _global_node_ids(per_instance, device)
    return node_ids, torch.tensor(sizes, dtype=torch.long, device=device)


def _pooled_contexts(
    readouts: Sequence[PooledReadout],
    contexts: Sequence[str],
    node_embeddings: torch.Tensor,
    encoding: EncodedTensors,
) -> list[torch.Tensor]:
    """One (batch, output_size) tensor per context: its nodes pooled per instance and transformed by its readout."""
    pooled: list[torch.Tensor] = []
    for readout, context in zip(readouts, contexts):
        node_ids, sizes = _context_nodes(context, encoding, node_embeddings.device)
        pooled.append(readout(node_embeddings.index_select(0, node_ids), sizes))
    return pooled


@dataclass
class GroupedLogProbs:
    """Log-probabilities of members competing within groups for one input instance.

    Rows are in the order of the relation they were decoded from, filtered to the
    instance. ``member_indices`` and ``group_indices`` are global node ids.
    """

    member_indices: torch.Tensor
    group_indices: torch.Tensor
    log_probs: torch.Tensor

    def _members_by_group(self) -> dict[int, list[int]]:
        rows_by_group: dict[int, list[int]] = {}
        for row, group_id in enumerate(self.group_indices.tolist()):
            rows_by_group.setdefault(group_id, []).append(row)
        return rows_by_group

    def argmax_per_group(self) -> dict[int, int]:
        """Return the most probable member node id of every group node id."""
        members = self.member_indices.tolist()
        log_probs = self.log_probs.detach().cpu()
        result: dict[int, int] = {}
        for group_id, rows in self._members_by_group().items():
            best_row = rows[int(log_probs[rows].argmax())]
            result[group_id] = members[best_row]
        return result

    def sample_per_group(self, generator: torch.Generator | None = None) -> dict[int, int]:
        """Sample one member node id per group node id from the group's distribution.

        Sampling runs on the CPU so that a CPU ``generator`` can be used regardless
        of the device of the log-probabilities.
        """
        members = self.member_indices.tolist()
        probs = self.log_probs.detach().cpu().exp()
        result: dict[int, int] = {}
        for group_id, rows in self._members_by_group().items():
            sampled = int(torch.multinomial(probs[rows], 1, generator=generator))
            result[group_id] = members[rows[sampled]]
        return result


class GroupedSoftmaxDecoder(Decoder):
    """Decoder for a softmax over members competing within groups.

    The decoder reads a relation whose rows pair member nodes with group nodes,
    for example ``resolves(resolver, flaw)``. Every row is scored by an MLP over
    the member embedding (and the group embedding), and the scores of rows with
    the same group node are normalized by a log-softmax. Rows of the same group
    may come from different relation entries but a group must belong to one
    instance, which holds since node ids are allocated per instance.

    The score can additionally see pooled contexts: for every name in
    ``contexts`` a :class:`PooledReadout` over that node group of the row's
    instance (``'objects'`` or a key of ``encoding.extras`` listing global node
    ids per instance), and with ``sibling_context`` a PooledReadout over the
    member embeddings of all rows of the same group, the row itself included.
    """

    # Class-level defaults: a decoder unpickled from a checkpoint written before contexts existed (see
    # RelationalGraphNeuralNetwork.save, which stores the decoders themselves) has no such attributes.
    _contexts: tuple[str, ...] = ()
    _sibling_context: bool = False

    def __init__(
        self,
        hparam_config: HyperparameterConfig,
        relation_name: str,
        member_position: int = 0,
        group_position: int = 1,
        include_group_embedding: bool = True,
        arity: int | None = None,
        contexts: Sequence[str] = (),
        sibling_context: bool = False,
        pooling: Pooling = 'sum',
    ):
        """Initialize the grouped softmax decoder.

        Args:
            hparam_config: The hyperparameter configuration containing embedding sizes.
            relation_name: The relation whose rows are the scored (member, group) pairs.
            member_position: The position of the member node id in a relation row.
            group_position: The position of the group node id in a relation row.
            include_group_embedding: Whether the score sees the group embedding
                next to the member embedding.
            arity: The arity of the relation. Defaults to the smallest arity that
                holds both positions.
            contexts: The node groups of the instance whose pooled embeddings the
                score sees: ``'objects'`` or keys of ``encoding.extras``.
            sibling_context: Whether the score sees the pooled member embeddings
                of the rows competing in the same group.
            pooling: How the contexts and the siblings are pooled.
        """
        super().__init__()
        assert isinstance(hparam_config, HyperparameterConfig), "hparam_config must be an instance of HyperparameterConfig."
        assert member_position != group_position, "member_position and group_position must differ."
        self._relation_name = relation_name
        self._member_position = member_position
        self._group_position = group_position
        self._include_group_embedding = include_group_embedding
        self._arity = max(member_position, group_position) + 1 if arity is None else arity
        assert self._arity > max(member_position, group_position), "arity must hold both positions."
        assert not isinstance(contexts, str), "contexts must be a sequence of context names, not a single string."
        assert len(set(contexts)) == len(contexts), "The contexts must be distinct."
        self._contexts = tuple(contexts)
        self._sibling_context = sibling_context
        embedding_size = hparam_config.embedding_size
        # Without contexts the module has exactly the parameters it had before contexts existed.
        if self._contexts:
            self._context_readouts = torch.nn.ModuleList(
                PooledReadout(embedding_size, embedding_size, pooling) for _ in self._contexts
            )
        if sibling_context:
            self._sibling_readout = PooledReadout(embedding_size, embedding_size, pooling)
        input_count = (2 if include_group_embedding else 1) + len(self._contexts) + (1 if sibling_context else 0)
        self._score = MLP(input_count * embedding_size, 1)

    def forward(self, node_embeddings: torch.Tensor, encoding: EncodedTensors) -> list[GroupedLogProbs]:
        """Compute grouped log-probabilities for every input instance.

        Args:
            node_embeddings: The node embeddings from the graph neural network.
            encoding: The encoding information containing the relation and node sizes.

        Returns:
            One GroupedLogProbs per input instance, in batch order.
        """
        batch_size = encoding.node_sizes.numel()
        relation = encoding.flattened_relations.get(self._relation_name)
        if relation is None or relation.numel() == 0:
            empty_ids = node_embeddings.new_empty((0,), dtype=torch.long)
            empty_values = node_embeddings.new_empty((0,))
            return [GroupedLogProbs(empty_ids, empty_ids, empty_values) for _ in range(batch_size)]
        rows = relation.view(-1, self._arity).long()
        member_ids = rows[:, self._member_position]
        group_ids = rows[:, self._group_position]

        instance_ids = _instance_of_nodes(group_ids, encoding)

        member_embeddings = node_embeddings.index_select(0, member_ids)
        score_inputs = [member_embeddings]
        if self._include_group_embedding:
            score_inputs.append(node_embeddings.index_select(0, group_ids))
        if self._contexts:
            readouts: list[PooledReadout] = list(self._context_readouts)  # type: ignore[arg-type]
            for pooled in _pooled_contexts(readouts, self._contexts, node_embeddings, encoding):
                score_inputs.append(pooled.index_select(0, instance_ids))
        if self._sibling_context:
            # Dense group numbers: pooling into a buffer of one row per node would transform mostly empty rows.
            groups, group_of_row = torch.unique(group_ids, return_inverse=True)
            siblings = self._sibling_readout.forward_indexed(member_embeddings, group_of_row, groups.numel())
            score_inputs.append(siblings.index_select(0, group_of_row))
        scores = self._score(score_inputs[0] if len(score_inputs) == 1 else torch.cat(score_inputs, dim=1)).view(-1)

        # Segment log-softmax: shift by the maximum score of each group, which does not change the result, then
        # normalize by the log of the summed exponentials. Group ids are node ids, so a node-sized buffer suffices.
        group_maxima = scores.new_full((encoding.node_count,), float('-inf'))
        group_maxima = group_maxima.index_reduce(0, group_ids, scores.detach(), 'amax', include_self=True)
        shifted = scores - group_maxima.index_select(0, group_ids)
        group_sums = scores.new_zeros((encoding.node_count,)).index_add(0, group_ids, shifted.exp())
        log_probs = shifted - group_sums.index_select(0, group_ids).log()

        stacked = torch.stack((member_ids, group_ids), dim=1)
        ids_per_instance = _split_by_instance(stacked, instance_ids, batch_size)
        log_probs_per_instance = _split_by_instance(log_probs, instance_ids, batch_size)
        return [
            GroupedLogProbs(ids[:, 0], ids[:, 1], values)
            for ids, values in zip(ids_per_instance, log_probs_per_instance)
        ]


class NodeScalarDecoder(Decoder):
    """Decoder for scalar values over nodes listed in the encoding extras.

    Custom encoders list the node ids to score under ``encoding.extras[key]`` as
    one sequence (list or LongTensor of global node ids) per input instance. Each
    listed node is scored by an MLP over its embedding and the pooled contexts of
    its instance. By default the only context is a sum readout of the instance's
    object embeddings, like ActionScalarDecoder uses for actions; ``contexts``
    names other node groups (``'objects'`` or keys of ``encoding.extras`` listing
    global node ids per instance), each pooled by its own :class:`PooledReadout`.
    """

    # Class-level defaults: a decoder unpickled from a checkpoint written before contexts existed (see
    # RelationalGraphNeuralNetwork.save, which stores the decoders themselves) has no such attributes.
    _contexts: tuple[str, ...] = (OBJECTS_CONTEXT,)
    _legacy_object_readout: bool = True

    def __init__(
        self,
        hparam_config: HyperparameterConfig,
        extras_key: str,
        contexts: Sequence[str] = (OBJECTS_CONTEXT,),
        pooling: Pooling = 'sum',
    ):
        """Initialize the node scalar decoder.

        Args:
            hparam_config: The hyperparameter configuration containing embedding sizes.
            extras_key: The key under ``encoding.extras`` listing the node ids per instance.
            contexts: The node groups of the instance whose pooled embeddings the
                value sees; empty to score a node from its embedding alone.
            pooling: How the contexts are pooled.
        """
        super().__init__()
        assert isinstance(hparam_config, HyperparameterConfig), "hparam_config must be an instance of HyperparameterConfig."
        assert not isinstance(contexts, str), "contexts must be a sequence of context names, not a single string."
        assert len(set(contexts)) == len(contexts), "The contexts must be distinct."
        self._extras_key = extras_key
        self._contexts = tuple(contexts)
        embedding_size = hparam_config.embedding_size
        # The default keeps the module and the parameter names it had before contexts existed, so that checkpoints
        # of it keep loading.
        self._legacy_object_readout = self._contexts == (OBJECTS_CONTEXT,) and pooling == 'sum'
        if self._legacy_object_readout:
            self._object_readout = SumReadout(embedding_size, embedding_size)
        elif self._contexts:
            self._context_readouts = torch.nn.ModuleList(
                PooledReadout(embedding_size, embedding_size, pooling) for _ in self._contexts
            )
        self._node_value = MLP((1 + len(self._contexts)) * embedding_size, 1)

    def forward(self, node_embeddings: torch.Tensor, encoding: EncodedTensors) -> list[torch.Tensor]:
        """Compute scalar values for the listed nodes of each instance.

        Args:
            node_embeddings: The node embeddings from the graph neural network.
            encoding: The encoding whose extras list the nodes to score.

        Returns:
            List of 1-D tensors, one per input instance, with one value per listed node.
        """
        batch_size = encoding.node_sizes.numel()
        per_instance = encoding.extras[self._extras_key]
        assert len(per_instance) == batch_size, (
            f"extras['{self._extras_key}'] must list the nodes of every instance ({batch_size}), got {len(per_instance)}."
        )
        node_ids, sizes = _global_node_ids(per_instance, node_embeddings.device)
        listed_embeddings = node_embeddings.index_select(0, node_ids)
        if self._legacy_object_readout:
            object_embeddings = node_embeddings.index_select(0, encoding.object_indices)
            pooled_contexts = [self._object_readout(object_embeddings, encoding.object_sizes)]
        elif self._contexts:
            readouts: list[PooledReadout] = list(self._context_readouts)  # type: ignore[arg-type]
            pooled_contexts = _pooled_contexts(readouts, self._contexts, node_embeddings, encoding)
        else:
            pooled_contexts = []
        value_inputs = [listed_embeddings]
        if pooled_contexts:
            listed_sizes = torch.tensor(sizes, dtype=torch.long, device=node_embeddings.device)
            for pooled in pooled_contexts:
                value_inputs.append(pooled.repeat_interleave(listed_sizes, dim=0, output_size=listed_embeddings.shape[0]))
        values = self._node_value(value_inputs[0] if len(value_inputs) == 1 else torch.cat(value_inputs, dim=1)).view(-1)
        if batch_size == 1:
            return [values]
        return list(values.split(sizes))


class GlobalScalarDecoder(Decoder):
    """Decoder for one scalar per instance from a sum readout over its nodes.

    By default the readout sums the object embeddings of each instance; with
    ``nodes='all'`` it sums every node of the instance, including auxiliary nodes
    such as those of a partial plan.
    """

    def __init__(self, hparam_config: HyperparameterConfig, nodes: Literal['objects', 'all'] = 'objects'):
        """Initialize the global scalar decoder.

        Args:
            hparam_config: The hyperparameter configuration containing embedding sizes.
            nodes: Which nodes of an instance the readout sums.
        """
        super().__init__()
        assert isinstance(hparam_config, HyperparameterConfig), "hparam_config must be an instance of HyperparameterConfig."
        assert nodes in ('objects', 'all'), "nodes must be 'objects' or 'all'."
        self._nodes = nodes
        self._readout = SumReadout(hparam_config.embedding_size, hparam_config.embedding_size)
        self._value = MLP(hparam_config.embedding_size, 1)

    def forward(self, node_embeddings: torch.Tensor, encoding: EncodedTensors) -> torch.Tensor:
        """Compute one scalar per input instance.

        Args:
            node_embeddings: The node embeddings from the graph neural network.
            encoding: The encoding information containing node and object sizes.

        Returns:
            Tensor of shape (batch,) with one value per input instance.
        """
        if self._nodes == 'objects':
            embeddings = node_embeddings.index_select(0, encoding.object_indices)
            aggregation = self._readout(embeddings, encoding.object_sizes)
        else:
            aggregation = self._readout(node_embeddings, encoding.node_sizes)
        return self._value(aggregation).view(-1)
