"""Carrying node embeddings between forward passes of a search driver."""

from collections.abc import Hashable, Sequence

import torch
import torch.nn as nn

from .bases import EncodedTensors


NodeKey = tuple[Hashable, Hashable]


class NodeStateStore(nn.Module):
    """Store of node embeddings keyed by (instance key, node key) across forward passes.

    A search driver that re-encodes a growing structure, such as a partial plan,
    can give every persistent node a key. ``build_initial`` then starts a node from
    its stored embedding when one exists and from a learned vector of its node type
    otherwise, and ``update`` stores the embeddings of keyed nodes after a forward
    pass. Node ids are global per batch and contiguous per instance, so keys and
    types are given per instance in the order of ``encoding.node_sizes``.
    """

    def __init__(self, embedding_size: int, type_names: Sequence[str], detach: bool = True):
        """Initialize the store.

        Args:
            embedding_size: The size of the node embeddings.
            type_names: The node types with a learned initial vector each.
            detach: Whether stored embeddings are detached from the graph that
                produced them. Keep this for search; disable it to backpropagate
                through consecutive forward passes.
        """
        super().__init__()
        assert len(type_names) > 0, "At least one node type is required."
        assert len(set(type_names)) == len(type_names), "Node type names must be unique."
        self._type_names = list(type_names)
        self._type_index = {name: index for index, name in enumerate(self._type_names)}
        # Small random vectors keep the types distinguishable from the first forward pass on.
        self._type_vectors = nn.Parameter(0.1 * torch.randn(len(self._type_names), embedding_size))
        self._detach = detach
        self._store: dict[NodeKey, torch.Tensor] = {}

    @property
    def type_names(self) -> list[str]:
        """The node types with a learned initial vector."""
        return list(self._type_names)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def get(self, key: NodeKey) -> torch.Tensor | None:
        """Return the stored embedding of a key, or None."""
        return self._store.get(key)

    def _check_alignment(self, encoding: EncodedTensors, per_instance: Sequence[Sequence[object]], what: str) -> None:
        node_sizes = encoding.node_sizes.tolist()
        assert len(per_instance) == len(node_sizes), (
            f"{what} must be given for every instance ({len(node_sizes)}), got {len(per_instance)}."
        )
        for instance_index, (values, size) in enumerate(zip(per_instance, node_sizes)):
            assert len(values) == size, (
                f"{what} of instance {instance_index} must have one entry per node ({size}), got {len(values)}."
            )

    def build_initial(
        self,
        encoding: EncodedTensors,
        node_keys: Sequence[Sequence[NodeKey | None]],
        node_types: Sequence[Sequence[str]],
    ) -> torch.Tensor:
        """Build initial node embeddings from stored rows and type vectors.

        Args:
            encoding: The encoding of the batch, for the node count and sizes.
            node_keys: Per instance, the key of every node or None for an unkeyed node.
            node_types: Per instance, the type name of every node.

        Returns:
            Tensor of shape (node_count, embedding_size): a node with a stored key
            gets its stored embedding, every other node the vector of its type.
        """
        self._check_alignment(encoding, node_keys, "node_keys")
        self._check_alignment(encoding, node_types, "node_types")
        type_ids = torch.tensor(
            [self._type_index[type_name] for types in node_types for type_name in types],
            dtype=torch.long,
            device=self._type_vectors.device,
        )
        assert type_ids.numel() == encoding.node_count, "The node types must cover every node of the batch."
        initial = self._type_vectors.index_select(0, type_ids)
        stored_ids: list[int] = []
        stored_rows: list[torch.Tensor] = []
        for node_id, key in enumerate(key for keys in node_keys for key in keys):
            if key is None:
                continue
            row = self._store.get(key)
            if row is not None:
                stored_ids.append(node_id)
                stored_rows.append(row)
        if not stored_ids:
            return initial
        rows = torch.stack(stored_rows).to(device=initial.device, dtype=initial.dtype)
        ids = torch.tensor(stored_ids, dtype=torch.long, device=initial.device)
        return initial.index_copy(0, ids, rows)

    def update(
        self,
        encoding: EncodedTensors,
        node_keys: Sequence[Sequence[NodeKey | None]],
        node_embeddings: torch.Tensor,
    ) -> None:
        """Store the embeddings of every keyed node.

        Args:
            encoding: The encoding of the batch, for the node sizes.
            node_keys: Per instance, the key of every node or None for an unkeyed node.
            node_embeddings: The final node embeddings of the forward pass.
        """
        self._check_alignment(encoding, node_keys, "node_keys")
        keyed = [(node_id, key) for node_id, key in enumerate(key for keys in node_keys for key in keys) if key is not None]
        if not keyed:
            return
        ids = torch.tensor([node_id for node_id, _ in keyed], dtype=torch.long, device=node_embeddings.device)
        rows = node_embeddings.index_select(0, ids)
        if self._detach:
            rows = rows.detach()
        for (_, key), row in zip(keyed, rows.unbind(0)):
            self._store[key] = row

    def forget(self, instance_key: Hashable) -> None:
        """Drop every stored embedding of an instance."""
        for key in [key for key in self._store if key[0] == instance_key]:
            del self._store[key]

    def clear(self) -> None:
        """Drop every stored embedding."""
        self._store.clear()
