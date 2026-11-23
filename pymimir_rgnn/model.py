import copy
import gc
import threading
import warnings
import weakref
import torch
import torch.nn as nn

from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
import pymimir as mm
from torch.overrides import TorchFunctionMode
from typing import Any, Callable, Iterator, Union

from .bases import Encoder, QuantizationRecord
from .configs import HyperparameterConfig, ModuleConfig
from .decoders import Decoder
from .encoders import EncodedTensors, get_input_from_encoders
from .modules import MLP, ChannelwiseAffine, SumReadout
from .utils import binarize

# Replaying the layers from a CUDA graph pays for recording it from this many layers on.
MIN_REPLAYED_LAYERS = 4

# Whether forward passes with gradients replay the layers as well. Such a forward pass supports one backward pass, so
# neither a second one with retain_graph=True nor differentiating twice with create_graph=True; these need this off.
REPLAY_TRAINING = True


@dataclass
class _ReplayResources:
    """What recordings of layers share on a device in a thread; kept outside the modules, which then stay picklable."""
    stream: torch.cuda.Stream
    # Every recording allocates from this pool: a recording's own pool would stay reserved after it is discarded. A
    # recording allocates only tensors that it frees before it ends, so a replay uses the memory only while it runs,
    # and recordings may share it as long as no two replays run at once, see `wait_for_other_replays`.
    memory_pool: Any
    latest_graph: torch.cuda.CUDAGraph | None = None
    # The stream of the latest replays.
    latest_replay_stream: torch.cuda.Stream | None = None
    warmed_up_backward: bool = False

    def wait_for_other_replays(self) -> None:
        """Make the current stream wait for the latest replays if they ran on another stream."""
        stream = torch.cuda.current_stream(self.stream.device)
        if self.latest_replay_stream is not None and self.latest_replay_stream != stream:
            stream.wait_stream(self.latest_replay_stream)
        self.latest_replay_stream = stream


# Per thread, the _ReplayResources of each device.
_replay_resources = threading.local()


@contextmanager
def _recording(graph: torch.cuda.CUDAGraph, resources: _ReplayResources) -> Iterator[None]:
    """Record the CUDA calls of this thread into a graph, on the stream of the replay resources."""
    current_stream = torch.cuda.current_stream(resources.stream.device)
    resources.stream.wait_stream(current_stream)
    # A garbage collection during the recording can free objects whose CUDA calls invalidate it.
    collecting_garbage = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.stream(resources.stream):
            # Only CUDA calls of this thread may invalidate the recording, not those of other threads.
            graph.capture_begin(pool=resources.memory_pool, capture_error_mode='thread_local')
            try:
                yield
            except BaseException:
                # A recording that is not ended leaves the stream recording, which breaks every later CUDA call of the
                # thread. Ending a failed recording fails as well; the layer's error is the one to report.
                try:
                    graph.capture_end()
                except RuntimeError:
                    pass
                if resources.latest_graph is None:
                    # No recording may use the pool anymore, after which PyTorch cannot record into it again.
                    resources.memory_pool = torch.cuda.graph_pool_handle()
                raise
            graph.capture_end()
    finally:
        if collecting_garbage:
            gc.enable()
    current_stream.wait_stream(resources.stream)
    # The memory pool is released once no recording uses it, after which PyTorch cannot record into it again.
    resources.latest_graph = graph


def _map_tensors(function: Callable[[torch.Tensor], Any], value: Any) -> Any:
    """Apply a function to the tensors in arguments or results of torch functions, also in lists, tuples and dicts."""
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, (list, tuple)):
        items = [_map_tensors(function, item) for item in value]
        # Named tuples take their items as arguments.
        return type(value)(*items) if hasattr(value, '_fields') else type(value)(items)
    if isinstance(value, dict):
        mapped = copy.copy(value)
        for key, item in value.items():
            mapped[key] = _map_tensors(function, item)
        return mapped
    return value


class _LayerBoundary(TorchFunctionMode):
    """Tracks the tensors that a layer reads from outside of it, and replaces those that require gradients by leaves.

    A recording reads these tensors until its last replay, after which, for instance, `MessageFunction.cleanup` may
    have released them. Backpropagation through a recorded layer must stop at the leaves, since the autograd nodes
    beyond them belong to graphs and streams outside of the recording.
    """

    def __init__(self, layer_input: torch.Tensor, leaves: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None):
        """Start tracking.

        Args:
            layer_input: The input of the layer, which is not from outside of it.
            leaves: If given, the leaves of the outside tensors so far by their ids, to which new ones are added.
        """
        super().__init__()
        self._inside: dict[int, weakref.ref] = {}
        self.outside: dict[int, torch.Tensor] = {}
        self.leaves = leaves
        self.new_leaves = False
        self._mark_inside(layer_input)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        args, kwargs = _map_tensors(self._enter, (args, kwargs or {}))
        result = func(*args, **kwargs)
        _map_tensors(self._mark_inside, result)
        return result

    def _mark_inside(self, tensor: torch.Tensor) -> torch.Tensor:
        key = id(tensor)
        if key not in self._inside:
            # Forgotten once the tensor is freed, before another one can have its id.
            self._inside[key] = weakref.ref(tensor, lambda _: self._inside.pop(key, None))
        return tensor

    def _enter(self, tensor: torch.Tensor) -> torch.Tensor:
        key = id(tensor)
        if key in self._inside:
            return tensor
        self.outside.setdefault(key, tensor)
        if self.leaves is None or not tensor.requires_grad:
            return tensor
        if key not in self.leaves:
            self.leaves[key] = (tensor, tensor.detach().requires_grad_())
            self.new_leaves = True
        return self.leaves[key][1]


class _Unrecordable(Exception):
    """The layer cannot be recorded for training; it runs without a recording."""


class _TrainingRecording:
    """The CUDA graphs of one layer and of backpropagating through it, which run every layer of one forward pass.

    The backward pass recomputes each layer from its input, as with activation checkpointing: the replays of a layer
    overwrite its intermediate tensors.
    """

    def __init__(self, stack: 'RelationalLayerStackModule', input: EncodedTensors, node_embeddings: torch.Tensor,
                 resources: _ReplayResources):
        """Record the graphs.

        Args:
            stack: The layers.
            input: The encoded graph input, as given to `RelationalLayerStackModule.forward`.
            node_embeddings: The node embeddings before the first layer.
            resources: The replay resources of the device.

        Raises:
            _Unrecordable: If the backward pass cannot be recorded, or the layer reads a tensor that requires gradients
                only when gradients are enabled.
        """
        self._num_layers = stack._config.num_layers
        self._resources = resources
        self._embeddings = torch.empty_like(node_embeddings)
        self._forward_graph = torch.cuda.CUDAGraph()
        forward_boundary = _LayerBoundary(self._embeddings)
        with _recording(self._forward_graph, resources), torch.no_grad(), forward_boundary:
            updated_embeddings, _ = stack._layer(0, input, self._embeddings)
            self._embeddings.copy_(updated_embeddings)
            del updated_embeddings
        leaves = {key: (tensor, tensor.detach().requires_grad_())
                  for key, tensor in forward_boundary.outside.items() if tensor.requires_grad}
        self.outside_tensors = [tensor for tensor, _ in leaves.values()]
        # Read by the graphs until their last replay.
        self._kept_alive: tuple[Any, ...] = (forward_boundary.outside,)
        self.backpropagates = bool(leaves) or node_embeddings.requires_grad
        if self.backpropagates:
            self._record_backward_pass(stack, input, leaves)

    def _record_backward_pass(self, stack: 'RelationalLayerStackModule', input: EncodedTensors,
                              leaves: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Record recomputing the layer and backpropagating through it, accumulating the gradients of the leaves."""
        self._layer_input = torch.empty_like(self._embeddings)
        self._gradient = torch.empty_like(self._embeddings)
        # Outside of the pool, since autograd may keep them.
        self._accumulated_gradients = [torch.zeros_like(tensor) for tensor, _ in leaves.values()]
        self._backward_graph = torch.cuda.CUDAGraph()
        try:
            with _recording(self._backward_graph, self._resources):
                input_gradient, gradients, boundary = stack._backpropagate_layer(
                    input, self._layer_input.detach().requires_grad_(), leaves, self._gradient)
                if boundary.new_leaves:
                    raise _Unrecordable("the layer reads tensors that require gradients only with gradients enabled")
                self._uses = [gradient is not None for gradient in gradients]
                used = [(accumulated, gradient) for accumulated, gradient in zip(self._accumulated_gradients, gradients)
                        if gradient is not None]
                if used:
                    torch._foreach_add_([accumulated for accumulated, _ in used], [gradient for _, gradient in used])
                if input_gradient is None:
                    self._gradient.zero_()
                else:
                    self._gradient.copy_(input_gradient)
                del input_gradient, gradients, used
        except RuntimeError as error:
            raise _Unrecordable(f"recording its backward pass failed: {error}") from error
        self._kept_alive += (boundary.outside, leaves)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Run every layer, keeping the input of each for the backward pass if there is one."""
        if self.backpropagates:
            self._saved_embeddings = node_embeddings.new_empty((self._num_layers,) + tuple(node_embeddings.shape))
        self._embeddings.copy_(node_embeddings)
        self._resources.wait_for_other_replays()
        for layer_index in range(self._num_layers):
            if self.backpropagates:
                self._saved_embeddings[layer_index].copy_(self._embeddings)
            self._forward_graph.replay()
        # Not replayed again; the output would otherwise keep the recording alive through its autograd node.
        output = self._embeddings
        del self._embeddings
        return output

    def backward(self, gradient: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Backpropagate through every layer, last first.

        Returns:
            The gradient of the node embeddings before the first layer, and of each of `outside_tensors`.
        """
        self._gradient.copy_(gradient)
        self._resources.wait_for_other_replays()
        for layer_index in reversed(range(self._num_layers)):
            self._layer_input.copy_(self._saved_embeddings[layer_index])
            self._backward_graph.replay()
        del self._saved_embeddings, self._kept_alive
        # Neither buffer is replayed into again, so autograd may keep them.
        return (self._gradient, *(accumulated if used else None
                                  for accumulated, used in zip(self._accumulated_gradients, self._uses)))


class _ReplayedLayers(torch.autograd.Function):
    """Runs the layers of a `_TrainingRecording` and backpropagates through them."""

    @staticmethod
    def forward(ctx, recording: _TrainingRecording, node_embeddings: torch.Tensor,  # type: ignore[override]
                *outside_tensors: torch.Tensor) -> torch.Tensor:
        # The outside tensors are inputs only so that autograd passes their gradients on.
        ctx.recording = recording
        return recording.forward(node_embeddings)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        recording: _TrainingRecording | None = ctx.recording
        if recording is None:
            raise RuntimeError("Layers replayed from a CUDA graph support only one backward pass; "
                               "set pymimir_rgnn.model.REPLAY_TRAINING = False for more.")
        ctx.recording = None
        return (None, *recording.backward(gradient.contiguous()))


class ForwardState:
    """Container for forward pass state and readout functions.

    This class encapsulates the state of a forward pass through the R-GNN,
    including the current layer index and cached readout functions for
    extracting outputs at any layer.
    """

    def __init__(
        self,
        layer_index: int,
        readouts: dict[str, Callable[[], Any]],
        quantization_records: tuple[QuantizationRecord, ...] = (),
    ):
        """Initialize the forward state.

        Args:
            layer_index: The current layer index in the forward pass.
            readouts: Dictionary of named readout functions for extracting outputs.
        """
        self._layer_index = layer_index
        self._readouts = readouts
        self._quantization_records = quantization_records

    def get_layer_index(self) -> int:
        """Get the current layer index.

        Returns:
            The current layer index in the forward pass.
        """
        return self._layer_index

    def readout(self, name: str) -> Any:
        """Extract output using a named readout function.

        Args:
            name: The name of the readout function to use.

        Returns:
            The output from the specified readout function.
        """
        return self._readouts[name]()

    def quantization_records(self) -> tuple[QuantizationRecord, ...]:
        """Return quantization records associated with this forward state."""
        return self._quantization_records

class RelationalLayerModule(nn.Module):
    """Message passing module for relational graph neural networks.

    This module implements the core message passing operations for R-GNNs,
    including message computation, aggregation, and node updates.
    """

    def __init__(self, hparam_config: HyperparameterConfig, module_config: ModuleConfig):
        """Initialize the message passing module.

        Args:
            hparam_config: The hyperparameter configuration.
            module_config: The module configuration specifying aggregation,
                         message, and update functions.
        """
        super().__init__()  # type: ignore
        self._embedding_size = hparam_config.embedding_size
        self._aggregation = module_config.aggregation_function
        self._message = module_config.message_function
        self._update = module_config.update_function
        self._relation_mlps = nn.ModuleDict()

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform one step of message passing.

        Args:
            node_embeddings: The current node embeddings.
            relations: Dictionary mapping relation names to their argument indices.

        Returns:
            Updated node embeddings after message passing.
        """
        messages, indices = self._message.forward(node_embeddings, relations)
        aggregated_messages = self._aggregation.forward(node_embeddings, messages, indices)
        return self._update.forward(node_embeddings, aggregated_messages)


def _center_per_graph(updates: torch.Tensor, node_sizes: torch.Tensor) -> torch.Tensor:
    """Subtract from each node's update the mean update of the nodes of its graph.

    The mean is taken per graph, so a graph's embeddings do not depend on the other graphs of the batch.
    """
    graphs = torch.repeat_interleave(torch.arange(node_sizes.numel(), device=node_sizes.device), node_sizes,
                                     output_size=updates.shape[0])
    sums = updates.new_zeros((node_sizes.numel(), updates.shape[1])).index_add(0, graphs, updates)
    means = sums / node_sizes.unsqueeze(1)
    return updates - means.index_select(0, graphs)


class RelationalLayerStackModule(nn.Module):
    """Module that implements multiple layers of relational message passing.

    This module orchestrates the execution of multiple message passing layers,
    handling global readout, normalization, and other layer-level operations.
    """

    def __init__(self, hparam_config: HyperparameterConfig, module_config: ModuleConfig):
        """Initialize the relational layers module.

        Args:
            hparam_config: The hyperparameter configuration.
            module_config: The module configuration.
        """
        super().__init__()  # type: ignore
        self._config = hparam_config
        self._relation_network = RelationalLayerModule(hparam_config, module_config)
        self._message = module_config.message_function
        self._latest_quantization_records: tuple[QuantizationRecord, ...] = ()
        self._latest_layer_quantization_records: tuple[QuantizationRecord, ...] = ()
        # Whether recording the layers for training failed, which is then not attempted again.
        self._unrecordable_for_training = False
        if hparam_config.global_readout:
            self._global_readout = SumReadout(hparam_config.embedding_size, hparam_config.embedding_size)
            self._global_update = MLP(2 * hparam_config.embedding_size, hparam_config.embedding_size)
        if hparam_config.normalize_updates:
            self._update_normalization: nn.Module
            if hparam_config.channelwise_normalization:
                self._update_normalization = ChannelwiseAffine(hparam_config.embedding_size)
            else:
                self._update_normalization = nn.LayerNorm(hparam_config.embedding_size)
        self._hooks: list[Callable[[int, torch.Tensor], None]] = []

    def _notify_hooks(self, iteration: int, embeddings: torch.Tensor) -> None:
        """Notify all registered hooks of the current layer state.

        Args:
            iteration: The current layer iteration.
            embeddings: The current node embeddings.
        """
        for hook in self._hooks:
            hook(iteration, embeddings)

    def quantization_records(self) -> tuple[QuantizationRecord, ...]:
        """Return quantization records from the latest full forward pass."""
        return self._latest_quantization_records

    def layer_quantization_records(self) -> tuple[QuantizationRecord, ...]:
        """Return quantization records from the latest completed layer."""
        return self._latest_layer_quantization_records

    def add_hook(self, hook_func: Callable[[int, torch.Tensor], None]) -> None:
        """Add a hook function to be called at each layer.

        Args:
            hook_func: Function to call with (iteration, embeddings) at each layer.
        """
        self._hooks.append(hook_func)

    def clear_hooks(self) -> None:
        """Remove all registered hook functions."""
        self._hooks.clear()

    def forward(self, input: EncodedTensors, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Run multiple layers of message passing.

        Args:
            input: The encoded graph input containing relations and node information.
            node_embeddings: Initial node embeddings for the encoded graph.

        Returns:
            Final node embeddings after all message passing layers.
        """
        self._message.setup(input.flattened_relations)
        try:
            self._latest_quantization_records = ()
            self._latest_layer_quantization_records = ()
            if self._can_replay_layers(node_embeddings):
                if not torch.is_grad_enabled():
                    return self._replay_layers(input, node_embeddings)
                recording = self._record_for_training(input, node_embeddings)
                if recording is not None and not recording.backpropagates:
                    return recording.forward(node_embeddings)
                if recording is not None:
                    return _ReplayedLayers.apply(recording, node_embeddings, *recording.outside_tensors)
            for layer_index in range(self._config.num_layers):
                node_embeddings, layer_records = self._layer(layer_index, input, node_embeddings)
                self._latest_layer_quantization_records = tuple(layer_records)
                self._latest_quantization_records += self._latest_layer_quantization_records
                self._notify_hooks(layer_index, node_embeddings)
            return node_embeddings
        finally:
            self._message.cleanup()

    def _layer(self, layer_index: int, input: EncodedTensors,
               node_embeddings: torch.Tensor) -> tuple[torch.Tensor, list[QuantizationRecord]]:
        """Run one layer of message passing.

        Args:
            layer_index: The index of the layer, for the quantization records.
            input: The encoded graph input containing relations and node information.
            node_embeddings: The node embeddings before the layer.

        Returns:
            The node embeddings after the layer, and the layer's quantization records.
        """
        self._message.begin_quantization_recording(layer_index)
        updates = self._relation_network(node_embeddings, input.flattened_relations)
        layer_records = list(self._message.quantization_records())

        if self._config.center_updates:
            updates = _center_per_graph(updates, input.node_sizes)
        if self._config.normalize_updates:
            updates = self._update_normalization(updates)
        if self._config.global_readout:
            global_embeddings = self._global_readout(node_embeddings, input.node_sizes)
            per_node_global_embeddings = global_embeddings.repeat_interleave(
                input.node_sizes,
                dim=0,
                output_size=node_embeddings.shape[0],
            )
            global_inputs = torch.cat((node_embeddings, per_node_global_embeddings), dim=1)
            global_updates = self._global_update(global_inputs)
            if self._config.normalize_updates:
                global_updates = self._update_normalization(global_updates)
            updates = global_updates + updates

        if self._config.binarize_updates:
            update_logits = updates
            updates = binarize(update_logits)
            layer_records.append(
                QuantizationRecord(
                    kind="binary_update",
                    layer_index=layer_index,
                    logits=update_logits,
                    thresholds=(0.0,),
                    values=updates,
                )
            )

        if self._config.residual_updates:
            return node_embeddings + updates, layer_records
        if self._config.or_residual_updates:
            return torch.maximum(node_embeddings, updates), layer_records
        return updates, layer_records

    def _can_replay_layers(self, node_embeddings: torch.Tensor) -> bool:
        """Whether the layers can be replayed from a CUDA graph of the first one.

        Hooks and quantization records need each layer's tensors, which each replay overwrites. Recording costs about
        as much as running two layers. The message, aggregation and update functions must support it, see
        `MessageFunction.supports_replay`. With gradients, the backward pass is replayed as well, see
        `_record_for_training`.
        """
        return (
            node_embeddings.is_cuda
            and self._config.num_layers >= MIN_REPLAYED_LAYERS
            and not self._hooks
            and not self._config.ternarize_messages
            and not self._config.binarize_updates
            and self._message.supports_replay
            and self._relation_network._aggregation.supports_replay
            and self._relation_network._update.supports_replay
        )

    def _replay_layers(self, input: EncodedTensors, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Record the first layer as a CUDA graph and replay it for every layer.

        All layers have the same operations on the same shapes, so one recording serves them all; it is not kept,
        as the next input has other shapes.
        """
        embeddings = node_embeddings.clone()
        resources = self._replay_resources(input, embeddings)
        graph = torch.cuda.CUDAGraph()
        with _recording(graph, resources):
            # Recorded, not run: each replay computes one layer from `embeddings` back into `embeddings`.
            updated_embeddings, _ = self._layer(0, input, embeddings)
            embeddings.copy_(updated_embeddings)
            del updated_embeddings
        resources.wait_for_other_replays()
        for _ in range(self._config.num_layers):
            graph.replay()
        return embeddings

    def _record_for_training(self, input: EncodedTensors, node_embeddings: torch.Tensor) -> _TrainingRecording | None:
        """Record the layers and backpropagating through them, unless this is off or they cannot be recorded.

        Every forward pass records anew, so separate forward and backward passes accumulate gradients as usual.
        """
        if not REPLAY_TRAINING or self._unrecordable_for_training:
            return None
        resources = self._replay_resources(input, node_embeddings)
        try:
            return _TrainingRecording(self, input, node_embeddings, resources)
        except _Unrecordable as reason:
            warnings.warn(f"Training runs every message-passing layer, since {reason}.", stacklevel=2)
            self._unrecordable_for_training = True
            return None

    def _backpropagate_layer(self, input: EncodedTensors, layer_input: torch.Tensor,
                             leaves: dict[int, tuple[torch.Tensor, torch.Tensor]], output_gradient: torch.Tensor
                             ) -> tuple[torch.Tensor | None, list[torch.Tensor | None], _LayerBoundary]:
        """Run one layer, with the tensors it reads from outside of it replaced by leaves, and backpropagate.

        Args:
            input: The encoded graph input.
            layer_input: The node embeddings before the layer, a leaf that requires gradients.
            leaves: The leaves of the outside tensors by their ids, to which the layer's new ones are added.
            output_gradient: The gradient of the node embeddings after the layer.

        Returns:
            The gradient of the layer input, the gradient of each leaf, and the boundary of the layer.
        """
        boundary = _LayerBoundary(layer_input, leaves)
        with torch.enable_grad():
            with boundary:
                updated_embeddings, _ = self._layer(0, input, layer_input)
            # With allow_unused, the gradient of an input that the layer does not use is None.
            gradients: tuple[torch.Tensor | None, ...] = torch.autograd.grad(
                updated_embeddings, [layer_input] + [leaf for _, leaf in leaves.values()], output_gradient,
                allow_unused=True)
        return gradients[0], list(gradients[1:]), boundary

    def _replay_resources(self, input: EncodedTensors, node_embeddings: torch.Tensor) -> _ReplayResources:
        """The replay resources of the device of the node embeddings in this thread, ready for recording the layers.

        Libraries such as cuBLAS initialize on their first use in a thread and stream, which a recording cannot contain;
        the layer runs once without a recording first, and so does its backward pass if gradients are enabled.
        """
        if not hasattr(_replay_resources, 'by_device'):
            _replay_resources.by_device = {}
        by_device: dict[torch.device, _ReplayResources] = _replay_resources.by_device
        device = node_embeddings.device
        if device not in by_device:
            stream = torch.cuda.Stream(device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream), torch.no_grad():
                self._layer(0, input, node_embeddings)
            by_device[device] = _ReplayResources(stream, torch.cuda.graph_pool_handle())
        resources = by_device[device]
        if torch.is_grad_enabled() and not resources.warmed_up_backward:
            resources.stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(resources.stream):
                layer_input = node_embeddings.detach().clone().requires_grad_()
                self._backpropagate_layer(input, layer_input, {}, torch.ones_like(layer_input))
            torch.cuda.current_stream(device).wait_stream(resources.stream)
            resources.warmed_up_backward = True
        return resources


class RelationalGraphNeuralNetwork(nn.Module):
    """Relational Graph Neural Network (R-GNN) for planning problems.

    This is the main class that implements a Relational Graph Neural Network.

    Example:
        >>> domain = mm.Domain.from_file('path/to/domain.pddl')
        >>> hparam_config = HyperparameterConfig(domain=domain, embedding_size=64)
        >>> input_spec = (StateEncoder(), GoalEncoder())
        >>> output_spec = [('value', ObjectsScalarDecoder(hparam_config))]
        >>> module_config = ModuleConfig(
        ...     aggregation_function=MeanAggregation(),
        ...     message_function=PredicateMLPMessages(hparam_config, input_spec),
        ...     update_function=MLPUpdates(hparam_config)
        ... )
        >>> model = RelationalGraphNeuralNetwork(
        ...     hparam_config, module_config, input_spec, output_spec
        ... )
    """
    def __init__(self,
                 hparam_config: HyperparameterConfig,
                 module_config: ModuleConfig,
                 input_spec: tuple[Encoder, ...],
                 output_spec: list[tuple[str, Decoder]]):
        """Initialize the Relational Graph Neural Network.

        Args:
            hparam_config: The hyperparameter configuration of the R-GNN.
            module_config: The module configuration containing aggregation, message,
                           and update functions.
            input_spec: The encoders that define how to transform PDDL structures
                        into graph neural network inputs. For example,
                        (StateEncoder(), GoalEncoder(), GroundActionsEncoder()) indicates
                        that each instance must be a tuple containing a state, followed
                        by a goal, followed by actions.
            output_spec: The named outputs of the forward pass using decoder objects.
                         For example, [("q_values", ActionScalarDecoder(config)),
                         ("state_value", ObjectsScalarDecoder(config))] defines two outputs
                         with different readout functions.
        """
        super().__init__()
        self._hparam_config = hparam_config
        self._module_config = module_config
        self._input_spec = input_spec
        self._output_spec = output_spec
        self._mpnn_module = RelationalLayerStackModule(hparam_config, module_config)
        self._readouts = nn.ModuleDict()
        for output_name, decoder in output_spec:
            assert isinstance(output_name, str), 'The first part of the output specification must be a name.'
            assert isinstance(decoder, Decoder), 'The second part of the output specification must be a Decoder.'
            self._readouts.add_module(output_name, decoder)
        self._dummy = nn.Parameter(torch.empty(0))
        self._hooks: list[Callable[[ForwardState], None]] = []
        self._latest_quantization_records: tuple[QuantizationRecord, ...] = ()

    def _notify_hooks(self, forward_state: ForwardState) -> None:
        """Notify all registered hooks of the current forward state.

        Args:
            forward_state: The current forward state containing layer info and readouts.
        """
        for hook in self._hooks:
            hook(forward_state)

    def get_hparam_config(self) -> HyperparameterConfig:
        """Get the hyperparameter configuration.

        Returns:
            The hyperparameter configuration used by this model.
        """
        return self._hparam_config

    def get_module_config(self) -> ModuleConfig:
        """Get the module configuration.

        Returns:
            The module configuration used by this model.
        """
        return self._module_config

    def quantization_records(self) -> tuple[QuantizationRecord, ...]:
        """Return quantization records from the latest full forward pass."""
        return self._latest_quantization_records

    def add_hook(self, hook_func: Callable[[ForwardState], None]) -> None:
        """Add a hook function to be called during forward passes.

        Args:
            hook_func: Function to call with ForwardState at each layer.
        """
        self._hooks.append(hook_func)

    def clear_hooks(self) -> None:
        """Remove all registered hook functions."""
        self._hooks.clear()

    def get_device(self):
        """Get the device this model is on.

        Returns:
            The torch device where this model is located.
        """
        return self._dummy.device

    def _internal_forward(self, input: 'EncodedTensors') -> ForwardState:
        """Run the neural network computation phase of the forward pass.

        This private method contains the shared logic for running the MPNN forward pass
        and creating the final ForwardState with readout functions.

        Args:
            input: The encoded tensors from the input encoders.

        Returns:
            ForwardState object containing the final layer index and readout functions.
        """
        # Pass the input through the MPNN module
        if len(self._hooks) > 0:
            def hook_function(layer_index: 'int', node_embeddings: 'torch.Tensor') -> 'None':
                nonlocal self, input
                def make_readout_func(readout: Any) -> Callable[[], Any]:
                    return lambda: readout(node_embeddings, input)
                curried_readouts = { name: make_readout_func(readout) for name, readout in self._readouts.items() }
                forward_state = ForwardState(layer_index, curried_readouts, self._mpnn_module.layer_quantization_records())
                self._notify_hooks(forward_state)
            self._mpnn_module.add_hook(hook_function)
        node_embeddings = self._dummy.new_zeros(
            (input.node_count, self._hparam_config.embedding_size),
            requires_grad=False,
        )
        node_embeddings = self._mpnn_module(input, node_embeddings)
        self._latest_quantization_records = self._mpnn_module.quantization_records()
        if len(self._hooks) > 0:
            self._mpnn_module.clear_hooks()
        def make_readout_func(readout: Any) -> Callable[[], Any]:
            return lambda: readout(node_embeddings, input)
        curried_readouts = { name: make_readout_func(readout) for name, readout in self._readouts.items() }
        return ForwardState(self._hparam_config.num_layers - 1, curried_readouts, self._latest_quantization_records)

    def forward(self, x: list[tuple]) -> ForwardState:  # type: ignore
        """Perform a forward pass through the R-GNN.

        Args:
            x: List of input tuples, where each tuple contains the inputs specified
               by the input_spec (e.g., state, goal, actions).

        Returns:
            ForwardState object containing the final layer index and readout functions
            for extracting outputs.
        """
        assert isinstance(x, list), 'Expected input to be a list.'
        input = get_input_from_encoders(x, self._input_spec, self.get_device())
        return self._internal_forward(input)

    def curry_forward(self, x: list[tuple]) -> Callable[[], ForwardState]:
        """Return a lambda that performs a forward pass through the R-GNN.

        This method precomputes the encoded tensors but delays the MPNN computation
        until the returned lambda is called. This is equivalent to model.forward(x)
        but allows for separating the encoding phase from the neural network computation.

        Args:
            x: List of input tuples, where each tuple contains the inputs specified
               by the input_spec (e.g., state, goal, actions).

        Returns:
            A lambda function that when called returns a ForwardState object containing
            the final layer index and readout functions for extracting outputs.
        """
        assert isinstance(x, list), 'Expected input to be a list.'
        input = get_input_from_encoders(x, self._input_spec, self.get_device())

        # Define a curried function that captures the input and runs the internal forward pass.
        def curried_forward() -> ForwardState:
            return self._internal_forward(input)

        return curried_forward

    def clone(self) -> 'RelationalGraphNeuralNetwork':
        """Create a deep copy of the model with identical weights and configuration.

        Returns:
            A new RelationalGraphNeuralNetwork instance with the same weights
            and hyperparameters as this model.
        """
        model_clone = RelationalGraphNeuralNetwork(self._hparam_config,
                                                   self._module_config,
                                                   self._input_spec,
                                                   self._output_spec)
        model_clone.load_state_dict(self.state_dict())
        return model_clone.to(self.get_device())

    def copy_to(self, destination_model: 'RelationalGraphNeuralNetwork') -> None:
        """Copy this model's weights to another RelationalGraphNeuralNetwork instance.

        Args:
            destination_model: The model to copy the weights to.
        """
        destination_model.load_state_dict(self.state_dict())

    def save(self, path: Union[Path, str], extras: dict = {}) -> None:
        """Save the model's state and hyperparameters to a file.

        Args:
            path: The path to save the model checkpoint.
            extras: Additional information to store in the checkpoint, e.g.,
                  the optimizer state.
        """
        module_dict = { f.name: getattr(self._module_config, f.name) for f in fields(self._module_config) }
        hparam_dict = { f.name: getattr(self._hparam_config, f.name) for f in fields(self._hparam_config) }
        del hparam_dict['domain']  # The domain is cannot be serialized.
        checkpoint = {
            'model': self.state_dict(),
            'hparam_config': hparam_dict,
            'module_config': module_dict,
            'input_spec': self._input_spec,
            'output_spec': self._output_spec,
            'extras': extras
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load(domain: mm.Domain, path: Union[Path, str], device: torch.device) -> tuple['RelationalGraphNeuralNetwork', dict]:
        """Load a model from a checkpoint file.

        Args:
            domain: The PDDL domain for the planning problem.
            path: The path to the model checkpoint file.
            device: The device to load the model to (e.g., 'cpu' or 'cuda').

        Returns:
            A tuple containing the loaded RelationalGraphNeuralNetwork model and
            a dictionary with additional information (e.g., optimizer state).
        """
        # weights_only=False is needed due to unpickle errors.
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        module_dict = checkpoint['module_config']
        hparam_dict = checkpoint['hparam_config']
        input_spec = checkpoint['input_spec']
        output_spec = checkpoint['output_spec']
        model_dict = checkpoint['model']
        extras_dict = checkpoint['extras']
        module_config = ModuleConfig(**module_dict)
        hparam_config = HyperparameterConfig(domain=domain, **hparam_dict)
        model = RelationalGraphNeuralNetwork(hparam_config, module_config, input_spec, output_spec)
        model.load_state_dict(model_dict)
        return model.to(device), extras_dict
