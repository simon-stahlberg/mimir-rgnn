import pymimir as mm
import torch
import torch.nn as nn

from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Union

from .bases import Encoder
from .configs import HyperparameterConfig, ModuleConfig
from .decoders import Decoder
from .encoders import EncodedTensors, get_input_from_encoders
from .modules import MLP, SumReadout
from .utils import gumbel_sigmoid


class ForwardState:
    """Container for forward pass state and readout functions.

    This class encapsulates the state of a forward pass through the R-GNN,
    including the current layer index and cached readout functions for
    extracting outputs at any layer.
    """

    def __init__(self, layer_index: int, readouts: dict[str, Callable[[], Any]]):
        """Initialize the forward state.

        Args:
            layer_index: The current layer index in the forward pass.
            readouts: Dictionary of named readout functions for extracting outputs.
        """
        self._layer_index = layer_index
        self._readouts = readouts

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
        if hparam_config.global_readout:
            self._global_readout = SumReadout(hparam_config.embedding_size, hparam_config.embedding_size)
            self._global_update = MLP(2 * hparam_config.embedding_size, hparam_config.embedding_size)
        if hparam_config.normalize_updates:
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

    def add_hook(self, hook_func: Callable[[int, torch.Tensor], None]) -> None:
        """Add a hook function to be called at each layer.

        Args:
            hook_func: Function to call with (iteration, embeddings) at each layer.
        """
        self._hooks.append(hook_func)

    def clear_hooks(self) -> None:
        """Remove all registered hook functions."""
        self._hooks.clear()

    def forward(self, input: EncodedTensors) -> torch.Tensor:
        """Run multiple layers of message passing.

        Args:
            input: The encoded graph input containing relations and node information.

        Returns:
            Final node embeddings after all message passing layers.
        """
        self._message.setup(input.flattened_relations)
        device = input.node_sizes.device
        node_embeddings: torch.Tensor = torch.zeros([int(input.node_sizes.sum()), self._config.embedding_size], dtype=torch.float, requires_grad=True, device=device)
        for iteration in range(self._config.num_layers):
            next_node_embeddings: torch.Tensor = self._relation_network(node_embeddings, input.flattened_relations)
            if self._config.normalize_updates:
                next_node_embeddings = self._update_normalization(next_node_embeddings)
            if self._config.global_readout:
                global_embedding: torch.Tensor = self._global_readout(node_embeddings, input.node_sizes)
                global_messages: torch.Tensor = self._global_update(torch.cat((node_embeddings, global_embedding.repeat_interleave(input.node_sizes, dim=0)), 1))
                if self._config.normalize_updates:
                    global_messages = self._update_normalization(global_messages)
                next_node_embeddings = global_messages + next_node_embeddings
            if self._config.binarize_updates:
                next_node_embeddings = gumbel_sigmoid(next_node_embeddings, hard=True)
            if self._config.residual_updates:
                next_node_embeddings = node_embeddings + next_node_embeddings
            node_embeddings = next_node_embeddings
            self._notify_hooks(iteration, node_embeddings)
        self._message.cleanup()
        return node_embeddings


class RelationalGraphNeuralNetwork(nn.Module):
    """Relational Graph Neural Network (R-GNN) for planning problems.

    This is the main class that implements a Relational Graph Neural Network.

    Example:
        >>> domain = mm.Domain('path/to/domain.pddl')
        >>> hparam_config = HyperparameterConfig(domain=domain, embedding_size=64)
        >>> module_config = ModuleConfig(
        ...     aggregation_function=MeanAggregation(),
        ...     message_function=PredicateMLPMessages(hparam_config, input_spec),
        ...     update_function=MLPUpdates(hparam_config)
        ... )
        >>> input_spec = (StateEncoder(), GoalEncoder())
        >>> output_spec = [('q_values', ActionScalarDecoder(hparam_config))]
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
                forward_state = ForwardState(layer_index, curried_readouts)
                self._notify_hooks(forward_state)
            self._mpnn_module.add_hook(hook_function)
        node_embeddings = self._mpnn_module.forward(input)
        if len(self._hooks) > 0:
            self._mpnn_module.clear_hooks()
        def make_readout_func(readout: Any) -> Callable[[], Any]:
            return lambda: readout(node_embeddings, input)
        curried_readouts = { name: make_readout_func(readout) for name, readout in self._readouts.items() }
        return ForwardState(self._hparam_config.num_layers - 1, curried_readouts)

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
