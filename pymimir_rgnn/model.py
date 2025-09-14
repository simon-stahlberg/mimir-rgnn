import pymimir as mm
import torch
import torch.nn as nn

from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Union

from .bases import AggregationFunction, Encoder, MessageFunction, UpdateFunction
from .configs import HyperparameterConfig, ModuleConfig
from .decoders import Decoder
from .encoders import EncodedTensors, get_input_from_encoders
from .modules import MLP, SumReadout
from .utils import gumbel_sigmoid


class ForwardState:
    def __init__(self, layer_index: int, readouts: dict[str, Callable[[], Any]]):
        self._layer_index = layer_index
        self._readouts = readouts

    def get_layer_index(self) -> int:
        return self._layer_index

    def readout(self, name: str) -> Any:
        return self._readouts[name]()

class RelationalMessagePassingModule(nn.Module):
    def __init__(self,
                 hparam_config: HyperparameterConfig,
                 module_config: ModuleConfig):
        super().__init__()  # type: ignore
        self._embedding_size = hparam_config.embedding_size
        self._aggregation = module_config.aggregation_function
        self._message = module_config.message_function
        self._update = module_config.update_function
        self._relation_mlps = nn.ModuleDict()

    def _compute_messages_and_indices(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]):
        output_messages_list: list[torch.Tensor] = []
        output_indices_list: list[torch.Tensor] = []
        for relation_name, argument_indices in relations.items():
            if argument_indices.numel() > 0:
                argument_embeddings = torch.index_select(node_embeddings, 0, argument_indices)
                argument_messages = self._message.forward(relation_name, argument_embeddings)
                output_messages = (argument_embeddings.view_as(argument_messages) + argument_messages).view(-1, self._embedding_size)
                output_messages_list.append(output_messages)
                output_indices_list.append(argument_indices)
        output_messages = torch.cat(output_messages_list, 0)
        output_indices = torch.cat(output_indices_list, 0)
        return output_messages, output_indices

    def forward(self, node_embeddings: torch.Tensor, relations: dict[str, torch.Tensor]) -> torch.Tensor:
        messages, indices = self._compute_messages_and_indices(node_embeddings, relations)
        aggregated_messages = self._aggregation.forward(node_embeddings, messages, indices)
        return self._update.forward(node_embeddings, aggregated_messages)


class RelationalLayersModule(nn.Module):
    def __init__(self,
                 hparam_config: HyperparameterConfig,
                 module_config: ModuleConfig):
        super().__init__()  # type: ignore
        self._config = hparam_config
        self._relation_network = RelationalMessagePassingModule(hparam_config, module_config)
        if hparam_config.global_readout:
            self._global_readout = SumReadout(hparam_config.embedding_size, hparam_config.embedding_size)
            self._global_update = MLP(2 * hparam_config.embedding_size, hparam_config.embedding_size)
        if hparam_config.normalize_updates:
            self._update_normalization = nn.LayerNorm(hparam_config.embedding_size)
        self._hooks: list[Callable[[int, torch.Tensor], None]] = []

    def _notify_hooks(self, iteration: int, embeddings: torch.Tensor) -> None:
        for hook in self._hooks:
            hook(iteration, embeddings)

    def add_hook(self, hook_func: Callable[[int, torch.Tensor], None]) -> None:
        self._hooks.append(hook_func)

    def clear_hooks(self) -> None:
        self._hooks.clear()

    def forward(self, input: EncodedTensors) -> torch.Tensor:
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
        return node_embeddings


class RelationalGraphNeuralNetwork(nn.Module):
    def __init__(self,
                 hparam_config: HyperparameterConfig,
                 module_config: ModuleConfig,
                 input_spec: tuple[Encoder, ...],
                 output_spec: list[tuple[str, Decoder]]):
        """
        Relational Graph Neural Network (RGNN) for planning states.

        :param hparam_config: The hyperparameter config of the R-GNN.
        :type hparam_config: HyperparameterConfig
        :param module_config: The module configuration containing aggregation, message, and update functions.
        :type module_config: ModuleConfig
        :param input_spec: The encoders that define how to transform PDDL structures into graph neural network inputs. For example, (StateEncoder(), GoalEncoder(), GroundActionsEncoder()) indicates that each instance must be a tuple containing a state, followed by a goal, followed by actions.
        :type input_spec: tuple[Encoder, ...]
        :param output_spec: The named outputs of the forward pass using decoder objects. For example, [("q_values", ActionScalarDecoder(config)), ("state_value", ObjectsScalarDecoder(config))] defines two outputs with different readout functions.
        :type output_spec: list[tuple[str, Decoder]]
        """
        super().__init__()
        self._hparam_config = hparam_config
        self._module_config = module_config
        self._input_spec = input_spec
        self._output_spec = output_spec
        self._mpnn_module = RelationalLayersModule(hparam_config, module_config)
        self._readouts = nn.ModuleDict()
        for output_name, decoder in output_spec:
            assert isinstance(output_name, str), 'The first part of the output specification must be a name.'
            assert isinstance(decoder, Decoder), 'The second part of the output specification must be a Decoder.'
            self._readouts.add_module(output_name, decoder)
        self._dummy = nn.Parameter(torch.empty(0))
        self._hooks: list[Callable[[ForwardState], None]] = []

    def _notify_hooks(self, forward_state: ForwardState) -> None:
        for hook in self._hooks:
            hook(forward_state)

    def get_config(self) -> HyperparameterConfig:
        return self._hparam_config

    def get_layer_count(self) -> int:
        return self._hparam_config.num_layers

    def set_layer_count(self, count: int) -> None:
        self._hparam_config.num_layers = count

    def add_hook(self, hook_func: Callable[[ForwardState], None]) -> None:
        self._hooks.append(hook_func)

    def clear_hooks(self) -> None:
        self._hooks.clear()

    def get_device(self):
        return self._dummy.device

    def forward(self, x: list[tuple]) -> ForwardState:  # type: ignore
        # Create input using encoder-based specification
        assert isinstance(x, list), 'Expected input to be a list.'
        input = get_input_from_encoders(x, self._input_spec, self.get_device())
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

    def clone(self) -> 'RelationalGraphNeuralNetwork':
        """
        Clones the model's weights and hyperparameters.

        :return: A new instance of RelationalGraphNeuralNetwork with the same weights and hyperparameters.
        :rtype: RelationalGraphNeuralNetwork
        """
        model_clone = RelationalGraphNeuralNetwork(self._hparam_config,
                                                   self._module_config,
                                                   self._input_spec,
                                                   self._output_spec)
        model_clone.load_state_dict(self.state_dict())
        return model_clone.to(self.get_device())

    def copy_to(self, destination_model: 'RelationalGraphNeuralNetwork') -> None:
        """
        Copies the model's weights to another instance of RelationalGraphNeuralNetwork.
        :param destination_model: The model to copy the weights to.
        :type destination_model: RelationalGraphNeuralNetwork
        """
        destination_model.load_state_dict(self.state_dict())

    def save(self, path: Union[Path, str], extras: dict = {}) -> None:
        """
        Saves the model's state and hyperparameters to a file.
        The parameter `extras` can be used to store additional information in the checkpoint, e.g., the optimizer state.

        :param path: The path to save the model checkpoint.
        :type path: 'Path | str'
        :param extras: Additional information to store in the checkpoint.
        :type extras: dict
        """
        config_dict = { f.name: getattr(self._hparam_config, f.name) for f in fields(self._hparam_config) }
        del config_dict['domain']  # The domain is cannot be serialized.
        checkpoint = {
            'model': self.state_dict(),
            'config': config_dict,
            'input_spec': self._input_spec,
            'output_spec': self._output_spec,
            'aggregation_function': self._module_config.aggregation_function,
            'message_function': self._module_config.message_function,
            'update_function': self._module_config.update_function,
            'extras': extras
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load(domain: mm.Domain, path: Union[Path, str], device: torch.device) -> tuple['RelationalGraphNeuralNetwork', dict]:
        """
        Loads a model from a checkpoint file.

        :param domain: The domain of the planning problem.
        :type domain: mimir.Domain
        :param path: The path to the model checkpoint file.
        :type path: 'Path | str'
        :param device: The device to load the model to (e.g., 'cpu' or 'cuda').
        :type device: 'torch.device'
        :return: A tuple containing the loaded model and a dictionary with additional information (e.g., optimizer state).
        :rtype: tuple[RelationalGraphNeuralNetwork, dict]
        """
        # weights_only=False is needed due to unpickle errors related to our enums.
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config_dict = checkpoint['config']
        input_spec = checkpoint['input_spec']
        output_spec = checkpoint['output_spec']
        aggregation_function = checkpoint['aggregation_function']
        message_function = checkpoint['message_function']
        update_function = checkpoint['update_function']
        model_dict = checkpoint['model']
        extras_dict = checkpoint['extras']
        config = HyperparameterConfig(domain=domain, **config_dict)
        module_config = ModuleConfig(
            aggregation_function=aggregation_function,
            message_function=message_function,
            update_function=update_function
        )
        model = RelationalGraphNeuralNetwork(config, module_config, input_spec, output_spec)
        model.load_state_dict(model_dict)
        return model.to(device), extras_dict
