import pymimir as mm
import torch
import torch.nn as nn

from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Union

from .encodings import InputType, OutputValueType, OutputNodeType, EncodedInput, encode_input, get_encoding
from .modules import MLP, SumReadout


class AggregationFunction(Enum):
    Add = 'add'
    Mean = 'mean'
    HardMaximum = 'hmax'
    SmoothMaximum = 'smax'


class UpdateFunction(Enum):
    MLP = 'mlp'


class MessageFunction(Enum):
    PredicateMLP = 'predicate_mlp'


@dataclass
class RelationalGraphNeuralNetworkConfig:
    domain: 'mm.Domain' = field(
        metadata={'doc': 'The domain of the planning problem.'}
    )

    input_specification: 'tuple[InputType, ...]' = field(
        metadata={'doc': 'The typed shape of the input. For example, (State, Goal) indicates that each instance must be a tuple containing a state followed by a goal.'}
    )

    output_specification: 'list[tuple[str, OutputNodeType, OutputValueType]]' = field(
        metadata={'doc': 'The named outputs of the forward pass. For example, [("actor", Scalar, Objects), ("critic", Scalar, Objects)] defines two outputs with different readout functions: one for the actor and one for the critic in RL; however, they share weights to compute the embeddings.'}
    )

    embedding_size: 'int' = field(
        default=32,
        metadata={'doc': 'The size of the node embeddings.'}
    )

    num_layers: 'int' = field(
        default=30,
        metadata={'doc': 'The number of message passing layers.'}
    )

    message_aggregation: 'AggregationFunction' = field(
        default=AggregationFunction.HardMaximum,
        metadata={'doc': 'The aggregation method for message passing.'},
    )

    message_function: 'MessageFunction' = field(
        default=MessageFunction.PredicateMLP,
        metadata={'doc': 'The type of the message function.'}
    )

    update_function: 'UpdateFunction' = field(
        default=UpdateFunction.MLP,
        metadata={'doc': 'The type of the update function.'}
    )

    normalize_updates: 'bool' = field(
        default=True,
        metadata={'doc': 'Whether to apply layer normalization to the embedding updates.'}
    )

    global_readout: 'bool' = field(
        default=False,
        metadata={'doc': 'Whether to use a global readout for the node embeddings.'}
    )

    random_initialization: 'bool' = field(
        default=False,
        metadata={'doc': 'Whether to use random initialization for the node embeddings.'}
    )


class ForwardState:
    def __init__(self, layer_index: 'int', readouts: 'dict[str, Callable[[], Any]]'):
        self._layer_index = layer_index
        self._readouts = readouts

    def get_layer_index(self) -> 'int':
        return self._layer_index

    def readout(self, name: 'str') -> Any:
        return self._readouts[name]()

class RelationMessagePassingBase(nn.Module):
    def __init__(self, config: 'RelationalGraphNeuralNetworkConfig'):
        super().__init__()
        self._embedding_size = config.embedding_size
        self._relation_mlps = nn.ModuleDict()
        for relation_name, relation_arity in get_encoding(config.domain, config.input_specification):
            input_size = relation_arity * config.embedding_size
            output_size = relation_arity * config.embedding_size
            if (input_size > 0) and (output_size > 0):
                assert config.message_function == MessageFunction.PredicateMLP, 'Other types of message functions are not implemented yet'
                self._relation_mlps[relation_name] = MLP(input_size, output_size)
        assert config.update_function == UpdateFunction.MLP, 'Other types of update functions are not implemented yet.'
        self._update = MLP(2 * config.embedding_size, config.embedding_size)

    def _compute_messages_and_indices(self, node_embeddings: 'torch.Tensor', relations: 'dict[str, torch.Tensor]'):
        output_messages_list = []
        output_indices_list = []
        for relation_name, relation_module in self._relation_mlps.items():
            if relation_name in relations:
                relation_values = relations[relation_name]
                input_embeddings = torch.index_select(node_embeddings, 0, relation_values).view(-1, relation_module.input_size)
                output_messages = (input_embeddings + relation_module(input_embeddings)).view(-1, self._embedding_size)
                output_messages_list.append(output_messages)
                output_indices_list.append(relation_values)
        output_messages = torch.cat(output_messages_list, 0)
        output_indices = torch.cat(output_indices_list, 0)
        return output_messages, output_indices


class MeanRelationMessagePassing(RelationMessagePassingBase):
    def __init__(self, config: 'RelationalGraphNeuralNetworkConfig'):
        super().__init__(config)

    def forward(self, node_embeddings: 'torch.Tensor', relations: 'dict[str, torch.Tensor]') -> 'torch.Tensor':
        output_messages, output_indices = self._compute_messages_and_indices(node_embeddings, relations)
        sum_msg = torch.zeros_like(node_embeddings)
        cnt_msg = torch.zeros_like(node_embeddings)
        sum_msg.index_add_(0, output_indices, output_messages)
        cnt_msg.index_add_(0, output_indices, torch.ones_like(output_messages))
        avg_msg = sum_msg / cnt_msg
        return self._update(torch.cat((avg_msg, node_embeddings), 1))


class SumRelationMessagePassing(RelationMessagePassingBase):
    def __init__(self, config: 'RelationalGraphNeuralNetworkConfig'):
        super().__init__(config)

    def forward(self, node_embeddings: 'torch.Tensor', relations: 'dict[str, torch.Tensor]') -> 'torch.Tensor':
        output_messages, output_indices = self._compute_messages_and_indices(node_embeddings, relations)
        sum_msg = torch.zeros_like(node_embeddings)
        sum_msg.index_add_(0, output_indices, output_messages)
        return self._update(torch.cat((sum_msg, node_embeddings), 1))


class HardMaximumRelationMessagePassing(RelationMessagePassingBase):
    def __init__(self, config: 'RelationalGraphNeuralNetworkConfig'):
        super().__init__(config)

    def forward(self, node_embeddings: 'torch.Tensor', relations: 'dict[str, torch.Tensor]') -> 'torch.Tensor':
        output_messages, output_indices = self._compute_messages_and_indices(node_embeddings, relations)
        max_msg = torch.full_like(node_embeddings, float('-inf')) # include_self=False leads to an error for some reason. Use -inf to get the same result.
        max_msg.index_reduce_(0, output_indices, output_messages, reduce='amax', include_self=True)
        return self._update(torch.cat((max_msg, node_embeddings), 1))


class SmoothMaximumRelationMessagePassing(RelationMessagePassingBase):
    def __init__(self, config: 'RelationalGraphNeuralNetworkConfig'):
        super().__init__(config)

    def forward(self, node_embeddings: 'torch.Tensor', relations: 'dict[str, torch.Tensor]') -> 'torch.Tensor':
        output_messages, output_indices = self._compute_messages_and_indices(node_embeddings, relations)
        exps_max = torch.zeros_like(node_embeddings)
        exps_max.index_reduce_(0, output_indices, output_messages, reduce="amax", include_self=False)
        exps_max = exps_max.detach()
        MAXIMUM_SMOOTHNESS = 12.0  # As the value approaches infinity, the hard maximum is attained
        max_offsets = exps_max.index_select(0, output_indices).detach()
        exps = (MAXIMUM_SMOOTHNESS * (output_messages - max_offsets)).exp()
        exps_sum = torch.full_like(node_embeddings, 1E-16)
        exps_sum.index_add_(0, output_indices, exps)
        max_msg = ((1.0 / MAXIMUM_SMOOTHNESS) * exps_sum.log()) + exps_max
        return self._update(torch.cat((max_msg, node_embeddings), 1))


class RelationalMessagePassingModule(nn.Module):
    def __init__(self, config: 'RelationalGraphNeuralNetworkConfig'):
        super().__init__()
        self._config = config
        if config.message_aggregation == AggregationFunction.Add:
            self._relation_network = SumRelationMessagePassing(config)
        elif config.message_aggregation == AggregationFunction.Mean:
            self._relation_network = MeanRelationMessagePassing(config)
        elif config.message_aggregation == AggregationFunction.SmoothMaximum:
            self._relation_network = SmoothMaximumRelationMessagePassing(config)
        elif config.message_aggregation == AggregationFunction.HardMaximum:
            self._relation_network = HardMaximumRelationMessagePassing(config)
        else:
            raise ValueError(f'aggregation is not one of ["{AggregationFunction.Add}", "{AggregationFunction.Mean}", "{AggregationFunction.SmoothMaximum}", "{AggregationFunction.HardMaximum}"]')
        if config.global_readout:
            assert config.update_function == UpdateFunction.MLP, 'Other types of update functions are not implemented yet.'
            self._global_readout = SumReadout(config.embedding_size, config.embedding_size)
            self._global_update = MLP(2 * config.embedding_size, config.embedding_size)
        if config.normalize_updates:
            self._update_normalization = nn.LayerNorm(config.embedding_size)
        self._hooks = []

    def _notify_hooks(self, iteration: 'int', embeddings: 'torch.Tensor') -> None:
        for hook in self._hooks:
            hook(iteration, embeddings)

    def add_hook(self, hook_func) -> 'None':
        self._hooks.append(hook_func)

    def clear_hooks(self) -> 'None':
        self._hooks.clear()

    def forward(self, input: 'EncodedInput') -> 'torch.Tensor':
        device = input.node_sizes.device
        node_embeddings: 'torch.Tensor' = torch.zeros([input.node_sizes.sum(), self._config.embedding_size], dtype=torch.float, requires_grad=True, device=device)
        if self._config.random_initialization:
            rng_state = torch.get_rng_state()
            torch.manual_seed(1234)  # TODO: The seed should probably be the hash of the instance.
            random_embeddings = torch.randn([input.object_indices.size(0), self._config.embedding_size], dtype=torch.float, requires_grad=True, device=device)
            node_embeddings = node_embeddings.index_add(0, input.object_indices, random_embeddings)
            torch.set_rng_state(rng_state)
        for iteration in range(self._config.num_layers):
            relation_messages = self._relation_network(node_embeddings, input.flattened_relations)
            if self._config.normalize_updates:
                relation_messages = self._update_normalization(relation_messages)  # Normalize the magnitude of the message's values to be between -1 and 1.
            if self._config.global_readout:
                global_embedding = self._global_readout(node_embeddings, input.node_sizes)
                global_messages = self._global_update(torch.cat((node_embeddings, global_embedding.repeat_interleave(input.node_sizes, dim=0)), 1))
                if self._config.normalize_updates:
                    global_messages = self._update_normalization(global_messages)
                node_embeddings = node_embeddings + global_messages + relation_messages
            else:
                node_embeddings = node_embeddings + relation_messages
            self._notify_hooks(iteration, node_embeddings)
        return node_embeddings


class ObjectScalarReadout(nn.Module):
    def __init__(self, config: 'RelationalGraphNeuralNetworkConfig'):
        super().__init__()
        self._object_readout = SumReadout(config.embedding_size, 1)

    def forward(self, node_embeddings: 'torch.Tensor', input: 'EncodedInput'):
        object_embeddings = node_embeddings.index_select(0, input.object_indices)
        return self._object_readout(object_embeddings, input.object_sizes)


class ObjectEmbeddingReadout(nn.Module):
    def forward(self, node_embeddings: 'torch.Tensor', input: 'EncodedInput'):
        return node_embeddings.index_select(0, input.object_indices)


class ActionScalarReadout(nn.Module):
    def __init__(self, config: 'RelationalGraphNeuralNetworkConfig'):
        super().__init__()
        self._object_readout = SumReadout(config.embedding_size, config.embedding_size)
        self._action_value = MLP(2 * config.embedding_size, 1)

    def forward(self, node_embeddings: 'torch.Tensor', input: 'EncodedInput'):
        action_embeddings = node_embeddings.index_select(0, input.action_indices)
        object_embeddings = node_embeddings.index_select(0, input.object_indices)
        object_aggregation: 'torch.Tensor' = self._object_readout(object_embeddings, input.object_sizes)
        object_aggregation = object_aggregation.repeat_interleave(input.action_sizes, dim=0)
        values: 'torch.Tensor' = self._action_value(torch.cat((action_embeddings, object_aggregation), dim=1))
        return [action_values.view(-1) for action_values in values.split(input.action_sizes.tolist())]


class ActionEmbeddingReadout(nn.Module):
    def forward(self, node_embeddings: 'torch.Tensor', input: 'EncodedInput'):
        return node_embeddings.index_select(0, input.action_indices)


class RelationalGraphNeuralNetwork(nn.Module):
    def __init__(self, config: 'RelationalGraphNeuralNetworkConfig'):
        """
        Relational Graph Neural Network (RGNN) for planning states.

        :param config: The config of the R-GNN.
        :type config: RelationalGraphNeuralNetworkConfig
        """
        super().__init__()
        self._config = config
        self._mpnn_module = RelationalMessagePassingModule(config)
        self._readouts = nn.ModuleDict()
        assert all([len(output) == 3 for output in config.output_specification]), 'The output specification must consist of a name, an output type, and an output node type.'
        for output_name, output_node_type, output_value_type in config.output_specification:
            assert isinstance(output_name, str), 'The first part of the output specification must be a name.'
            assert isinstance(output_node_type, OutputNodeType), 'The third part of the output specification must be an output node type.'
            assert isinstance(output_value_type, OutputValueType), 'The second part of the output specification must be an output type.'
            readout = None
            if (output_node_type == OutputNodeType.Objects) and (output_value_type == OutputValueType.Scalar):
                readout = ObjectScalarReadout(config)
            if (output_node_type == OutputNodeType.Objects) and (output_value_type == OutputValueType.Embeddings):
                readout = ObjectEmbeddingReadout()
            if (output_node_type == OutputNodeType.Action) and (output_value_type == OutputValueType.Scalar):
                readout = ActionScalarReadout(config)
            if (output_node_type == OutputNodeType.Action) and (output_value_type == OutputValueType.Embeddings):
                readout = ActionEmbeddingReadout()
            if readout is None:
                raise NotImplementedError(f'Output "{output_value_type}" over "{output_node_type}" is not implemented yet.')
            self._readouts.add_module(output_name, readout)
        self._dummy = nn.Parameter(torch.empty(0))
        self._hooks = []

    def _notify_hooks(self, forward_state: 'ForwardState') -> None:
        for hook in self._hooks:
            hook(forward_state)

    def add_hook(self, hook_func: 'Callable[[ForwardState], None]') -> 'None':
        self._hooks.append(hook_func)

    def clear_hooks(self) -> 'None':
        self._hooks.clear()

    def get_device(self):
        return self._dummy.device

    def forward(self, x: 'list[tuple]') -> 'ForwardState':
        # Create input
        input = encode_input(x, self._config.input_specification, self.get_device())
        # Pass the input through the MPNN module
        if len(self._hooks) > 0:
            def hook_function(layer_index: 'int', node_embeddings: 'torch.Tensor') -> 'None':
                nonlocal self, input
                curried_readouts = { name: lambda: readout(node_embeddings, input) for name, readout in self._readouts.items() }
                forward_state = ForwardState(layer_index, curried_readouts)
                self._notify_hooks(forward_state)
            self._mpnn_module.add_hook(hook_function)
        node_embeddings = self._mpnn_module.forward(input)
        if len(self._hooks) > 0:
            self._mpnn_module.clear_hooks()
        curried_readouts = { name: lambda: readout(node_embeddings, input) for name, readout in self._readouts.items() }
        return ForwardState(self._config.num_layers - 1, curried_readouts)

    def clone(self) -> 'RelationalGraphNeuralNetwork':
        """
        Clones the model's weights and hyperparameters.

        :return: A new instance of RelationalGraphNeuralNetwork with the same weights and hyperparameters.
        :rtype: RelationalGraphNeuralNetwork
        """
        model_clone = RelationalGraphNeuralNetwork(self._config)
        model_clone.load_state_dict(self.state_dict())
        return model_clone.to(self.get_device())

    def copy_to(self, destination_model: 'RelationalGraphNeuralNetwork') -> 'None':
        """
        Copies the model's weights to another instance of RelationalGraphNeuralNetwork.
        :param destination_model: The model to copy the weights to.
        :type destination_model: RelationalGraphNeuralNetwork
        """
        destination_model.load_state_dict(self.state_dict())

    def save(self, path: 'Union[Path, str]', extras: 'dict' = {}) -> 'None':
        """
        Saves the model's state and hyperparameters to a file.
        The parameter `extras` can be used to store additional information in the checkpoint, e.g., the optimizer state.

        :param path: The path to save the model checkpoint.
        :type path: 'Path | str'
        :param extras: Additional information to store in the checkpoint.
        :type extras: dict
        """
        config_dict = { f.name: getattr(self._config, f.name) for f in fields(self._config) }
        del config_dict['domain']  # The domain is cannot be serialized.
        checkpoint = { 'model': self.state_dict(), 'config': config_dict, 'extras': extras }
        torch.save(checkpoint, path)

    @staticmethod
    def load(domain: 'mm.Domain', path: 'Union[Path, str]', device: 'torch.device') -> 'tuple[RelationalGraphNeuralNetwork, dict]':
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
        model_dict = checkpoint['model']
        extras_dict = checkpoint['extras']
        config = RelationalGraphNeuralNetworkConfig(domain=domain, **config_dict)
        model = RelationalGraphNeuralNetwork(config)
        model.load_state_dict(model_dict)
        return model.to(device), extras_dict
