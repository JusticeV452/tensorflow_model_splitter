import copy
import keras
import numpy as np
import networkx as nx
import tensorflow.keras as keras

from nnom.scripts.nnom_utils import is_input_layer, get_input_list
from splitters import split_by_num_segments
from utils import iter_layers, get_parent_result, model_wrap


def addr(obj: object):
    return hex(id(obj))


def get_prev_layer(keras_tensor):
    return keras_tensor._keras_history.layer


def format_node_connections(nodes, connections=None):
    if isinstance(nodes, keras.Model):
        nodes, connections = segment_branching_model(nodes)
    elif isinstance(nodes, SegmentedModel):
        connections = copy.deepcopy(nodes.connections)
        nodes = copy.deepcopy(nodes.nodes)
    return nodes, connections


def sort_connections(connections):
    connections = list(connections)

    while connections:
        for i, (inputs, outputs) in enumerate(connections):
            # Check if all inputs are satisfied or have no dependencies
            if all(
                inp not in [out for _, output_list in connections for out in output_list]
                for inp in inputs
            ):
                yield connections.pop(i)
                break


def segmented_model_to_graph(sm):
    G = nx.DiGraph()
    for (inputs, outputs), concat in sm.connections.items():
        concat_name = "(None)" if concat is None else concat.__class__.__name__
        for inp in inputs:
            for out in outputs:
                G.add_edge(inp, out, concat=concat_name)
                # G.add_node(inp, label=inp)
                # G.add_node(out, label=out)
    return G


def segmented_models_isomorphic(sm1, sm2):
    return nx.algorithms.isomorphism.DiGraphMatcher(
        segmented_model_to_graph(sm1),
        segmented_model_to_graph(sm2),
        edge_match=lambda e1, e2: e1["concat"] == e2["concat"]
    ).is_isomorphic()


class SegmentedModel:
    def __init__(self, nodes, connections=None):
        if connections is None:
            assert isinstance(nodes, keras.Model | SegmentedModel | str)
            if isinstance(nodes, str):
                nodes = keras.saving.load_model(nodes)
            nodes, connections = format_node_connections(nodes)
        else:
            assert isinstance(nodes, dict) and isinstance(connections, dict)
        self.nodes = nodes
        self.connections = connections
        self.last_intermediates = None
        self.last_inputs = None
        all_layer_inputs = set(
            inp for inputs, _ in self.connections for inp in inputs
        )
        self._input_names = [
            node_name for node_name in self.nodes
            if "input" in node_name
        ]
        self._output_names = self._input_names if not self.connections else [
            out for _, outputs in self.connections
            for out in outputs
            if out not in all_layer_inputs
        ]

    @property
    def input_names(self):
        return self._input_names

    @property
    def output_names(self):
        return self._output_names

    def __repr__(self):
        return repr(self.to_dict())

    def __str__(self):
        return str(self.to_dict())

    def __iter__(self):
        return iter((self.nodes, self.connections))

    def run_order(self):
        yielded = set()
        all_layer_inputs = set(
            inp for inputs, _ in self.connections for inp in inputs
        )
        for inputs, outputs in sort_connections(self.connections):
            for inp in inputs:
                if inp in yielded:
                    continue
                yielded.add(inp)
                yield inp
            for out in outputs:
                if out in all_layer_inputs:
                    continue
                yield out
    def segments(self, ordered=True):
        node_names = (
            self.run_order()
            if self.connections and ordered
            else self.nodes
        )
        for node_name in node_names:
            layers = self.nodes[node_name]
            segment = model_wrap(layers) if isinstance(layers, list) else layers
            yield node_name, segment

    def __eq__(self, other):
        if not isinstance(other, SegmentedModel):
            return False
        return (
            self.nodes == other.nodes
            and self.connections == other.connections
        )

    def to_dict(self):
        return {"nodes": self.nodes, "connections": self.connections}
    
    def to_graph(self):
        return segmented_model_to_graph(self)

    def make_input(self, input_gen=np.random.rand, batch_size=1):
        inps = []
        for inp_name in self.input_names:
            segment = self.nodes[inp_name]
            keras_inp = (
                segment[0].input if isinstance(segment, list)
                else segment.input
            )
            inps.append(input_gen(batch_size, *keras_inp.shape[1:]))
        return inps

    def __call__(self, *inps, merge_cast=lambda x: x.numpy()):
        if len(inps) == 1 and isinstance(inps[0], list | tuple | dict):
            inps = inps[0]
        inp_dict = inps
        if not isinstance(inp_dict, dict):
            inp_dict = {
                layer_name: arr
                for layer_name, arr in zip(self.input_names, inps)
            }
        intermediate_results = {}
        self.last_inputs = {}
        for node_name, segment in self.segments():
            _, parent_result = get_parent_result(
                node_name, self.connections, intermediate_results,
                default_func=lambda node_name: inp_dict[node_name],
                merge_cast=merge_cast
            )
            self.last_inputs[node_name] = parent_result
            intermediate_results[node_name] = segment(parent_result)
        self.last_intermediates = intermediate_results
        outputs = [
            intermediate_results[node_name]
            for node_name in self.output_names
        ]
        if len(self.output_names) == 1:
            outputs = outputs[0]
        return outputs

    def func_eq(self, other):
        if not isinstance(other, keras.Model | SegmentedModel):
            return False
        if isinstance(other, SegmentedModel):
            rand_inp = self.make_input()
            rand_allclose = np.allclose(self(rand_inp), other(rand_inp))
            ones_inp = self.make_input(lambda *inps: np.ones((*inps,)))
            return rand_allclose and np.allclose(self(ones_inp), other(ones_inp))
        try:
            check_segment_split(other, self.nodes, self.connections)
        except:
            return False
        return True
    
    def struct_eq(self, other):
        if not isinstance(other, SegmentedModel):
            return False
        return segmented_models_isomorphic(self, other)

    def extend(self, splitter):
        nodes, connections = self
        if not isinstance(splitter, dict):
            splitter = {key: splitter for key in nodes}
        splitter_dict = {
            key: split_by_num_segments(s) if isinstance(s, int) else s
            for key, s in splitter.items()
        }

        def get_node_name(base_name, prev_len, idx):
            return (
                (base_name[0], base_name[1] + prev_len + idx)
                if isinstance(base_name, tuple) else
                ((base_name, idx) if idx > 0 else base_name)
            )

        core_segment_lengths = {}
        def get_prev_len(node_name):
            core_segment_name = node_name[0] if isinstance(node_name, tuple) else node_name
            core_segment_lengths.setdefault(core_segment_name, [])
            segment_lengths = core_segment_lengths[core_segment_name]
            return (
                core_segment_name,
                sum(segment_lengths) - len(segment_lengths)
            )

        new_nodes = {}
        new_connections = {}
        num_segments = {}

        # Skip connections added to link split segments
        conn_to_skip = {
            (inputs, outputs) for inputs, outputs in connections            
            if len(inputs) == 1 and len(outputs) == 1
        }

        for node_name, layers in nodes.items():
            core_segment_name, prev_len = get_prev_len(node_name)
            if prev_len > 0 and core_segment_name in core_segment_lengths:
                # Make a connection between node segments when multiple nodes
                # based on the same core node are present (.extend a second time)
                segment_name = get_node_name(node_name, prev_len - 1, 0)
                conn_name = (
                    (segment_name,), (get_node_name(node_name, prev_len - 1, 1),)
                )
                new_connections[conn_name] = None

            segment_sizes = splitter_dict.get(node_name, lambda l: [len(nodes[node_name])])(layers)
            segment_indices = [
                (prev_sum := sum(segment_sizes[:i]), prev_sum + segment_sizes[i])
                for i in range(len(segment_sizes))
            ]
            for i, (start, end) in enumerate(segment_indices):
                segment_name = get_node_name(node_name, prev_len, i)
                new_nodes[segment_name] = layers[start:end]
                if i < len(segment_indices) - 1:
                    conn_name = (
                        (segment_name,),
                        (get_node_name(node_name, prev_len, i + 1),)
                    )
                    new_connections[conn_name] = None
                elif len(segment_indices) == 1:
                    # This node's connections should not be modified,
                    # if it had a linking connection, add it back from conn_to_skip
                    new_connections.update({
                        conn: connections[conn] for conn in conn_to_skip
                        if node_name in conn[0]
                    })
            num_segments[node_name] = len(segment_indices)
            # Keep track of the number of segments created to determine names for other segments
            # the same core name (when extending a second time, .extend(...).extend(...))
            core_segment_lengths[core_segment_name].append(num_segments[node_name])

        for (inputs, outputs), merge_func in connections.items():
            # Skip connections added to link split segments
            if (inputs, outputs) in conn_to_skip:
                continue
            new_inputs = tuple(
                get_node_name(inp, 0, get_prev_len(inp)[1])
                for inp in inputs
            )
            new_connections[(new_inputs, outputs)] = merge_func
        return SegmentedModel(new_nodes, new_connections)


def segment_branching_model(model: keras.Model):
    blocks = []
    connections = {}
    seen = set()

    search_failed_on = None
    def find_block_by_tail(tail_name):
        nonlocal search_failed_on
        for block in blocks:
            if tail_name == block[-1].name:
                return block
        search_failed_on = tail_name
        return None

    def add_to_parent_block(layer, inp):
        input_name = get_prev_layer(inp).name
        target_block = find_block_by_tail(input_name)
        assert target_block
        target_block.append(layer)

    all_model_layers = list(iter_layers(model))
    for i, layer in enumerate(all_model_layers):
        if addr(layer) in seen:
            continue
        # print(layer.name)
        seen.add(addr(layer))
        if is_input_layer(layer):
            blocks.append([layer])
            continue
        inputs = get_input_list(layer)
        outputs = layer.output
        single_input = len(inputs) == 1
        children = []
        # Check if other layers use this layer as an input
        for other_layer in all_model_layers[i + 1:]:
            input_names = [l.name.split('/')[0] for l in get_input_list(other_layer)]
            if layer.name in input_names:
                children.append(other_layer)
        single_output = not isinstance(outputs, list) and len(children) < 2

        ## Extend existsing block
        if single_input and single_output:
            add_to_parent_block(layer, inputs[0])
            continue

        ## Create node and new blocks for each output
        try:
            node_input_names = tuple(
                find_block_by_tail(get_prev_layer(inp).name)[0].name
                for inp in inputs
            )
        except TypeError as e:
            print("Tried to find:", search_failed_on)
            print("blocks:")
            pp.pprint({block[0].name: [layer.name for layer in block] for block in blocks})
            print("connections:")
            pp.pprint(connections)
            print("Inputs to layers accepting multiple inputs must be the output of a block.")
            raise e
        if single_input:
            add_to_parent_block(layer, inputs[0])

        node_output_names = []
        # Search remaining layers for layers that use one of current layer's output as input
        for search_layer in all_model_layers[i + 1:]:
            search_layer_inp = search_layer.input
            if (
                isinstance(search_layer_inp, list)
                or get_prev_layer(search_layer_inp).name != layer.name
            ):
                continue
            block_start_layer = search_layer
            node_output_names.append(block_start_layer.name)
            blocks.append([block_start_layer])
            seen.add(addr(block_start_layer))

        if not node_output_names:
            continue
        node_name = (node_input_names, tuple(node_output_names))
        connections[node_name] = None if getattr(layer, "weights", []) else layer
    return SegmentedModel(
        {block[0].name: block for block in blocks}, connections
    )


def check_segment_split(model, segments_dict, connections, inps=None):
    inp_list = get_input_list(model)
    if type(inps) is type(None):
        inps = [np.random.rand(1, *layer.shape[1:]) for layer in inp_list]
    expected = model(inps)
    inp_dict = {layer.name: arr for layer, arr in zip(inp_list, inps)}
    pred = SegmentedModel(segments_dict, connections)(inp_dict)
    if type(expected) is list:
        assert type(pred) is list and len(pred) == len(expected)
        for p, ex in zip(pred, expected):
            assert np.array_equal(p, ex)
        return
    assert np.array_equal(pred, expected)
