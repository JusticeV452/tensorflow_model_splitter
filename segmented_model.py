import copy
import numpy as np
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


class SegmentedModel:
    def __init__(self, nodes, connections=None):
        if connections is None:
            assert isinstance(nodes, keras.Model | SegmentedModel)
            nodes, connections = format_node_connections(nodes)
        self.nodes = nodes
        self.connections = connections
        self.last_intermediates = None

    @property
    def input_names(self):
        return [node_name for node_name in self.nodes if "input" in node_name]

    def __repr__(self):
        return repr(self.to_dict())

    def __str__(self):
        return str(self.to_dict())

    def __iter__(self):
        return iter((self.nodes, self.connections))

    def __eq__(self, other):
        if not isinstance(other, SegmentedModel):
            return False
        return self.nodes == other.nodes and self.connections == other.connections

    def to_dict(self):
        return {"nodes": self.nodes, "connections": self.connections}

    def make_input(self, input_gen=np.random.rand):
        inps = []
        for inp_name in self.input_names:
            segment = self.nodes[inp_name]
            keras_inp = (
                segment[0].input if isinstance(segment, list)
                else segment.input
            )
            inps.append(input_gen(1, *keras_inp.shape[1:]))
        return inps

    def __call__(self, *inps):
        if len(inps) == 1 and isinstance(inps[0], list | tuple | dict):
            inps = inps[0]
        inp_dict = inps
        if not isinstance(inp_dict, dict):
            inp_dict = {
                layer_name: arr
                for layer_name, arr in zip(self.input_names, inps)
            }
        intermediate_results = {}
        node_ids = get_segment_ids(self.nodes.keys(), self.connections)
        for node_name in node_ids:
            _, parent_result = get_parent_result(
                node_name, self.connections, intermediate_results,
                default_func=lambda node_name: inp_dict[node_name]
            )
            segment = self.nodes[node_name]
            if isinstance(segment, list):
                segment = model_wrap(segment)
            intermediate_results[node_name] = segment(parent_result)
        self.last_intermediates = intermediate_results
        return intermediate_results[node_name]

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

        for node_name, layers in nodes.items():
            core_segment_name, prev_len = get_prev_len(node_name)
            if prev_len > 0 and core_segment_name in core_segment_lengths:
                segment_name = get_node_name(node_name, prev_len - 1, 0)
                conn_name = (
                    (segment_name,), (get_node_name(node_name, prev_len - 1, 1),)
                )
                new_connections[conn_name] = None

            segment_sizes = splitter_dict[node_name](layers)
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
            num_segments[node_name] = len(segment_indices)
            core_segment_lengths[core_segment_name].append(num_segments[node_name])

        for (inputs, outputs), merge_func in connections.items():
            # Skip connections added to link split segments
            if len(inputs) == 1 and len(outputs) == 1:
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

    def find_block_by_tail(tail_name):
        for block in blocks:
            if tail_name == block[-1].name:
                return block
        return None

    def add_to_parent_blocks(layer, inputs):
        for inp in inputs:
            input_name = get_prev_layer(inp).name
            target_block = find_block_by_tail(input_name)
            assert target_block
            target_block.append(layer)

    all_model_layers = list(iter_layers(model))

    for i, layer in enumerate(all_model_layers):
        if addr(layer) in seen:
            continue
        if is_input_layer(layer):
            blocks.append([layer])
            continue
        inputs = get_input_list(layer)
        outputs = layer.output
        single_input = len(inputs) == 1
        children = []
        for other_layer in all_model_layers[i + 1:]:
            input_names = [l.name.split('/')[0] for l in get_input_list(other_layer)]
            if layer.name in input_names:
                children.append(other_layer)
        single_output = not isinstance(outputs, list) and len(children) < 2

        ## Extend existsing block
        if single_input and single_output:
            add_to_parent_blocks(layer, inputs)
            seen.add(addr(layer))
            continue

        ## Create node and new blocks for each output
        try:
            node_input_names = tuple(
                find_block_by_tail(get_prev_layer(inp).name)[0].name
                for inp in inputs
            )
        except TypeError as e:
            print("Inputs to layers accepting multiple inputs must be the output of a block.")
            raise e
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

        node_name = (node_input_names, tuple(node_output_names))
        connections[node_name] = None if getattr(layer, "weights", []) else layer
        if single_input:
            add_to_parent_blocks(layer, inputs)
        seen.add(addr(layer))
    return SegmentedModel(
        {block[0].name: block for block in blocks}, connections
    )


def get_segment_ids(node_names, connections=None):
    """
    

    Parameters
    ----------
    node_names : TYPE
        DESCRIPTION.
    connections : TYPE
        DESCRIPTION.

    Returns
    -------
    segment_ids : TYPE
        DESCRIPTION.

    """
    if connections is None:
        node_names, connections = format_node_connections(node_names)
    if isinstance(node_names, dict):
        node_names = node_names.keys()

    segment_ids = {}

    # Create connections list and get unsegmented node sizes
    node_sizes = {}
    connections_list = []
    for conn in connections:
        inputs, outputs = conn
        # Skip connections linking segments in the same block
        if len(inputs) == 1 and len(outputs) == 1:
            group_name, _ = (
                (inputs[0], 0) if isinstance(inputs[0], str) else inputs[0]
            )
            node_sizes.setdefault(group_name, 0)
            node_sizes[group_name] += 1
            continue
        inputs = tuple(
            inp[0] if isinstance(inp, list | tuple)
            else inp for inp in inputs
        )
        connections_list.append((inputs, outputs))

    all_layer_inputs = set(
        inp for inputs, _ in connections_list for inp in inputs
    )

    depth = 0
    group_id = 0
    def update_segment_ids(node_names, outputs=False):
        for row_id, node_name in enumerate(node_names):
            if outputs and node_name in all_layer_inputs:
                continue
            d = depth + int(outputs)
            segment_ids[node_name] = f"{d}_{group_id}_{row_id}-{len(node_names)}_0"
            for s in range(node_sizes.get(node_name, 0)):
                segment_ids[(node_name, s + 1)] = f"{d}_{group_id}_{row_id}-{len(node_names)}_{s + 1}"

    while connections_list:
        # Find connections that do not use ouptuts of remaining connection
        found_parent = False
        group = []
        for i, (inputs, outputs) in enumerate(connections_list):
            found_parent = False
            for j, (_, other_outputs) in enumerate(connections_list):
                if j == i:
                    continue
                # Check if connection's inputs contains a segment name that
                # is an output of any remaining connections
                if any(inp in set(other_outputs) for inp in inputs):
                    found_parent = True
                    break
            # If connection is a child of another connection check next connection
            if found_parent:
                continue
            group.append((i, inputs, outputs))

        for c, inputs, outputs in group:
            update_segment_ids(inputs)
            update_segment_ids(outputs, outputs=True)
            group_id += 1

        for i, (c, *_) in enumerate(group):
            connections_list.pop(c - i)
        depth += 1

    # Model has no branches
    if not segment_ids:
        for i, node_name in enumerate(node_names):
            segment_ids[node_name] = f"0_0_0_{i}"
    assert len(segment_ids) == len(node_names), (
        f"{len(segment_ids)} != {len(node_names)}"
    )
    return segment_ids


def check_segment_split(model, segments_dict, connections, inps=None):
    inp_list = get_input_list(model)
    if type(inps) is type(None):
        inps = [np.random.rand(1, *layer.shape[1:]) for layer in inp_list]
    expected = model(inps).numpy()
    inp_dict = {layer.name: arr for layer, arr in zip(inp_list, inps)}
    pred = SegmentedModel(segments_dict, connections)(inp_dict)
    assert np.array_equal(pred, expected)
