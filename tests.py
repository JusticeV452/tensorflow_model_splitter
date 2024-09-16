import unittest
import numpy as np
import tensorflow.keras as keras

from tensorflow.keras import layers as kl
from utils import iter_layers, model_wrap
from segmented_model import segment_branching_model

def check_split(model, segments, inp):
    expected = model(inp).numpy()
    segments_result = inp
    for segment in segments:
        segments_result = segment(segments_result)
    return np.array_equal(expected, segments_result)


def concatenate(*inputs):
    return keras.layers.Concatenate()(list(inputs))

def add(*inputs):
    return keras.layers.Add()(list(inputs))


def create_seq_output(
        inp, hidden_size, hidden_layers=1,
        activation="relu", standalone_activations=False
    ):
    out = inp
    for _ in range(hidden_layers):
        layer = kl.Dense(
            hidden_size,
            activation=(
                activation if not standalone_activations else "linear"
            )
        )
        out = layer(out)
        if standalone_activations:
            activ = activation
            if isinstance(activation, str):
                activ = keras.layers.Activation(activation)
            out = activ(out)
    return out


def create_seq_model(input_size, hidden_size, hidden_layers=1,
        activation="relu", standalone_activations=False
    ):
    inp = keras.Input(shape=(input_size,))
    return keras.Model(inputs=inp, outputs=create_seq_output(
        inp, hidden_size=hidden_size,
        hidden_layers=hidden_layers, activation=activation,
        standalone_activations=standalone_activations
    ))


def multiinput_oneoutput(
        input_sizes, hidden_size=30, out_size=1,
        hidden_layers=1, activation="relu",
        out_activation="linear",
        standalone_activations=False,
        merge_func=concatenate
    ):
    if not isinstance(activation, str):
        standalone_activations = True
    # the first branch operates on the first input
    concat_inputs = []
    inputs = [keras.Input(shape=(input_size,)) for input_size in input_sizes]
    for inp in inputs:
        output = create_seq_output(
            inp, hidden_size=hidden_size,
            hidden_layers=hidden_layers, activation=activation,
            standalone_activations=standalone_activations
        )
        concat_inputs.append(keras.Model(inputs=inp, outputs=output))
    # combine the output of the two branches
    combined = merge_func(*[model.output for model in concat_inputs])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    hidden_out = create_seq_output(
        combined, hidden_size=hidden_size,
        hidden_layers=hidden_layers, activation=activation,
        standalone_activations=standalone_activations
    )
    result = kl.Dense(out_size, activation=out_activation)(hidden_out)
    # our model will accept the inputs of the two branches and
    # then output a single value
    return keras.Model(
        inputs=[model.input for model in concat_inputs],
        outputs=result
    )


class TestModelWrap(unittest.TestCase):

    def assertNpEqual(self, result, expected):
        self.assertTrue(np.allclose(result, expected))

    def test_single_input_wrap(self):
        input_size = 20
        inp = keras.Input((input_size,))
        model = model_wrap(keras.Model(inputs=inp, outputs=inp).layers)
        test_inp = np.arange(input_size).reshape(1, input_size)
        self.assertNpEqual(model(test_inp), test_inp)

    def test_functional_equivalence(self):
        input_size = 20
        base_model = create_seq_model(
            input_size=input_size, hidden_size=20,
            hidden_layers=3, activation="relu",
            standalone_activations=False
        )
        model = model_wrap(base_model.layers)

        test_inp = np.arange(input_size).reshape(1, input_size)
        self.assertNpEqual(base_model(test_inp), model(test_inp))

    def test_activation_expand(self):
        input_size = 20
        # Model wrap should separate activations into their own layers
        model = model_wrap(create_seq_model(
            input_size=input_size, hidden_size=20,
            hidden_layers=3, activation="relu",
            standalone_activations=False
        ).layers)
        ex_model = create_seq_model(
            input_size=input_size, hidden_size=20,
            hidden_layers=3, activation="relu",
            standalone_activations=True
        )

        # Check that model structure is correct
        model_layers = list(iter_layers(model))
        ex_model_layers = list(iter_layers(ex_model))
        self.assertEqual(
            len(model_layers),
            len(ex_model_layers)
        )
        for layer, ex_layer in zip(model_layers, ex_model_layers):
            self.assertEqual(type(layer), type(ex_layer))


class TestSegmentedModel(unittest.TestCase):

    def assertConnectionsEqual(self, connections, ex_connections):
        self.assertEqual(len(connections), len(ex_connections))
        for connection, merge_func in connections.items():
            self.assertIn(connection, ex_connections)
            ex_value = ex_connections[connection]
            self.assertTrue(
                (merge_func is None and ex_value is None) or
                (ex_value and isinstance(merge_func, ex_value))
            )

    def assertNodesEqual(self, nodes, ex_nodes):
        self.assertEqual(len(nodes), len(ex_nodes))
        for node_name in nodes:
            self.assertIn(node_name, ex_nodes)

    def test_func_eq(self):
        inp_len = 3
        inp = keras.Input(shape=(inp_len,))
        eq_model = keras.Model(inputs=inp, outputs=create_seq_output(
            inp, hidden_size=3,
            hidden_layers=1, activation="relu",
            standalone_activations=False
        ))
        uneq_model = keras.Model(inputs=inp, outputs=create_seq_output(
            inp, hidden_size=3,
            hidden_layers=1, activation="relu",
            standalone_activations=False
        ))
        test_inputs = [
            np.ones((1, inp_len)),
            np.random.rand(1, inp_len),
            np.zeros((1, inp_len)),
        ]
        segmented_model = segment_branching_model(eq_model)
        # True
        self.assertTrue(segmented_model.func_eq(eq_model))
        self.assertFalse(segmented_model.func_eq(uneq_model))
        for test_inp in test_inputs:
            self.assertTrue(np.array_equal(segmented_model(test_inp), eq_model(test_inp)))
        uneq_compare_results = [
            np.array_equal(segmented_model(test_inp), uneq_model(test_inp))
            for test_inp in test_inputs
        ]
        self.assertFalse(all(uneq_compare_results))

    def test_non_branching_model(self):
        # single input - single output
        inp = keras.Input(shape=(20,))
        model = keras.Model(inputs=inp, outputs=create_seq_output(
            inp, hidden_size=20,
            hidden_layers=3, activation="relu",
            standalone_activations=False
        ))
        segmented_model = segment_branching_model(model)

        # Should only be a single node
        self.assertConnectionsEqual(segmented_model.connections, {})
        self.assertEqual(len(segmented_model.nodes), 1)
        # Check that node_name matches model input name
        self.assertNodesEqual(segmented_model.nodes, (model.layers[0].name,))
        # Model should be functionally equivalent
        self.assertTrue(segmented_model.func_eq(model))

    def test_branching_model(self):
        # two inputs >- output
        model = multiinput_oneoutput(
            [5, 5], hidden_size=10, out_size=1,
            hidden_layers=1, activation="relu",
            standalone_activations=False,
            merge_func=concatenate
        )
        # Expected node names based on number of model layers
        # and position of concatenate
        expected_in_nodes = (model.layers[0].name, model.layers[1].name)
        expected_out_nodes = (model.layers[5].name,)  # Layer right after concat
        segmented_model = segment_branching_model(model)
        # Check connections
        self.assertConnectionsEqual(segmented_model.connections, {
            (expected_in_nodes, expected_out_nodes): keras.layers.Concatenate
        })
        # Check node names
        self.assertNodesEqual(
            segmented_model.nodes,
            expected_in_nodes + expected_out_nodes
        )
        # Model should be functionally equivalent
        self.assertTrue(segmented_model.func_eq(model))

    def test_single_extend(self):
        model = multiinput_oneoutput(
            [20, 20], hidden_size=30, out_size=1,
            hidden_layers=1, activation="relu",
            standalone_activations=False,
            merge_func=add
        )
        segmented = segment_branching_model(model)
        single_extend = segmented.extend(1)
        self.assertConnectionsEqual(
            single_extend.connections,
            {key: type(val) for key, val in segmented.connections.items()
        })
        self.assertNodesEqual(single_extend.nodes, segmented.nodes)
        self.assertTrue(single_extend.func_eq(segmented))

    def test_extend(self):
        # =>- to ==>-- (split inputs and output into two connected segments)
        extend_num = 2
        model = multiinput_oneoutput(
            [20, 20], hidden_size=30, out_size=1,
            hidden_layers=4, activation="relu",
            standalone_activations=False,
            merge_func=add
        )
        inp1_name = model.layers[0].name
        inp2_name = model.layers[1].name
        out_name = model.layers[11].name
        ex_connections = {
            ((inp1_name,), ((inp1_name, 1),)): None,
            ((inp2_name,), ((inp2_name, 1),)): None,
            ((out_name,), ((out_name, 1),)): None,
            (((inp1_name, 1), (inp2_name, 1)), (out_name,)): keras.layers.Add
        }
        segmented_model = segment_branching_model(model)
        extended_model = segmented_model.extend(extend_num)
        self.assertConnectionsEqual(extended_model.connections, ex_connections)
        self.assertNodesEqual(extended_model.nodes, (
            inp1_name, (inp1_name, 1),
            inp2_name, (inp2_name, 1),
            out_name, (out_name, 1))
        )
        self.assertTrue(extended_model.func_eq(model))

    def test_multi_extend(self):
        model = multiinput_oneoutput(
            [20, 10], hidden_size=30, out_size=1,
            hidden_layers=12, activation="tanh",
            standalone_activations=False,
            merge_func=concatenate
        )
        inp1_name = model.layers[0].name
        inp2_name = model.layers[1].name
        out_name = model.layers[27].name
        segmented_model = segment_branching_model(model)
        single_extended = segmented_model.extend(2)
        multi_extended = single_extended.extend(2)
        orig_node_names = node_names = (inp1_name, inp2_name, out_name)
        for i in range(len(node_names)):
            node_names = node_names + tuple((node_names[i], j) for j in range(1, 4))
        ex_connections = {
            ((inp1_name,), ((inp1_name, 1),)): None,
            ((inp2_name,), ((inp2_name, 1),)): None,
            ((out_name,), ((out_name, 1),)): None,
            (((inp1_name, 3), (inp2_name, 3)), (out_name,)): keras.layers.Concatenate
        }
        # Add remaining segment connections
        for node_name in orig_node_names:
            for i in range(2):
                ex_connections[(((node_name, i + 1),), ((node_name, i + 2),))] = None
        self.assertConnectionsEqual(multi_extended.connections, ex_connections)
        self.assertNodesEqual(multi_extended.nodes, node_names)
        self.assertTrue(single_extended.func_eq(multi_extended))


if __name__ == '__main__':
    unittest.main()
