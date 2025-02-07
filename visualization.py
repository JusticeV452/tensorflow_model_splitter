import io
import random
import pydot

from PIL import Image
from tensorflow.keras.utils import model_to_dot
from networkx.drawing.nx_pydot import to_pydot as nx_to_pydot
from segmented_model import SegmentedModel

GRAPH_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
    "gray", "cyan", "magenta", "gold", "navy", "teal",    
    "chocolate", "salmon", "violet", "indigo", "brown", "lime",
    "maroon", "turquoise", "lavender", "coral"
]

GRAPH_BORDERS = [
    "solid",
    "dashed",
    "dotted"
]


def get_dot_graph_node(graph, name):
    for node in graph.get_nodes():
        attributes = node.get_attributes()
        if "label" not in attributes:
            continue
        if name in attributes["label"]:
            return node
    raise Exception(f"Node with name '{name}' not found.")


def graph_to_img(graph, save_path=""):
    png_stream = io.BytesIO(graph.create(format='png'))
    image = Image.open(png_stream)
    if save_path:
        image.save(save_path)
    return image


def make_graph_img(model, save_path=None):
    if isinstance(model, SegmentedModel):
        dot_data = nx_to_pydot(model.to_graph())
    else:
        dot_data = model_to_dot(model, show_shapes=True, rankdir="TB")
    graph = pydot.graph_from_dot_data(dot_data.to_string())[0]
    return graph_to_img(graph, save_path=save_path)


def make_grouped_graph_img(model, save_path=None, node_styles={}):
    nodes, _ = SegmentedModel(model)
    dot_data = model_to_dot(model, show_shapes=True, rankdir="TB")
    graph = pydot.graph_from_dot_data(dot_data.to_string())[0]

    for node_name, layers in nodes.items():
        subgraph = pydot.Subgraph(f"cluster_{node_name}")
        subgraph.set("label", f"Group: {node_name}")
        if node_name not in node_styles:
            attr_dict = {
                "color": random.choice(GRAPH_COLORS),
                "style": random.choice(GRAPH_BORDERS)
            }
        else:
            attr_dict = node_styles.get(node_name)
        for attr, val in attr_dict.items():
            subgraph.set(attr, val)
        for layer in layers:
            subgraph.add_node(get_dot_graph_node(graph, layer.name))
        graph.add_subgraph(subgraph)
    return graph_to_img(graph, save_path=save_path)
