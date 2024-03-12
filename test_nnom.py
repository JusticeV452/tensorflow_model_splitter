import numpy as np
import sys

from nnom.scripts.nnom_utils import generate_model
from nnom.examples.auto_test.main import build_model
from model_splitter import tiny_model_func

np.random.seed(0)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tiny":
        shape = (10, 1)
        model = tiny_model_func(shape)
    else:
        shape = (224, 224, 3)
        model = build_model(shape)
    generate_model(model, np.random.rand(1000, *shape))
