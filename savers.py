import os
from nnom.scripts.nnom_utils import generate_model
def get_nnom_saver(save_root, nnom_subdir=True):
    def save(segment_id, model, x_test):
        assert not isinstance(x_test, type(None)), (
            "A reference dataset is required for NNoM quantization"
        )
        weights_path = None
        if save_root:
            _, save_name = os.path.split(save_root)
            nnom_dir = save_root
            if nnom_subdir:
                nnom_dir = os.path.join(save_root, "nnom")
            os.makedirs(nnom_dir, exist_ok=True)
            weights_path = os.path.join(nnom_dir, f"{save_name}_{segment_id}.h")
        return weights_path is not None, generate_model(model, x_test, name=weights_path)
    return save
