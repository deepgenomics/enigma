import importlib

import torch.nn as nn
from jaxtyping import Float, Integer
from torch import Tensor

from enigma.data.sequences import idx_to_onehot


class OneHotEmbedding(nn.Module):
    """One-hot embedding layer for converting integer indices to one-hot encoded
    tensors

    Unknown indices are converted to zero vectors.
    e.g. if num_classes=4, then
    - 0 -> [1, 0, 0, 0]
    - 1 -> [0, 1, 0, 0]
    - 2 -> [0, 0, 1, 0]
    - 3 -> [0, 0, 0, 1]
    - 4 -> [0, 0, 0, 0]

    In this project, the encoding used is "ACGT" -> [0, 1, 2, 3].
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()

        self.num_classes = num_classes

    def forward(self, x: Integer[Tensor, "b l"]) -> Float[Tensor, "b l c"]:
        one_hot = idx_to_onehot(x.long()).float()

        return one_hot


def import_class_from_path(class_path: str):
    """Import a class from a string path like 'module.submodule.ClassName'.

    Args:
        class_path: Full path to the class, e.g., 'flash_rna.models.flashrna.FlashRNA'

    Returns:
        The imported class
    """
    # Split the class path into module and class name
    module_path, class_name = class_path.rsplit(".", 1)

    # Import the module
    module = importlib.import_module(module_path)

    # Get the class from the module
    cls = getattr(module, class_name)

    return cls
