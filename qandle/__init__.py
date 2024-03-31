from .qcircuit import *
from .measurements import *
from .embeddings import *
from .ansaetze import *
from .drawer import *
from .splitter import *
from . import config
from .convolution import *
from .errors import *
from .operators import *
from .utils import *
from .qasm import *


def __reimport():  # pragma: no cover
    print("reimporting qandle")
    import importlib
    import sys
    import snakeviz

    modules = {k: v for k, v in sys.modules.items()}
    for module in modules:
        if module.startswith("qandle"):
            importlib.reload(sys.modules[module])

    # Patch snakeviz to not show in notebook (always open in new tab)
    snakeviz.ipymagic._check_ipynb = lambda: False


def __count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
