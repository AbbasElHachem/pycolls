import numpy as np
import pyximport

from .hbvpy import HBVCyPy

pyximport.install(setup_args={"include_dirs":np.get_include()})
from hbv_cy import (hbv_de, get_ns_py, get_ln_ns_py, loop_HBV_cpy,
                    get_kge_py, loop_HBV_py, route_flow_py)