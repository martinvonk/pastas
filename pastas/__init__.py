# flake8: noqa
import logging
from warnings import warn

from pandas.plotting import register_matplotlib_converters

import pastas.objective_functions as objfunc
import pastas.plotting.plots as plots
import pastas.recharge as rch
import pastas.stats as stats
import pastas.timeseries_utils as ts
from pastas import extensions
from pastas.dataset import list_datasets, load_dataset
from pastas.decorators import set_use_numba
from pastas.model import Model
from pastas.noisemodels import ArmaModel, ARMANoiseModel, ARNoiseModel, NoiseModel
from pastas.plotting.modelcompare import CompareModels
from pastas.plotting.plots import TrackSolve
from pastas.rcparams import rcParams
from pastas.rfunc import (
    DoubleExponential,
    Exponential,
    FourParam,
    Gamma,
    Hantush,
    HantushWellModel,
    Kraijenhoff,
    One,
    Polder,
    Spline,
)
from pastas.solver import EmceeSolve, LeastSquares, LmfitSolve
from pastas.stressmodels import (
    ChangeModel,
    Constant,
    LinearTrend,
    RechargeModel,
    StepModel,
    StressModel,
    TarsoModel,
    WellModel,
)
from pastas.timeseries import validate_oseries, validate_stress
from pastas.transform import ThresholdTransform
from pastas.utils import initialize_logger, set_log_level
from pastas.version import __version__, show_versions

warn(
    """
    As of Pastas 1.5, no noisemodel is added to the pastas Model class by default anymore,
    and the noise argument in ml.solve will have no effect. To solve your model using a
    noisemodel, you have to explicitely add a noisemodel to your model before solving. For
    more information, and how to adapt your code, please see the this issue on GitHub:
    https://github.com/pastas/pastas/issues/735
    """,
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)
initialize_logger(logger)

# Register matplotlib converters when using Pastas
# https://github.com/pastas/pastas/issues/92

register_matplotlib_converters()
