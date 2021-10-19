import logging

import pastas.recharge as rch

from .model import Model
from .noisemodels import ArmaModel, NoiseModel
from .plots import TrackSolve
from .rcparams import rcParams
from .read import (read_dino, read_dino_level_gauge, read_knmi, read_meny,
                   read_waterbase)
from .rfunc import (DoubleExponential, Exponential, FourParam, Gamma, Hantush,
                    HantushWellModel, Kleur, One, Polder)
from .solver import LeastSquares, LmfitSolve, LmfitSolveNew
from .stressmodels import (Constant, LinearTrend, RechargeModel, StepModel,
                           StressModel, StressModel2, TarsoModel, WellModel,
                           ChangeModel)
from .timeseries import TimeSeries
from .transform import ThresholdTransform
from .utils import initialize_logger, set_log_level, show_versions
from .version import __version__

logger = logging.getLogger(__name__)
initialize_logger(logger)

# Register matplotlib converters when using Pastas
# https://github.com/pastas/pastas/issues/92
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
