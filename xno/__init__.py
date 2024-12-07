__version__ = '0.3.0'

# from .models import TFNO3d, TFNO2d, TFNO1d, TFNO
from .models import TZNO, TZNO1d, TZNO2d, TZNO3d

from .models import get_model
# from .data import datasets, transforms
from . import mpu
from .training import Trainer
from .losses import LpLoss, H1Loss, BurgersEqnLoss, ICLoss, WeightedSumLoss
