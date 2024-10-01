"""definition of several numpy types"""

from numpy.typing import NDArray
import numpy as np

float_array = NDArray[np.float32 | np.float64]
complex_array = NDArray[np.complex128]
int_array = NDArray[np.int32 | np.int64]
