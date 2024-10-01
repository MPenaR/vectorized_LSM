"""package with the different implementations"""

import numpy as np
from numpy_types import complex_array, float_array
from numpy.linalg import svd
from scipy.spatial.distance import cdist
from scipy.special import j0, y0


def h0_1(x: float_array) -> complex_array:
    """
    Computes the hankel function of first kind and zero order assuming real
    argument.
    """
    return j0(x) + 1j * y0(x)


def monofrequency_near_field_LSM(
    A: complex_array,
    xy_E: float_array,
    xy_R: float_array,
    xy: float_array,
    k: float,
    alpha=0.0,
    reverse_time_dependency=False,
) -> float_array:
    """
    Computes the linear sampling indicator for a single frequency assuming the
    data is in the near field.

    Inputs:
    - A: scattered field matrix A where the a_ij element corresponds to the
    measure at antenna i when the incident field is radiated at antena j.
    The number of emitting and receiving directions do not need to  be equal.
    - xy_E : N_E x 2 array containing the x and y coordinates of the position
    vector of the emitting antenas.
    - xy_R : N_R x 2 array containing the x and y coordinates of the position
    vector of the receiving antenas.
    - xy : N x 2 array containing the x and y coordinates of the sampling
    points. They do not need to be arranged into a cartesian grid.
    - k : wavenumber of the incident field.
    - alpha : parameter of the Tikhonov regularization:
        || Ax - b ||² + alfa ||x||²
    - reverse_time_dependency: If True, assume the time dependant signal is
    given by u(t) = Re(U exp(iwt)). By default is False, which
    corresponds to the more common choice of u(t) = Re( U exp(-iwt)).

    Output:
    - Ind : N dimensional array containing the value of the indicator at each
    sampling point

    """
    if reverse_time_dependency:
        A = np.conjugate(A)

    u, s, _ = svd(a=A, full_matrices=False)
    b = h0_1(k * cdist(xy, xy_R))
    g = np.sum((s / (s**2 + alpha)) ** 2 * np.dot(b, np.conj(u)) ** 2, axis=1)
    return g
