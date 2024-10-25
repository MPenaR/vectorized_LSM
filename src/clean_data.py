import numpy as np
from pathlib import Path
from scipy.constants import c

N_E = 36
N_M = 49
N_R = 2 * N_E

N_F = 8  # for now


def file_to_npz(filename: str | Path) -> None:
    folder_in = Path("../Fresnel_data")
    data = np.loadtxt(fname=folder_in / f"{filename}.txt", skiprows=10)
    shape = (N_F, N_M, N_E)
    U_inc = data[:, 3] - 1j * data[:, 4]
    U_tot = data[:, 5] - 1j * data[:, 6]
    A_compact = (U_tot - U_inc).reshape(shape, order="F")
    r_ID = data[:, 1].astype(int).reshape(shape, order="F") - 1
    A = np.zeros((N_F, N_R, N_E), dtype=np.complex128)
    for k in range(N_F):
        for e in range(N_E):
            A[k, r_ID[k, :, e], e] = A_compact[k, :, e]

    f = np.linspace(2, 16, 8) * 1e9
    kappa = 2 * np.pi * f / c
    folder_out = Path("../Fresnel")
    np.savez(file=folder_out / f"{filename}.npz", A=A, kappa=kappa)
    return


if __name__ == "__main__":
    file_to_npz("uTM_shaped")
