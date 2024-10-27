import numpy as np
from pathlib import Path
from scipy.constants import c
from itertools import dropwhile, takewhile

N_E = 36
N_M = 49
N_R = 2 * N_E


def file_to_npz(filename: str | Path) -> None:
    folder_in = Path("../Fresnel_data")
    file = folder_in / f"{filename}.txt"
    data = np.loadtxt(fname=file, skiprows=10)

    with open(file=file, mode="r") as of:
        for _ in range(4):
            of.readline()
        line = of.readline().replace("(", "").replace(")", "").replace(",", "")
        freqs = list(
            dropwhile(
                lambda word: ":" not in word,
                takewhile(lambda word: "GHz" not in word, line.split(" ")),
            )
        )
        N_F = int(freqs[1])
        f = np.array([int(f_) for f_ in freqs[2:]]) * 1e9

    shape = (N_F, N_M, N_E)
    U_inc = data[:, 3] - 1j * data[:, 4]
    U_tot = data[:, 5] - 1j * data[:, 6]
    A_compact = (U_tot - U_inc).reshape(shape, order="F")
    r_ID = data[:, 1].astype(int).reshape(shape, order="F") - 1
    A = np.zeros((N_F, N_R, N_E), dtype=np.complex128)
    for k in range(N_F):
        for e in range(N_E):
            A[k, r_ID[k, :, e], e] = A_compact[k, :, e]

    kappa = 2 * np.pi * f / c
    folder_out = Path("../Fresnel")
    np.savez(file=folder_out / f"{filename}.npz", A=A, kappa=kappa)
    return


if __name__ == "__main__":
    list_of_files = [
        "uTM_shaped",
        "dielTM_dec4f",
        "dielTM_dec8f",
        "rectTM_cent",
        "rectTM_dece",
        "twodielTM_8f",
    ]
    for file in list_of_files:
        file_to_npz(file)
