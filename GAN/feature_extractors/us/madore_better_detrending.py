import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from tqdm import tqdm

def extract_wave(phase, roi):
    l = 0.0154 # mm
    hilbert_d_phase = (phase[roi[0]:roi[1], 1:] - phase[roi[0]:roi[1], :-1])/2

    V = l/np.pi * hilbert_d_phase
    v = np.median(V, axis=0)
    z = np.zeros_like(v)
    for i in range(v.shape[0]):
        z[i] = np.sum(v[:i+1])

    return z


def main():
    path = "F:\\Formatted_datasets\\F1\\surrogates.pickle"

    with open(path, "rb") as file:
        us = pickle.load(file)["us"].T

    roi = (0, 1000)
    hilbert_us = hilbert(us, axis=0)
    hilbert_phase = np.angle(hilbert_us)
    hilbert_magnitude = np.abs(hilbert_us)

    z = extract_wave(hilbert_phase, roi)[14000:]

    similarities = np.ones((400, len(z))) * np.inf
    for T in tqdm(range(len(z))):
        for i in range(max(T-250, 0), T):
            for j in range(max(T-400, 0), i):
                similarities[i, j] = np.sum(np.abs(hilbert_magnitude[:, i] - hilbert_magnitude[:, j]))

    f = np.argsort(similarities, axis=0)

    C = np.zeros_like(z)
    for T in range(len(z)):
        m = np.zeros((3, 250))

        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                m[i, j] = (z[i] - z[f[i, j]]) / (i - f[i, j])

        m_0 = np.median(m)

        C[T] = T * m_0

    print(C)
            


    




if __name__ == "__main__":
    main()