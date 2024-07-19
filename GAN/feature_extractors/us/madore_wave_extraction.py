import pickle
import os
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm
import json


def detrend(z, magnitude):
    length = z.shape[0]
    window = 8*50
    m = np.zeros((3, length))
    for i in tqdm(range(length)):
        start_idx = max(0, i-window)
        results = np.zeros(i-start_idx)
        for j in range(start_idx, i):
            f = np.sum(np.abs(magnitude[:, i] - magnitude[:, j]))
            results[j-start_idx] = f

        for j, combi in enumerate(np.argsort(results)[:3]):
            m[j, i] = (z[i] - z[combi])/(i-combi)

    m_0 = np.median(m)
    C_lin = np.zeros(length)
    for i in tqdm(range(1, length)):
        C_lin[i] = C_lin[i-1] + m_0

    return C_lin


def extract_wave(phase, roi):
    l = 0.0154 # mm
    hilbert_d_phase = (phase[roi[0]:roi[1], 1:] - phase[roi[0]:roi[1], :-1])/2

    V = l/np.pi * hilbert_d_phase
    v = np.median(V, axis=0)
    z = np.zeros_like(v)
    for i in range(v.shape[0]):
        z[i] = np.sum(v[:i+1])

    return z


def prompt_user():
    while True:
        user_input = input("Do you want to save this array? (Y/N): ").strip()
        if user_input in ['Y', 'N']:
            return user_input == 'Y'
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")


def update_settings_file(path, roi):
    with open(os.path.join(path, "settings.json"), 'r') as file:
        data = json.load(file)

    data["US"]['ROI'] = {"0": roi[0], "1": roi[1]}

    with open(os.path.join(path, "settings.json"), 'w') as file:
        json.dump(data, file, indent=4)


def get_wave_from_us(us, roi):
    assert us.shape[0] < us.shape[1]
    hilbert_us = hilbert(us, axis=0)
    hilbert_phase = np.angle(hilbert_us)
    z = extract_wave(hilbert_phase, roi)
    return z
    

def main():
    subject = "A3"
    path = os.path.join("C:", os.sep, "data", "Formatted_datasets", subject)

    with open(os.path.join(path, "surrogates.pickle"), "rb") as file:
        surrogates = pickle.load(file)
        
    us = np.array(surrogates["us"]).T
    roi = (0, 1000)
    
    hilbert_us = hilbert(us, axis=0)
    # hilbert_magnitude = np.abs(hilbert_us)
    hilbert_phase = np.angle(hilbert_us)
    z = extract_wave(hilbert_phase, roi)

    plt.plot(z)
    plt.show()

    # trend = detrend(z, hilbert_magnitude)

    # z_final = z

    # plt.plot(z_final)
    # plt.show()

    if prompt_user():
        update_settings_file(path, roi)
        surrogates["us_wave"] = z
        with open(os.path.join(path, "surrogates_new.pickle"), "wb") as file:
            pickle.dump(surrogates, file)
    else:
        print("Array not saved")


if __name__ == '__main__':
    with open(os.path.join("C:", os.sep, "data", "Formatted_datasets", "A3", "surrogates_new.pickle"), "rb") as file:
        surrs = pickle.load(file)
        print(surrs.keys())

        plt.plot(surrs["us_wave"])
        plt.show()

    # main()