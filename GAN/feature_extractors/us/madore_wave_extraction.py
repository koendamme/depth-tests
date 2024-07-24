import pickle
import os
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, detrend, butter, lfilter
from tqdm import tqdm
import json


def detrend_madore(z, magnitude):
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
    for i in range(1, length):
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
    for pp in ["A", "B", "C"]:
        for s in [1, 2, 3]:
            subject = pp+str(s)
            if subject == "A1":
                continue
            print(f"Detrending subject {subject}...")
            path = os.path.join("C:", os.sep, "data", "Formatted_datasets", subject)

            with open(os.path.join(path, "surrogates.pickle"), "rb") as file:
                surrogates = pickle.load(file)

            with open(os.path.join(path, "splits.pickle"), "rb") as file:
                splits = pickle.load(file)

            with open(os.path.join(path, "mr2us_new.pickle"), "rb") as file:
                mr2us = pickle.load(file)["mr2us"]

            sb = mr2us[splits["Shallow Breathing"]["start"]-1], mr2us[splits["Shallow Breathing"]["end"]]
            rb = mr2us[splits["Regular Breathing"]["start"]-1], mr2us[splits["Regular Breathing"]["end"]]
            db = mr2us[splits["Deep Breathing"]["start"]-1], mr2us[splits["Deep Breathing"]["end"]]
            dbh = mr2us[splits["Deep BH"]["start"]-1], mr2us[splits["Deep BH"]["end"]]
            hbh = mr2us[splits["Half Exhale BH"]["start"]-1], mr2us[splits["Half Exhale BH"]["end"]]
            febh = mr2us[splits["Full Exhale BH"]["start"]-1], mr2us[splits["Full Exhale BH"]["end"]]

            us = np.array(surrogates["us"]).T
            roi = (0, 1000)
            hilbert_us = hilbert(us, axis=0)
            hilbert_phase = np.angle(hilbert_us)
            hilbert_magnitude = np.abs(hilbert_us)
            z = extract_wave(hilbert_phase, roi)

            trend = np.zeros_like(z)
            for w in [dbh, sb, hbh, rb, febh, db]:
                z_w = z[w[0]:w[1]]
                z_trend = detrend_madore(z_w, hilbert_magnitude[:, w[0]:w[1]])
                trend[w[0]:w[1]] = z_trend + trend[w[0]-1]

            # final_trend = trends[0]
            # for i in range(1, len(trends)):
            #     val = final_trend[-1]
            #     new_trend = trends[i] + val
            #     final_trend = np.concatenate([final_trend, new_trend])

            detrended = z - trend

            plt.figure()
            plt.title("Detrended")
            plt.plot(detrended - detrended.mean())
            plt.ylim([-.5, .5])
            plt.show()

            plt.savefig(f"{subject}.png")

            with open(os.path.join(path, "us_wave_detrended.pickle"), "wb") as file:
                pickle.dump(detrended, file)


if __name__ == '__main__':
    main()