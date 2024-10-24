from GAN.dataset import CustomDataset
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
import numpy as np
import pandas as pd


def create_image():
    root = os.path.join("F:", os.sep, "Formatted_datasets")
    subject = "A2"
    pattern = "Deep Breathing"
    dataset = CustomDataset(root, subject)
    start = dataset.splits[pattern]["start"]
    end = dataset.splits[pattern]["end"]

    x = dataset.mr_wave[start+100:start+250]
    peaks, _ = find_peaks(x, distance=5, prominence=5)
    troughs, _ = find_peaks(-x, distance=5, prominence=5)

    plt.plot(x)
    plt.scatter(peaks, x[peaks], label="Peaks")
    plt.scatter(troughs, x[troughs], label="Troughs")
    plt.legend(loc="upper right")
    plt.ylabel("Displacement (mm)")
    plt.xlabel("Frame")
    plt.show()




def main():
    root = os.path.join("F:", os.sep, "Formatted_datasets")
    patterns = ["Shallow Breathing", "Regular Breathing", "Deep Breathing"]

    final_df = None
    # for subject in ["D1", "D2", "D3", "E1", "E2", "E3", "F1", "F3", "F4", "G2", "G3", "G4"]:
    for subject in ["E1", "E2", "E3"]:
        df = pd.DataFrame(index=[subject+"ptt", subject+"ptt_std"], columns=patterns)
        dataset = CustomDataset(root, subject)

        for pattern in patterns:
            start = dataset.splits[pattern]["start"]
            end = dataset.splits[pattern]["end"]

            x = dataset.mr_wave[start:end]
            peaks, _ = find_peaks(x, distance=5, prominence=5)
            troughs, _ = find_peaks(-x, distance=5, prominence=5)

            plt.plot(x)
            plt.scatter(peaks, x[peaks])
            plt.scatter(troughs, x[troughs])
            plt.show()

    #         shortest_length = min(peaks.shape[0], troughs.shape[0])

    #         peaks = peaks[:shortest_length]
    #         troughs = troughs[:shortest_length]

    #         ptt = (np.abs(x[peaks]-x[troughs])*1.9).numpy()
    #         df.loc[subject+"ptt", pattern] = ptt.mean()
    #         df.loc[subject+"ptt_std", pattern] = ptt.std()

    #     if final_df is None:
    #         final_df = df
    #     else:
    #         final_df = pd.concat([final_df, df])

    # final_df.to_excel(os.path.join("F:", os.sep, "results", "ppt_final_subjects.xlsx"))
        


if __name__ == '__main__':
    # create_image()
    main()
