from GAN.dataset import CustomDataset
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
import numpy as np
import pandas as pd


def main():
    print("Hi")
    root = os.path.join("F:", os.sep, "Formatted_datasets")
    patterns = ["Shallow Breathing", "Regular Breathing", "Deep Breathing"]

    final_df = None
    for pp in ["A", "B", "C"]:
        for session in [1, 2, 3]:
            subject = pp + str(session)
            print(subject)

            df = pd.DataFrame(index=[subject+"ptt", subject+"ptt_std"], columns=patterns)

            dataset = CustomDataset(root, subject)

            for pattern in patterns:
                start = dataset.splits[pattern]["start"]
                end = dataset.splits[pattern]["end"]

                x = dataset.mr_wave[start:end]
                peaks, _ = find_peaks(x, distance=5, prominence=5)
                troughs, _ = find_peaks(-x, distance=5, prominence=5)

                shortest_length = min(peaks.shape[0], troughs.shape[0])

                peaks = peaks[:shortest_length]
                troughs = troughs[:shortest_length]

                ptt = (np.abs(x[peaks]-x[troughs])*1.9).numpy()
                df.loc[subject+"ptt", pattern] = ptt.mean()
                df.loc[subject+"ptt_std", pattern] = ptt.std()
  
            if final_df is None:
                final_df = df
            else:
                final_df = pd.concat([final_df, df])

    final_df.to_excel(os.path.join("F:", os.sep, "results", "ppt.xlsx"))
        


if __name__ == '__main__':
    main()