import pickle
import numpy as np
import pandas as pd

def load_to_memory(path):
    data = []
    ts = []
    with open(path, "rb") as openfile:
        while True:
            try:
                row = pickle.load(openfile)
                data.append(row["data"])
                ts.append(row["ts"])
            except EOFError:
                break

    return data, ts


def main():
    data, ts = load_to_memory(r"C:\dev\ultrasound\data\test1.pickle")
    ts2 = pd.read_pickle(r"C:\data\mri_us_experiments_14-5\us\2024-05-14 11,06,37_times.pickle")

    ts = np.array(ts)
    ts2 = np.array(ts2)

    diff = np.diff(ts)
    diff2 = np.diff(ts2)

    print(np.std(diff), np.std(diff2))
    print(np.mean(diff[:20]))


if __name__ == "__main__":
    main()