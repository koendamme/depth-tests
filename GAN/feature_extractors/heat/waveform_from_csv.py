import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np


def read_heat_waveform(path):
    df = pd.read_csv(path)
    c = df.columns.copy().to_list()
    c[-1] = "Temp"
    df.columns = c

    temps = df.loc[:, "Temp"].to_numpy()
    times = df.loc[:, "abstime"].to_list()
    timestamps = [(datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') + timedelta(hours=2)).timestamp() for d in times]

    return np.array(timestamps), temps


def main():
    p = os.path.join("C:", os.sep, "data", "A_raw", "session3 (2 rerun)", "heat", "raw_waveform.csv")
    t, d = read_heat_waveform(p)
    print()



if __name__ == '__main__':
    main()