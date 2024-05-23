import pickle
import numpy as np
from datetime import datetime
from us.extract_us_wave import get_wave
import matplotlib.pyplot as plt
import math


def find_synchronization_points(us_waveform, mri_waveform):
    us_points, mri_points = [], []
    def on_click_first(event):
        if event.inaxes is not None and event.button == 3:
            mri_points.append(event.xdata)
            event.inaxes.plot(event.xdata, event.ydata, 'ro')
            event.inaxes.figure.canvas.draw()
    def on_click_second(event):
        if event.inaxes is not None and event.button == 3:
            us_points.append(event.xdata)
            event.inaxes.plot(event.xdata, event.ydata, 'ro')
            event.inaxes.figure.canvas.draw()

        if len(us_points) == len(mri_points):
            plt.close(event.canvas.figure)

    fig, ax = plt.subplots()
    ax.plot(mri_waveform)
    fig.canvas.mpl_connect('button_press_event', on_click_first)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(us_waveform)
    fig.canvas.mpl_connect('button_press_event', on_click_second)
    plt.show()

    return np.array(us_points), np.array(mri_points)


def synchronize(mri_waveform, us_waveform, show_result=True):
    us_freq, mri_freq = 20, 2.90

    us_points, mri_points = find_synchronization_points(us_waveform, mri_waveform)
    print(us_points, mri_points)

    us_times = us_points / us_freq
    mri_times = mri_points / mri_freq

    offset = np.mean(mri_times - us_times)
    us_idxs = np.arange(len(us_waveform))
    mr_idxs = np.arange(len(mri_waveform))

    if show_result:
        t_us = offset + us_idxs/us_freq
        t_mr = mr_idxs/mri_freq

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        ax1.set_title("MRI")
        ax1.plot(t_mr, mri_waveform)
        ax2.set_title("US")
        ax2.plot(t_us, us_waveform)
        plt.show()

    mr2us = []
    for mr in mr_idxs:
        us = math.floor((mr/mri_freq - offset)*us_freq)
        mr2us.append(us)

    return mr2us


def main():
    with open("mri/waveform_data.pickle", 'rb') as file:
        mri_waveform = pickle.load(file)

    us_waveform = get_wave(r"C:\data\mri_us_experiments_14-5\us\2024-05-14 11,06,37.pickle", 200, 800)

    mr2us = synchronize(mri_waveform, us_waveform)

    # with open("waveform_data.pickle", 'wb') as file:
    #     pickle.dump(mr2us, file)


if __name__ == '__main__':
    main()
