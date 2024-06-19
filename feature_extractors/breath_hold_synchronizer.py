import pickle
import numpy as np
from us.extract_us_wave import get_wave_updated
import matplotlib.pyplot as plt
import math
from feature_extractors.surrogate_synchronizer import synchronize_signals


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


def synchronize(mri_waveform, us_waveform, us_freq, mri_freq, show_result=True):
    us_points, mri_points = find_synchronization_points(us_waveform, mri_waveform)

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

    heat_path = r'C:\Users\kjwdamme\Desktop\Rec-000017.seq'
    us_path = r"C:\data\MRI-28-5\session1.pickle"

    surrogates = synchronize_signals(heat_path, us_path)
    us_waveform = get_wave_updated(surrogates["us"], 500, 1000, smooth=True)

    mr2us = synchronize(mri_waveform, us_waveform)

    # with open("waveform_data.pickle", 'wb') as file:
    #     pickle.dump(mr2us, file)


if __name__ == '__main__':
    main()
