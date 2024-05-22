import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
from us.recording_to_image import get_full_image
from us.extract_us_wave import get_wave
import numpy as np
from datetime import datetime, timedelta


def process_us_timestamps(start_timestamp, counters):
    differences = np.diff(counters)
    us_timestamps = []
    total_time = 0

    us_timestamps.append(start_timestamp)
    for i in range(1, len(counters)):
        total_time += differences[i-1]
        us_timestamps.append(start_timestamp + total_time)

    final = [datetime.fromtimestamp(d_obj).replace(year=1900, month=1, day=1) for d_obj in us_timestamps]
    return final


def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))


# us_path = r"D:\techmed_synchronisatie_1-5\Test1\us\2024-05-01 13,50,58.pickle"
us_path = r"D:\experiment-13-5\Test1\us\2024-05-13 15,35,02.pickle"
start_date = us_path.split("\\")[-1].replace(".pickle", "")
start_timestamp = datetime.strptime(start_date, '%Y-%m-%d %H,%M,%S').timestamp()

us_image, counters = get_full_image(us_path)
# us_image = np.clip(us_image, a_min=0, a_max=None)
us_image = us_image[300:800]
us_wave = get_wave(us_image)

us_timestamps = process_us_timestamps(start_timestamp, counters)

# heat_data = pd.read_pickle(os.path.join("heat", "heat_wave.pickle"))
# cleaned_heat_timestamps = [ts.split(':', 1)[1] for ts in heat_data["timestamps"]]
# cleaned_heat_time_objects = [(datetime.strptime(ts, '%H:%M:%S.%f') + timedelta(hours=2)) for ts in cleaned_heat_timestamps]

depth_data = pd.read_pickle(os.path.join("depth", "depth_wave_arend_experiment.pickle"))
depth_timestamps = [datetime.fromtimestamp(row).replace(year=1900, month=1, day=1) for row in depth_data["timestamps"]]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.imshow(us_image, cmap='gray', extent=[0, len(us_timestamps), 0, 10])
ax1.set_title("Depth data")
ax1.plot(depth_timestamps, depth_data["distandes"])
#
# ax2.set_title("Heat data")
# ax2.plot(cleaned_heat_time_objects, heat_data["temps"])

ax2.set_title("US data")
ax2.plot(us_timestamps, us_wave)
ax2.set_xlabel("Timestamp")
plt.tight_layout()

def on_click(event):
    global points

    if event.button == 3:  # Check if left mouse button is clicked
        # plt.axvline(event.xdata, color='r', linestyle='--')
        print(mdates.num2date(event.xdata))
        ax1.vlines(event.xdata, ymin=-.002, ymax=.006, color='r', linestyle='--')
        ax2.vlines(event.xdata, ymin=150, ymax=400, color='r', linestyle='--')
        points.append(event.xdata)
        plt.draw()  # Redraw the plot with the vertical line

# Connect mouse click event to the function
points = []
plt.gcf().canvas.mpl_connect('button_press_event', on_click)
plt.show()

# print(points)
#
# print(datetime.fromtimestamp(-25566.419715305154))

# diff = points[1] - points[0]
#
# print(diff)

