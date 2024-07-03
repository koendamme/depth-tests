from fnv.file import ImagerFile
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from datetime import datetime, timedelta


def process_timestamp(timestamp):
    day_of_year, time_part = timestamp.split(':', 1)
    day_of_year = int(day_of_year)
    base_date = datetime(year=2024, month=1, day=1)
    actual_date = base_date + timedelta(days=day_of_year - 1)
    final_datetime_str = actual_date.strftime('%Y-%m-%d') + ' ' + time_part
    final_datetime = datetime.strptime(final_datetime_str, '%Y-%m-%d %H:%M:%S.%f')
    final_datetime = final_datetime + timedelta(hours=2)
    return final_datetime


def extract_heat(path):
    x, y = None, None
    circle_radius = 25

    def on_click(event):
        nonlocal x, y
        if event.button == 1:  # Left mouse button
            x = event.xdata
            y = event.ydata

            circle = plt.Circle((x, y), circle_radius, color='red', fill=False)
            ax.add_patch(circle)
            fig.canvas.draw()

    im = ImagerFile(path)

    im.get_frame(2500)
    heat_img = np.array(im.final)
    heat_img = heat_img.reshape((480, 640))

    fig, ax = plt.subplots()
    ax.imshow(heat_img)

    # Connect the click event to the handler
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Display the plot
    plt.show()
    x_grid, y_grid = np.meshgrid(np.arange(640), np.arange(480))

    dists = ((x_grid - x) ** 2 + (y_grid - y) ** 2) ** .5
    mask = dists < circle_radius

    temps = []
    timestamps = []

    print("Extracting temperatures...")
    for i in tqdm(range(im.num_frames)):
        im.get_frame(i)
        curr = np.array(im.final)
        curr = curr.reshape((480, 640))
        temps.append(np.average(curr[mask]))

        t = im.frame_info[0]["value"]
        t = process_timestamp(t).timestamp()
        timestamps.append(t)

    return timestamps, temps


def main():
    ts, temps = extract_heat(r'C:\Users\kjwdamme\Desktop\Rec-000017.seq')
    plt.plot(temps)
    plt.show()


if __name__ == '__main__':
    main()
