from fnv.file import ImagerFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import pickle


def main():
    im = ImagerFile(r'D:\techmed_synchronisatie_1-5\Test1\heat\Rec-000005.seq')
    print("Loaded Sequence file")

    im.get_frame(2500)
    heat_img = np.array(im.final)
    heat_img = heat_img.reshape((480, 640))

    fig, ax = plt.subplots()

    ax.imshow(heat_img)

    circle_center = (450, 100)
    circle_radius = 60

    ax.add_patch(patches.Circle(circle_center, circle_radius, edgecolor='r', facecolor='none'))

    x, y = np.meshgrid(np.arange(640), np.arange(480))

    dists = ((x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2) ** .5
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
        timestamps.append(t)

    with open("heat_wave.pickle", 'wb') as f:
        pickle.dump({"temps": temps, "timestamps": timestamps}, f)


if __name__ == '__main__':
    main()
