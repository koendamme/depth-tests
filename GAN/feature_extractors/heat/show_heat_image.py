import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fnv.file import ImagerFile

def main():
    im = ImagerFile(r'D:\techmed_synchronisatie_1-5\Test1\heat\Rec-000005.seq')
    print("Loaded Sequence file")

    im.get_frame(4000)
    heat_img = np.array(im.final)
    heat_img = heat_img.reshape((480, 640))

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(heat_img)

    circle_center = (450, 100)
    circle_radius = 60

    ax.add_patch(patches.Circle(circle_center, circle_radius, edgecolor='r', facecolor='none'))
    plt.show()


if __name__ == '__main__':
    main()