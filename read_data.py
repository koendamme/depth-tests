import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import glob
import open3d
from matplotlib.widgets import RectangleSelector

img_path = glob.glob("depth/*.png")[0]
depth = open3d.io.read_image(img_path)
fig, ax = plt.subplots()
ax.imshow(depth, cmap="gray")
roi = None

def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    global roi
    roi = [[x1, y1], [x2, y2]]
    plt.close()

rs = RectangleSelector(ax, line_select_callback,
                       useblit=False, button=[1],
                       minspanx=5, minspany=5, spancoords='pixels',
                       interactive=True)


plt.connect("key_press_event", rs)
plt.show()

