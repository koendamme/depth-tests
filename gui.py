import tkinter as tk
from depth_camera import DepthCamera
from PIL import Image, ImageTk
import numpy as np
import threading
import cv2


class GUI:
    def __init__(self, depth_camera):
        self.depth_camera = depth_camera

        self.depth_frame = None
        self.color_frame = None
        self.thread = None
        self.stopEvent = None

        self.root = tk.Tk()
        self.depth_panel = None
        self.color_panel = None

        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.video_loop, args=())
        self.thread.start()
        self.root.wm_title("Depth Camera")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def video_loop(self):
        try:
            while not self.stopEvent.is_set():
                if self.depth_camera.device_connected:
                    print("Imgaes from device")
                    depth, color = self.depth_camera.get_frame()
                    depth = ImageTk.PhotoImage(image=Image.fromarray((depth*255).astype(np.uint8)))
                    color = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor((color*255).astype(np.uint8), cv2.COLOR_BGR2RGB)))
                else:
                    depth = ImageTk.PhotoImage(image=Image.fromarray((np.random.rand(480, 640)*255).astype(np.uint8)))
                    color = ImageTk.PhotoImage(image=Image.fromarray((np.random.rand(480, 640, 3)*255).astype(np.uint8)))

                # if the panel is not None, we need to initialize it
                if self.depth_panel is None:
                    self.depth_panel = tk.Label(image=depth)
                    self.depth_panel.place(x=10, y=10)
                    self.depth_panel.pack(side="left")

                if self.color_panel is None:
                    self.color_panel = tk.Label(image=color)
                    self.color_panel.place(x=700, y=10)
                    self.color_panel.pack(side="right")

                # otherwise, simply update the panel
                else:
                    self.depth_panel.configure(image=depth)
                    self.depth_panel.image = depth

                    self.color_panel.configure(image=color)
                    self.color_panel.image = color

        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    def onClose(self):
        print("[INFO] closing...")
        self.stopEvent.set()
        self.root.quit()


def main():
    print("Hallo?")
    depth_camera = DepthCamera((.3, 6))

    gui = GUI(depth_camera)
    gui.root.mainloop()


if __name__ == "__main__":
    main()



#
# window = tk.Tk()
#
# while True:
#     depth, color = depth_camera.get_frame()
#
#     canvas = tk.Canvas(window, width = 1000, height = 800)
#     canvas.pack(expand=True)
#
#     pil_depth = ImageTk.PhotoImage(image=Image.fromarray((depth*255).astype(np.uint8)))
#     pil_color = ImageTk.PhotoImage(image=Image.fromarray((color*255).astype(np.uint8)))
#
#     tk.Label(image=pil_depth, width=298, height=298).place(x=10, y=10)
#     tk.Label(image=pil_color, width=298, height=298).place(x=320, y=10)
#
# # canvas = tk.Canvas(window,width=300,height=300)
# # canvas.pack()
# # canvas.create_image(20,20, anchor="nw", image=img)
# window.mainloop()