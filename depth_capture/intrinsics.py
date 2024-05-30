import open3d


class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __init__(self, width, height):
        if width == 640 and height == 480:
            self.width = 640
            self.height = 480
            self.fx = 464.66015625
            self.fy = 464.49609375
            self.cx = 292.8359375
            self.cy = 264.26171875

        elif width == 1024 and height == 768:
            self.width = 1024
            self.height = 768
            self.fx = 743.8359375
            self.fy = 743.8515625
            self.cx = 468.515625
            self.cy = 422.953125

        elif width == 320 and height == 240:
            self.width = 320
            self.height = 240
            self.fx = 232.955078125
            self.fy = 233.029296875
            self.cx = 146.12890625
            self.cy = 131.91796875

    def get_intrinsics(self):
        return open3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.fx, self.fy, self.cx, self.cy)