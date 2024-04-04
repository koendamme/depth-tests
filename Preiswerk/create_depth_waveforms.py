import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    patient_paths = [
        # os.path.join("datasets", "A", "1", "depth_data.npy"),
        os.path.join("datasets", "B", "2", "depth_data.npy"),
        # os.path.join("datasets", "H", "2", "depth_data.npy")
    ]

    for p in patient_paths:
        print(p)
        with open(p, "rb") as f:
            d = np.load(f)

        dists = np.mean(np.abs(d - d[0, :]), axis=1)

        patient = p.split("/")[1]
        plt.title(f"Depth waveform patient {patient}")
        plt.xlabel("Frame")
        plt.ylabel("Distance")
        plt.plot(dists)
        plt.show()


if __name__ == '__main__':
    main()