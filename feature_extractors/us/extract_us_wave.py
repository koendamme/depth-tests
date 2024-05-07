import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_wave(full_image):
    full_image = np.clip(full_image, a_min=0, a_max=None)

    index = np.arange(500)[None, :]
    a = index @ full_image
    b = np.sum(full_image, axis=0)[None, :]
    c = a/b

    return c[0]

def main():
    data = pd.read_pickle(r"D:\techmed_synchronisatie_1-5\Test1\us\2024-05-01 13,50,58.pickle")

    full_image = np.zeros((len(data), len(data[0][0])))

    for i in range(len(data)):
        full_image[i, :] = data[i][0]

    full_image = full_image.T
    # full_image = full_image[500:1000]
    # full_image = np.clip(full_image, a_min=0, a_max=None)

    # index = np.arange(500)[None, :]
    # a = index @ full_image
    # b = np.sum(full_image, axis=0)[None, :]
    # c = a/b

    c = get_wave(full_image)

    plt.plot(c)
    plt.show()



if __name__ == '__main__':
    main()