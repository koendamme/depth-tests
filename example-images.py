import os
import glob
import cv2
from skimage.metrics import structural_similarity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_ssims():
    root_path = "/Volumes/T9/results"
    model = "combined_model"
    # breathing_patterns = ["Deep Breathing", "Shallow Breathing", "Regular Breathing", "Half Exhale BH", "Full Exhale BH", "Deep BH"]
    breathing_patterns = ["Deep Breathing", "Shallow Breathing", "Regular Breathing"]

    results = []
    for subject in ["A1", "A2", "A3"]:
        for pattern in breathing_patterns:
            path = os.path.join(root_path, model, subject, pattern, "*.png")

            for p in glob.glob(path):
                img = cv2.imread(p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                fake = img[:, :img.shape[1] // 2]
                real = img[:, img.shape[1] // 2:]

                ssim = structural_similarity(real, fake, data_range=255, win_size=11)
                results.append({"path": p, "ssim": ssim})

    df = pd.DataFrame(results, columns=['path', 'ssim'])
    return df


def main():
    results = compute_ssims()
    results.sort_values(by=["ssim"], inplace=True, ascending=True)
    print(results.head())

    worst = cv2.imread(results.iloc[0]["path"])
    worst = cv2.cvtColor(worst, cv2.COLOR_BGR2GRAY)
    worst_real = worst[:, :worst.shape[1] // 2]
    worst_fake = worst[:, worst.shape[1] // 2:]

    median = cv2.imread(results.iloc[len(results) // 2]["path"])
    median = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    median_real = median[:, :median.shape[1] // 2]
    median_fake = median[:, median.shape[1] // 2:]
    
    best = cv2.imread(results.iloc[len(results) - 1]["path"])
    best = cv2.cvtColor(best, cv2.COLOR_BGR2GRAY)
    best_real = best[:, :best.shape[1] // 2]
    best_fake = best[:, best.shape[1] // 2:]

    worst_ssim = results.iloc[0]["ssim"]
    median_ssim = results.iloc[len(results) // 2]["ssim"]
    best_ssim = results.iloc[len(results) - 1]["ssim"]

    print(worst_ssim, median_ssim, best_ssim)

    cv2.imwrite("worst_real.png", worst_real)
    cv2.imwrite("worst_fake.png", worst_fake)

    cv2.imwrite("median_real.png", median_real)
    cv2.imwrite("median_fake.png", median_fake)

    cv2.imwrite("best_real.png", best_real)
    cv2.imwrite("best_fake.png", best_fake)

    # fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    # ax1.imshow(worst_real, cmap="gray")
    # ax1.axis("off")
    # ax2.imshow(worst_fake, cmap="gray")
    # ax2.axis("off")
    # worst_ssim = results.iloc[0]["ssim"]
    # print(f"Worst: {worst_ssim}")
    # fig.suptitle(f"SSIM: {round(worst_ssim, 2)}")
    # fig.tight_layout()
    # fig.subplots_adjust(left=0.04,
    #                     bottom=0.033,
    #                     right=0.96,
    #                     top=.979,
    #                     wspace=0.05,
    #                     hspace=0.05)
    # plt.savefig("worst.png")
    # # plt.show()

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    # ax1.imshow(median_real, cmap="gray")
    # ax1.axis("off")
    # ax2.imshow(median_fake, cmap="gray")
    # ax2.axis("off")
    # median_ssim = results.iloc[len(results) // 2]["ssim"]
    # print(f"Median: {median_ssim}")
    # fig.suptitle(f"SSIM: {round(median_ssim, 2)}")
    # fig.tight_layout()
    # fig.subplots_adjust(left=0.04,
    #                     bottom=0.033,
    #                     right=0.96,
    #                     top=.979,
    #                     wspace=0.05,
    #                     hspace=0.05)
    # plt.savefig("median.png")

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    # ax1.imshow(best_real, cmap="gray")
    # ax1.axis("off")
    # ax2.imshow(best_fake, cmap="gray")
    # ax2.axis("off")
    # best_ssim = results.iloc[len(results) - 1]["ssim"]
    # print(f"Best: {best_ssim}")
    # fig.suptitle(f"SSIM: {round(best_ssim, 2)}")
    # fig.tight_layout()
    # fig.subplots_adjust(left=0.04,
    #                     bottom=0.033,
    #                     right=0.96,
    #                     top=.979,
    #                     wspace=0.05,
    #                     hspace=0.05)
    # plt.savefig("best.png")

    # axs[1, 0].imshow(median_real, cmap="gray")
    # axs[1, 0].axis("off")
    # axs[1, 1].imshow(median_fake, cmap="gray")
    # axs[1, 1].axis("off")

    # axs[2, 0].imshow(best_real, cmap="gray")
    # axs[2, 0].axis("off")
    # axs[2, 1].imshow(best_fake, cmap="gray")
    # axs[2, 1].axis("off")

    # fig.text(0.5, 0.92, 'Title for Row 1', ha='center', fontsize=16)
    # fig.text(0.5, 0.62, 'Title for Row 2', ha='center', fontsize=16)
    # fig.text(0.5, 0.32, 'Title for Row 3', ha='center', fontsize=16)

    # plt.tight_layout()
    # # fig.subplots_adjust(left=0.09,
    # #                     bottom=0.25,
    # #                     right=0.5,
    # #                     top=.93,
    # #                     wspace=0.05,
    # #                     hspace=0.05)
    # plt.show()








if __name__ == "__main__":
    main()