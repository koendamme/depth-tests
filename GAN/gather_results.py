import pandas as pd
import os
import json
import glob
import cv2
from feature_extractors.mri.mri_improvement import get_current_border_position
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle


def get_tracking_settings(path):
    with open(path) as file:
        settings = json.load(file)
        threshold = settings["MRI"]["Updated_Waveform_parameters"]["Threshold"]
        x = settings["MRI"]["Updated_Waveform_parameters"]["x"]

    return threshold, x


def main():
    model = "coil_model"
    breathing_patterns = ["Deep Breathing", "Shallow Breathing", "Regular Breathing", "Half Exhale BH", "Full Exhale BH", "Deep BH"]
    metrics = ["MAE", "MAE_std","R2", "SSIM", "SSIM_std", "PIQUE", "PIQUE_std"]
    final_df = None
    for subject in ["A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]:
        df = pd.DataFrame(columns=metrics, index=breathing_patterns)
        save_path = os.path.join("C:", os.sep, "data", "results", model)
        settings_path = os.path.join("C:", os.sep, "data", "Formatted_datasets", subject, "settings.json")
        threshold, x = get_tracking_settings(settings_path)

        for pattern in breathing_patterns:
            image_paths = glob.glob(os.path.join(save_path, subject, pattern, "*.png"))
            real_waveform, fake_waveform, ssims = [], [], []
            image_paths.sort(key=lambda p: int(p.split("\\")[-1].split(".")[0]))
            for path in image_paths:
                img = cv2.imread(path)
                assert np.any(img[:, :, 0] == img[:, :, 1]) and np.any(img[:, :, 1] == img[:, :, 2])
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                real = img_gray[:, img_gray.shape[1]//2:]
                fake = img_gray[:, :img_gray.shape[1]//2]
                y_real, _ = get_current_border_position(real, threshold, x-32)
                y_fake, _ = get_current_border_position(fake, threshold, x-32)
                real_waveform.append(y_real*1.9)
                fake_waveform.append(y_fake*1.9)
                ssims.append(ssim(real, fake, data_range=255))

            real_waveform, fake_waveform, ssims = np.array(real_waveform), np.array(fake_waveform), np.array(ssims)

            plt.figure()
            plt.title(f"{subject}_{pattern}")
            plt.plot(real_waveform/1.9, label="real")
            plt.plot(fake_waveform/1.9, label="real")
            if subject.startswith("A"):
                plt.ylim([40, 140])
            else:
                plt.ylim([20, 110])

            plt.savefig(os.path.join(save_path, f"{subject}_{pattern}.png"))
            plt.close()

            mae = np.abs(real_waveform - fake_waveform)
            r2 = r2_score(real_waveform, fake_waveform)
            df.loc[pattern, "MAE"] = mae.mean()
            df.loc[pattern, "MAE_std"] = mae.std()
            df.loc[pattern, "R2"] = r2
            df.loc[pattern, "SSIM"] = ssims.mean()
            df.loc[pattern, "SSIM_std"] = ssims.std()

        if final_df is None:
            final_df = df
        else:
            final_df = pd.concat([final_df, df])
    
    final_df.to_excel(os.path.join(save_path, f"sub_results_{model}.xlsx"))


if __name__ == "__main__":
    # img = cv2.imread("C:\\data\\results\\combined_model\\B3\\Deep Breathing\\3.png")
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # real = img_gray[:, img_gray.shape[1]//2:]
    # fake = img_gray[:, :img_gray.shape[1]//2]

    # with open("C:\data\Formatted_datasets\B3\mr.pickle", "rb") as file:
    #     mr = np.array(pickle.load(file)["images"])[500]
    #     mr = np.clip(mr, a_min=0, a_max=255)
    #     mr = cv2.addWeighted(mr, 1.7, np.zeros(mr.shape, mr.dtype), 0, 0)
    #     mr = mr[:128, 32:-32]

    # hist, bins = np.histogram(mr.flatten(), bins=256, range=[0, 256])

    # plt.figure()
    # plt.title("Real")
    # plt.xlabel("Intensity Value")
    # plt.ylabel("Pixel Count")
    # plt.bar(bins[:-1], hist, width=1, edgecolor="black")
    # plt.show()

    main()