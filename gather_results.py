import pandas as pd
import os
import json
import glob
import cv2
from GAN.feature_extractors.mri.mri_improvement import get_current_border_position
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle


def get_tracking_settings(path):
    with open(path) as file:
        settings = json.load(file)
        threshold = settings["MRI"]["Updated_waveform_parameters"]["Threshold"]
        x = settings["MRI"]["Updated_waveform_parameters"]["x"]

    return threshold, x


def create_tracking_video():
    model = "combined_model"
    pattern = "Deep Breathing"
    subject = "A2"
    settings_path = os.path.join("F:", os.sep, "Formatted_datasets", subject, "settings.json")
    threshold, x = get_tracking_settings(settings_path)

    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fps = 10
    # out = cv2.VideoWriter(f"eval.mp4", fourcc, fps, (128*2, 128))

    image_paths = glob.glob(f"F:\\results\\{model}\\{subject}\\{pattern}\\*.png")
    image_paths.sort(key=lambda p: int(p.split("\\")[-1].split(".")[0]))
    real_wave, fake_wave = [], []
    for path in image_paths:
        img = cv2.imread(path)
        assert np.any(img[:, :, 0] == img[:, :, 1]) and np.any(img[:, :, 1] == img[:, :, 2])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fake = img_gray[:, :img_gray.shape[1]//2]
        real = img_gray[:, img_gray.shape[1]//2:]
        y_fake, _ = get_current_border_position(fake, threshold, x)
        y_real, _ = get_current_border_position(real, threshold, x)

        fake_wave.append(y_fake * 1.9)
        real_wave.append(y_real * 1.9)
        # cv2.circle(img, (x-5, y_fake), 3, (0, 0, 255), -1)
        # cv2.circle(img, (x + 128-5, y_real), 3, (0, 0, 255), -1)

        # out.write(img)

    # out.release()

    plt.plot(real_wave, label="Real")
    plt.plot(fake_wave, label="Fake")
    plt.ylabel("Displacement (mm)")
    plt.xlabel("Frame")
    plt.legend()
    plt.ylim([100, 200])
    plt.show()



def main():
    for model in ["coil_model", "combined_model", "heat_model", "us_model"]:
        print(f"Computing results for {model}...")

        breathing_patterns = ["Deep Breathing", "Shallow Breathing", "Regular Breathing", "Half Exhale BH", "Full Exhale BH", "Deep BH"]
        metrics = ["MAE", "MAE_std","R2", "SSIM", "SSIM_std", "PIQUE", "PIQUE_std"]
        final_df = None
        for subject in ["D1", "D2", "D3", "E1", "E2", "E3", "F1", "F3", "F4", "G2", "G3", "G4"]:
            df = pd.DataFrame(columns=metrics, index=breathing_patterns)
            save_path = os.path.join("F:", os.sep, "results", model)
            settings_path = os.path.join("F:", os.sep, "Formatted_datasets", subject, "settings.json")
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
                    y_real, _ = get_current_border_position(real, threshold, x)
                    y_fake, _ = get_current_border_position(fake, threshold, x)
                    if y_fake != 0 and y_real != 0:
                        real_waveform.append(y_real*1.9)
                        fake_waveform.append(y_fake*1.9)
                        ssims.append(ssim(real, fake, data_range=255))
                    else:  
                        print("Excluding from evaluation because there was no border found...")

                real_waveform, fake_waveform, ssims = np.array(real_waveform), np.array(fake_waveform), np.array(ssims)

                plt.figure()
                plt.title(f"{subject}_{pattern}")
                plt.ylabel("Displacement (mm)")
                plt.xlabel("Frame")
                plt.plot(real_waveform/1.9, label="real")
                plt.plot(fake_waveform/1.9, label="real")
                if subject.startswith("F") or subject.startswith("G"):
                    plt.ylim([50, 150])
                else:
                    plt.ylim([20, 110])

                plt.savefig(os.path.join(save_path, subject, f"{pattern}.png"))
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
        
        final_df.to_excel(os.path.join(save_path, f"sub_results_{model}_finalsubjects.xlsx"))


if __name__ == "__main__":
    # create_tracking_video()

    main()