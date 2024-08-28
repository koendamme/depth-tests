import pandas as pd
import os
import glob
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def main():
    for model in ["combined_model", "heat_model", "coil_model", "us_model"]:
        print(f"Gathering results for {model}...")
        breathing_patterns = ["Deep Breathing", "Shallow Breathing", "Regular Breathing", "Half Exhale BH", "Full Exhale BH", "Deep BH"]
        metrics = ["SSIM", "SSIM_std"]
        final_df = None
        for subject in ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]:
            df = pd.DataFrame(columns=metrics, index=breathing_patterns)
            # save_path = os.path.join("F:", os.sep, "results", model)
            save_path = f"/Volumes/T9/results"

            for pattern in breathing_patterns:
                image_paths = glob.glob(os.path.join(save_path, model, subject, pattern, "*.png"))
                image_paths.sort(key=lambda p: int(p.split("/")[-1].split(".")[0]))
                ssims = []
                for path in image_paths:
                    img = cv2.imread(path)
                    assert np.any(img[:, :, 0] == img[:, :, 1]) and np.any(img[:, :, 1] == img[:, :, 2])
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    real = img_gray[:, img_gray.shape[1]//2:]
                    fake = img_gray[:, :img_gray.shape[1]//2]
                    ssims.append(ssim(real, fake, data_range=255, win_size=11))

                ssims = np.array(ssims)
                df.loc[pattern, "SSIM"] = ssims.mean()
                df.loc[pattern, "SSIM_std"] = ssims.std()

            if final_df is None:
                final_df = df
            else:
                final_df = pd.concat([final_df, df])
        
        final_df.to_excel(os.path.join(save_path, f"sub_results_ssim_{model}.xlsx"))


if __name__ == "__main__":
    main()