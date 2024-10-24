import os
import glob
import cv2


def main():
    root_path = "F:\\results"

    models = ["coil_model", "combined_model", "heat_model", "us_model"]
    subjects = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2", "D3", "E1", "E2", "E3", "F1", "F3", "F4", "G2", "G3", "G4"]
    patterns = ["Regular Breathing", "Shallow Breathing", "Deep Breathing", "Deep BH", "Half Exhale BH", "Full Exhale BH"]

    for model in models:
        print(f"Generating videos for {model}...")
        for subject in subjects:
            for pattern in patterns:
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                fps = 10
                out = cv2.VideoWriter(os.path.join(root_path, model, subject, f"{pattern}.mp4"), fourcc, fps, (256, 128))

                curr_path = os.path.join(root_path, model, subject, pattern, "*.png")
                img_paths = glob.glob(curr_path)
                img_paths.sort(key=lambda p: int(p.split("\\")[-1].replace(".png", "")))

                for path in img_paths:
                    img = cv2.imread(path)
                    out.write(img)

                out.release()


if __name__ == "__main__":
    main()