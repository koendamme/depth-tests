import glob
import cv2
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm


def own_ssim(real, fake):
    mu_x = real.mean()
    mu_y = fake.mean()
    sigma_x_sq = real.var()
    sigma_y_sq = fake.var()

    sigma_xy = ((real - real.mean())*(fake - fake.mean())).mean()

    c_1 = (0.01*255)**2
    c_2 = (0.03*255)**2

    o = ((2*mu_x*mu_y + c_1)*(2*sigma_xy + c_2))/((mu_x**2 + mu_y**2 + c_1)*(sigma_x_sq + sigma_y_sq + c_2))

    return o


def compute_ssim(path, window):
    paths = glob.glob(path)
    paths.sort(key=lambda p: int(p.split("\\")[-1].replace(".png", "")))

    ssims = []
    for path in paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fake = img[:, :img.shape[1]//2]
        real = img[:, img.shape[1]//2:]

        ssim = structural_similarity(real, fake, data_range=256, win_size=window)
        ssims.append(ssim)

    return ssims


def compute_results(path):
    results = {
        "combined_model": {"Breathing": [], "Holding": []},
        "coil_model": {"Breathing": [], "Holding": []},
        "heat_model": {"Breathing": [], "Holding": []},
        "us_model": {"Breathing": [], "Holding": []}
    }
    for w in tqdm([11, 23, 35, 49, 75, 83, 99, 111, 127]):
        for model in results.keys():
            breathing_ssims, holding_ssims = [], []
            # print(f"Computing ssim values for {model}...")
            for subject in ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2", "D3", "E1", "E2", "E3", "F1", "F3", "F4", "G2", "G3", "G4"]:
                    subject_path = os.path.join(path, model, subject)

                    for breathing in ["Deep Breathing", "Regular Breathing", "Shallow Breathing"]:
                        pattern_path = os.path.join(subject_path, breathing, "*.png")
                        breathing_ssims.extend(compute_ssim(pattern_path, window=w))

                    for holding in ["Deep BH", "Full Exhale BH", "Half Exhale BH"]:
                        holding_path = os.path.join(subject_path, holding, "*.png")
                        holding_ssims.extend(compute_ssim(holding_path, window=w))

            results[model]["Breathing"].append(sum(breathing_ssims)/len(breathing_ssims))
            results[model]["Holding"].append(sum(holding_ssims)/len(holding_ssims))

    return results


def create_figure():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 13})
    path = "F:\\results"
    results = compute_results(path)
    
    with open('ssim_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # with open('ssim_results.json', 'r') as f:
    #     results = json.load(f)

    window_sizes = [11, 23, 35, 49, 75, 83, 99, 111, 127]
    models = ["Combined", "External", "Airflow", "internal"]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 5))

    ax1.set_title("Breathing")
    ax2.set_title("Breath Holding")
    for i, model in enumerate(results.keys()):
        ax1.plot(window_sizes, results[model]["Breathing"], label=models[i])
        ax2.plot(window_sizes, results[model]["Holding"], label=models[i])

    ax1.set_ylabel('SSIM')

    # Hide individual x-axis labels
    ax1.set_xlabel("Window size")
    ax2.set_xlabel("Window size")

    # Add a single x-axis label
    # fig.text(0.535, 0.01, 'Window size', ha='center', va='center')

    # fig.suptitle("Structural Similarity Index Measure")
    plt.tight_layout()
    plt.legend(loc="lower right")
    plt.savefig("ssim.png")
    plt.show()


def main():
    path = "/Volumes/T9/results/coil_model/A2/Regular Breathing/*.png"
    paths = glob.glob(path)
    paths.sort(key=lambda p: int(p.split("/")[-1].replace(".png", "")))

    final = None
    for p in paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fake = img[:, :img.shape[1] // 2]
        real = img[:, img.shape[1] // 2:]

        ssim = structural_similarity(real, fake, data_range=256, win_size=11, full=True)
        print(ssim[0])
        cv2.imshow("Frame", ssim[1])
        cv2.waitKey(0)

    #     final = ssim[1][None] if final is None else np.concatenate([final, ssim[1][None]], axis=0)
    #
    # avg = np.average(final, axis=0)
    #
    # plt.imshow(avg, cmap="gray")
    # plt.show()


if __name__ == "__main__":
    # main()
    create_figure()