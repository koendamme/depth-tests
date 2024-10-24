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


def get_min_value(mr_wave_path):
    with open(mr_wave_path, "rb") as file:
        wave = np.array(pickle.load(file)["mri_waveform"])
        return np.min(wave[wave != 0])


def single_session():
    # plt.rcParams.update({'font.size': 13})
    model = "coil_model"
    fig, axs = plt.subplots(nrows=1, ncols=6, sharey=True, figsize=(13, 2))
    # session = "B2"
    session = "F3"
    breathing_patterns = ["Deep Breathing", "Shallow Breathing", "Regular Breathing", "Half Exhale BH", "Full Exhale BH", "Deep BH"]
    settings_path = f"F:\\Formatted_datasets\\{session}\\settings.json"
    min_value = get_min_value(f"F:\\Formatted_datasets\\{session}\\mr_wave.pickle")
    threshold, x = get_tracking_settings(settings_path)

    for j, pattern in enumerate(breathing_patterns):
        image_paths = glob.glob(os.path.join("F:\\results", model, session, pattern, "*.png"))
        real_waveform, fake_waveform, ssims = [], [], []
        image_paths.sort(key=lambda p: int(p.split("\\")[-1].split(".")[0]))
        for path in image_paths:
            img = cv2.imread(path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fake = img_gray[:, img_gray.shape[1] // 2:]
            real = img_gray[:, :img_gray.shape[1] // 2]
            y_real, _ = get_current_border_position(real, threshold, x)
            y_fake, _ = get_current_border_position(fake, threshold, x)
            real_waveform.append(y_real * 1.9)
            fake_waveform.append(y_fake * 1.9)

        real_waveform, fake_waveform = np.array(real_waveform), np.array(fake_waveform)
        axs[j].plot(real_waveform - min_value)
        axs[j].plot(np.clip(fake_waveform - min_value, a_min=0, a_max=None))
        axs[j].set_ylim([0, 100])

    # fig.suptitle(f"Subject {session[0]}")
    # fig.supxlabel('Frame')
    # fig.supylabel('Liver Displacement (mm)')
    # fig.tight_layout()
    # fig.subplots_adjust(left=0.09,
    #                     bottom=0.1,
    #                     right=0.9,
    #                     top=.93,
    #                     wspace=0.1,
    #                     hspace=0.3)
    plt.savefig(f"{model}_{session}.png")


def main():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 13})
    model = "coil_model"
    fig, axs = plt.subplots(nrows=7, ncols=6, sharey=True, figsize=(15, 10))
    for i, session in enumerate(["A3", "B2", "C3", "D3", "E2", "F3", "G4"]):
        breathing_patterns = ["Deep Breathing", "Shallow Breathing", "Regular Breathing", "Half Exhale BH", "Full Exhale BH", "Deep BH"]
        settings_path = f"F:\\Formatted_datasets\\{session}\\settings.json"
        min_value = get_min_value(f"F:\\Formatted_datasets\\{session}\\mr_wave.pickle")
        threshold, x = get_tracking_settings(settings_path)

        for j, pattern in enumerate(breathing_patterns):
            image_paths = glob.glob(os.path.join("F:\\results", model, session, pattern, "*.png"))
            real_waveform, fake_waveform, ssims = [], [], []
            image_paths.sort(key=lambda p: int(p.split("\\")[-1].split(".")[0]))
            for path in image_paths:
                img = cv2.imread(path)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                fake = img_gray[:, img_gray.shape[1] // 2:]
                real = img_gray[:, :img_gray.shape[1] // 2]
                y_real, _ = get_current_border_position(real, threshold, x)
                y_fake, _ = get_current_border_position(fake, threshold, x)
                real_waveform.append(y_real * 1.9)
                fake_waveform.append(y_fake * 1.9)

            real_waveform, fake_waveform = np.array(real_waveform), np.array(fake_waveform)
            axs[i, j].plot(real_waveform - min_value)
            axs[i, j].plot(fake_waveform - min_value)
            axs[i, j].set_ylim([0, 100])
            if i == 0:
                axs[i, j].set_title(pattern)
            if j == 0:
                axs[i, j].set_ylabel(f"Subject {session[0]}")

        # fig.suptitle(f"Subject {session[0]}")
        fig.supxlabel('Frame')
        fig.supylabel('Liver Displacement (mm)')
        fig.tight_layout()
        fig.subplots_adjust(left=0.09,
                            bottom=0.1,
                            right=0.9,
                            top=.93,
                            wspace=0.1,
                            hspace=0.3)
    plt.savefig(f"{model}_all_subjects.png")
        # plt.close()


if __name__ == "__main__":
    single_session()