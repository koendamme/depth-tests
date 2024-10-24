import pickle
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def save_single_plot(subject):
        root = os.path.join("F:\\Formatted_datasets", subject)
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 4), sharex=True)

        with open(os.path.join(root, "surrogates.pickle"), 'rb') as file:
            surrogates = pickle.load(file)
        with open(os.path.join(root, "us_wave_detrended.pickle"), 'rb') as file:
            us_wave = pickle.load(file)
        with open(os.path.join(root, "mr_wave.pickle"), 'rb') as file:
            mr_wave = np.array(pickle.load(file)["mri_waveform"])
        with open(os.path.join(root, "mr2us_new.pickle"), 'rb') as file:
            mr2us = np.array(pickle.load(file)["mr2us"])

        us = us_wave[mr2us[0]:mr2us[-1]]
        heat = surrogates["heat"][mr2us[0]:mr2us[-1]]
        coil = surrogates["coil"][mr2us[0]:mr2us[-1]]

        x_mr = np.linspace(0, 1, len(mr_wave))
        x_surr = np.linspace(0, 1, us.shape[0])

        mr_filter = (x_mr>=.15) & (x_mr <= .2)
        surr_filter = (x_surr>=.15) & (x_surr <= .2)

        axs[0].plot(x_mr[mr_filter], mr_wave[mr_filter] - np.min(mr_wave[mr_filter]))
        # axs[1].plot(x_surr[surr_filter], us[surr_filter])
        axs[1].plot(x_surr[surr_filter], heat[surr_filter])
        # axs[3].plot(x_surr[surr_filter], coil[surr_filter])
        plt.savefig("heat_noise.png")

def main():
    for subject in ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2", "D3", "E1", "E2", "E3", "F1", "F3", "F4", "G2", "G3", "G4"]:
        root = os.path.join("F:\\Formatted_datasets", subject)

        with open(os.path.join(root, "surrogates.pickle"), 'rb') as file:
            surrogates = pickle.load(file)
        with open(os.path.join(root, "us_wave_detrended.pickle"), 'rb') as file:
            us_wave = pickle.load(file)
        with open(os.path.join(root, "mr_wave.pickle"), 'rb') as file:
            mr_wave = pickle.load(file)["mri_waveform"]
        with open(os.path.join(root, "mr2us_new.pickle"), 'rb') as file:
            mr2us = np.array(pickle.load(file)["mr2us"])

        us = us_wave[mr2us[0]:mr2us[-1]]
        heat = surrogates["heat"][mr2us[0]:mr2us[-1]]
        coil = surrogates["coil"][mr2us[0]:mr2us[-1]]

        x_mr = np.linspace(0, 1, len(mr_wave))
        x_surr = np.linspace(0, 1, us.shape[0])

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

        fig.add_trace(go.Scatter(x=x_mr, y=mr_wave, mode="lines", name="MR"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_surr, y=us, mode="lines", name="US"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_surr, y=coil, mode="lines", name="Coil"), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_surr, y=heat, mode="lines", name="Heat"), row=4, col=1)
        fig.update_layout(height=800, width=1200, title="Synchronized Surrogates")
        fig.write_html(f"F:\Formatted_datasets\Surrogate Figures\{subject}.html")


if __name__ == "__main__":
    # main()
    save_single_plot("C3")