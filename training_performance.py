import wandb
from datetime import datetime
from run_names import combined_runs, heat_runs, coil_runs, us_runs


def main():

    api = wandb.Api()
    results = [0, 0, 0, 0]
    for i, model in enumerate([combined_runs, heat_runs, coil_runs, us_runs]):

        durations = []
        for run_name in model.values():
            # Fetch the run by name
            runs = api.runs(f"thesis-koen/CustomData-cProGAN-All_Surrogates", {"display_name": run_name})
            durations.append(runs[0].summary["_wandb"].runtime)

        results[i] = sum(durations)/len(durations)


    print(results)


if __name__ == "__main__":
    main()