import os, pickle, glob

def main():
    # save_path = os.path.join("C:", os.sep, "data", "Formatted_datasets", "A1")

    # with open(os.path.join(save_path, "surrogates.pickle"), "rb") as file:
    #     surrogates = pickle.load(file)

    # print(surrogates["ts"])

    coil_paths = glob.glob("C:\data\session1_rgb\*.png")
    print(coil_paths[0].split("_")[-1].replace())



if __name__ == '__main__':
    main()