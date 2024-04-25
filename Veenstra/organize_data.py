import pydicom
import glob
import os
import json


def organize_mr():
    paths = glob.glob(os.path.join("A", "MRI", "I*"))
    paths.sort(key=lambda x: int(x.split(os.sep + "I")[1]))
    imgs, times = [], []

    for i, path in enumerate(paths):
        dcm = pydicom.dcmread(path)
        try:
            img = dcm.pixel_array
            if dcm.SeriesDescription.startswith("DynbFFE") and 'M_FFE' in dcm.ImageType:
                imgs.append(img.tolist())
                times.append(dcm.AcquisitionTime)
        except Exception:
            pass

    with open(os.path.join("new_A", "mri.json"), 'w') as f:
        json.dump({
            "times": times,
            "mri_data": imgs
        }, f)


def organize_us():
    us_data = []
    with open(os.path.join("A", "US", "Wave.opt"), "r") as f:
        times, us_data = [], []

        for i, line in enumerate(f):
            row = []
            if i % 2 == 0:
                time = line.split(",")[0].replace("TIME = ", "")
                times.append(time)
            else:
                for y in line.replace("\n", "").split(", "):
                    row.append(float(y))
                us_data.append(row)

    with open(os.path.join("new_A", "us.json"), 'w') as f:
        json.dump({
            "times": times,
            "us_data": us_data
        }, f)


def main():
    organize_mr()
    organize_us()


if __name__ == '__main__':
    main()