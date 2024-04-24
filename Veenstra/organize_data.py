import pydicom
import glob
import os
import json


def organize_mr():
    paths = glob.glob("A/DICOM/I*")
    paths.sort(key=lambda x: int(x.split("/I")[1]))
    mr_data = []

    for i, path in enumerate(paths):
        dcm = pydicom.dcmread(path)
        try:
            img = dcm.pixel_array
            if dcm.SeriesDescription.startswith("DynbFFE") and 'M_FFE' in dcm.ImageType:
                d = {
                    "timestamp": dcm.AcquisitionTime,
                    "img": img.tolist()
                }
                mr_data.append(d)
        except Exception:
            pass

    with open(os.path.join("new_A", "mri.json"), 'w') as f:
        json.dump(mr_data, f)


def organize_us():
    us_data = []
    with open("A/US/Wave.opt", "r") as f:
        times, us_data = [], []

        for i, line in enumerate(f):
            row = []
            if i % 2 == 0:
                time = line.split(",")[0].replace("TIME =", "")
                times.append(time)
            else:
                for y in line.replace("\n", "").split(", "):
                    row.append(float(y))
                us_data.append(row)

    print(len(times), len(us_data))

    with open(os.path.join("new_A", "us.json"), 'w') as f:
        json.dump({
            "times": times,
            "us_data": us_data
        }, f)


def main():
    # organize_mr()
    organize_us()

if __name__ == '__main__':
    main()