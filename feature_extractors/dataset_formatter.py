from mri.extract_images_from_dicom import extract_images
from breath_hold_synchronizer import synchronize
import os
import pickle
from us.extract_us_wave import get_wave
from us.recording_to_image import get_full_image
from mri.extract_breathing_waveform import extract_waveform


def main():
    root_dir = r"C:\data\A"

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    mri_images = extract_images(root_dir=r"C:\data\mri_us_experiments_14-5\mri\session3")
    with open(os.path.join(root_dir, "mr.pickle"), 'wb') as file:
        pickle.dump({"images": mri_images}, file)

    us_image = get_full_image(r"C:\data\mri_us_experiments_14-5\us\2024-05-14 11,06,37.pickle")
    with open(os.path.join(root_dir, "us.pickle"), 'wb') as file:
        pickle.dump({"img": us_image}, file)

    us_waveform = get_wave(r"C:\data\mri_us_experiments_14-5\us\2024-05-14 11,06,37.pickle", 200, 800)
    mri_waveform = extract_waveform(mri_images)

    mr2us = synchronize(mri_waveform, us_waveform)

    with open(os.path.join(root_dir, "mr2us.pickle"), 'wb') as file:
        pickle.dump({"mr2us": mr2us}, file)


if __name__ == '__main__':
    main()