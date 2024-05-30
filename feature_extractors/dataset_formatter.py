from mri.extract_images_from_dicom import extract_images
from breath_hold_synchronizer import synchronize
import os
import pickle
from feature_extractors.us.extract_us_wave import get_wave_updated
from feature_extractors.mri.extract_breathing_waveform import extract_waveform
from feature_extractors.surrogate_synchronizer import synchronize_signals


def main():
    root_dir = r"C:\data\A"
    heat_path = r'C:\Users\kjwdamme\Desktop\Rec-000017.seq'
    us_path = r"C:\data\MRI-28-5\session1.pickle"
    save_path = os.path.join(root_dir, "synchronized_surrogates.pickle")

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    mri_images = extract_images(root_dir=r"C:\data\MRI-28-5\MRI")
    with open(os.path.join(root_dir, "mr.pickle"), 'wb') as file:
        pickle.dump({"images": mri_images}, file)

    d = synchronize_signals(heat_path, us_path, save_path)

    us_waveform = get_wave_updated(d["us"], 500, 1000, smooth=True)
    mri_waveform = extract_waveform(mri_images)

    mr2us = synchronize(mri_waveform, us_waveform)

    with open(os.path.join(root_dir, "mr2us.pickle"), 'wb') as file:
        pickle.dump({"mr2us": mr2us}, file)


if __name__ == '__main__':
    main()