import numpy as np
import pydicom
from datetime import datetime


def preprocess_image(img):
    img = np.array(img)
    return np.uint8(img)


def get_frequency():
    dcm1 = pydicom.read_file(r"C:\data\mri_us_experiments_14-5\mri\session3\1\DICOM\1.3.12.2.1107.5.2.18.152379.30000024051408133963400000004-1004-1-1g51hzo.dcm")
    dcm2 = pydicom.read_file(r"C:\data\mri_us_experiments_14-5\mri\session3\1014\DICOM\1.3.12.2.1107.5.2.18.152379.30000024051408133963400000004-2017-1-cvvjoq.dcm")
    t1 = datetime.fromtimestamp(float(dcm1.AcquisitionTime))
    t2 = datetime.fromtimestamp(float(dcm2.AcquisitionTime))
    print(t1, t2)
    return 0



if __name__ == '__main__':
    freq = get_frequency()

    print(freq)