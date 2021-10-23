import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import config
import cv2


def load_dicom_image(path, img_size=config.IMAGE_SIZE, voi_lut=True, rotate=0):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    if rotate > 0:
        rot_choices = [
            0,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        data = cv2.rotate(data, rot_choices[rotate])

    data = cv2.resize(data, (img_size, img_size))
    data = data - np.min(data)
    if np.min(data) < np.max(data):
        data = data / np.max(data)
    return data


def crop_img(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    c1, c2 = False, False
    try:
        rmin, rmax = np.where(rows)[0][[0, -1]]
    except:
        rmin, rmax = 0, img.shape[0]
        c1 = True

    try:
        cmin, cmax = np.where(cols)[0][[0, -1]]
    except:
        cmin, cmax = 0, img.shape[1]
        c2 = True
    bb = (rmin, rmax, cmin, cmax)
    
    if c1 and c2:
        return img[0:0, 0:0]
    else:
        return img[bb[0] : bb[1], bb[2] : bb[3]]


def extract_cropped_image_size(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    cropped_data = crop_img(data)
    resolution = cropped_data.shape[0]*cropped_data.shape[1]  
    return resolution