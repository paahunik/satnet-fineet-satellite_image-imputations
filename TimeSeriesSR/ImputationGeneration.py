import time

import numpy as np
import tensorflow as tf
from libtiff import TIFF
import libtiff
import stippy
libtiff.libtiff_ctypes.suppress_warnings()
from time import strptime, mktime
from skimage import exposure
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import cv2
import datetime
import socket

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

def get_all_geohashes():
    """
    :return: list of all possible geohashes of length 5 for laton-ca-20km
    """
    geohashs_found = set()
    listImages = stippy.list_images('127.0.0.1:15606', album='laton-ca-20km',
                                    geocode='9q', recurse=True,
                                    platform='Sentinel-2', source='raw',
                                    start_timestamp=1514844983, end_timestamp=1515449783)

    for (node, image) in listImages:
        geohashs_found.add(image.geocode[:-1])
    return list(geohashs_found)


def get_encoding():
    """
    :return: label encoder to convert categorical labels to numerical values
    """
    le = LabelEncoder()
    return le.fit(get_all_geohashes())


def scale_images(imgs):
    """
    :param imgs: Input batch of images with pixel values between (0, 255)
    :return: Returns images after normalizing pixel values to (-1 to 1) range
    """
    return imgs / 127.5 - 1


def unscale_images(imgs):
    """
    :param imgs: Input batch of images with normalized pixel values between (-1, 1)
    :return:  Returns images with pixel values between (0 to 255) range
    """
    return (imgs + 1.) * 127.5


def paths_to_rgb_convertor(paths, isSentinel=True):
    """
    :param paths: List of n paths to be loaded as rgb image
    :param isSentinel: True, if path is for Sentinel-2 image; False, of loading MODIS image
    :return: loadedImages: n RGB array of images loaded from given paths
    """
    loadedImages = []
    if isSentinel:
        for p in paths:
            tifff = TIFF.open(p)
            image = tifff.read_image()
            image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            image = scale_images(image)
            loadedImages.append(image)
    else:
        scaleFactor = 0.0001
        fillValue = 32767
        gamma = 1.4
        minP = 0.0
        maxP = 4095.0

        for p in paths:
            tifff = TIFF.open(p)
            image = tifff.read_image()
            redB = image[:, :, 0] * scaleFactor
            greenB = image[:, :, 3] * scaleFactor
            blueB = image[:, :, 2] * scaleFactor

            rgbImage = np.dstack((redB, greenB, blueB))
            rgbImage[rgbImage == fillValue * scaleFactor] = 0
            rgbImage = exposure.rescale_intensity(rgbImage, in_range=(minP, maxP * scaleFactor))

            rgbStretched = exposure.adjust_gamma(rgbImage, gamma)
            rgbStretched = cv2.resize(rgbStretched, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)
            rgbStretched = scale_images(rgbStretched)

            loadedImages.append(rgbStretched)
    return loadedImages

def convertDateToEpoch(inputDate, delta=5):
    """
    :param inputDate:  date in string format dd-mm-yyyy
    :return: int, epoch timestamp after delta days (>=1)
    """
    dateF = strptime(inputDate + ' 00:00', '%Y%m%d %H:%M')
    epochTime = mktime(dateF)
    return int(epochTime) + (86400 * delta)


def preprocess_inputs(encoder, target_timestamps, geohashes=['9qdb9']):
    """
    :param encoder: instance of label encoder to predict numerical label for given geohash of imputated image
    :param target_timestamps: list of epoch timestamps for imputed images
    :param geohashes: list of geohashes of imputed images (given batch size > 1)
    :return: Numerical encodings of imputed image's geohashes, week of the year and seasonality
    """
    def getWeekOfYear(inpDateEpoch):
        # starts from 0 = Jan 1st week
        t = datetime.datetime.fromtimestamp(inpDateEpoch)
        return int(t.strftime("%W"))

    def get_season(woy):
        if 48 <= woy <= 52 or 0 <= woy <= 8:
            return 1  # WINTER
        elif 9 <= woy <= 21:
            return 2  # SPRING
        elif 22 <= woy <= 34:
            return 3  # SUMMER
        elif 35 <= woy <= 47:
            return 4  # AUTUMN
        else:
            return 0

    target_geo = np.array(encoder.transform([g[:-1] for g in geohashes])) / (32 * 32)  # get geohash info only for length 4
    woys = np.array([getWeekOfYear(target_timestamp) for target_timestamp in target_timestamps])
    target_woy = woys / 53
    target_soy = np.array([get_season(woy) for woy in woys])/4

    return target_geo, target_woy, target_soy


def sentinel2_path_to_image_helper(paths):
    """
    :param paths:
    :return:
    """
    samples = []
    for i in paths:
        single_sample = paths_to_rgb_convertor(i, True)
        samples.append(single_sample)
    return np.array(samples)

def extract_geohash_from_path(paths):
    """
    :param paths: Sentinel2 paths;Shape = (batch_size, no_of_timestamps=3)
    :return: 1d list of geohashes for which imputation is made
    """
    return [i[i.find('9q'):i.find('9q')+5] for i in paths]


def load_model_from_disk():
    """
    :return: model: loaded generator with weights to make imputations
    """
    path_to_model = "/s/" + socket.gethostname() + "/a/nobackup/galileo/mind-the-gap-trained-models/"
    layers = open(path_to_model + "GeneratorModel.json", 'r')
    model_structure = layers.read()
    layers.close()

    model = model_from_json(model_structure)
    model.load_weights(path_to_model + "GeneratorModel.h5")
    return model


def impute_image(sentinel2_paths, modis_path, model, encoder, target_timestamps=None):
    """
    :param sentinel2_paths: paths for loading input Sentinel images;Shape = (batch_size, no_of_timestamps=3)
    :param modis_path: paths for loading input MODIS images;Shape = (batch_size,1)
    :param model: Loaded model for making imputations
    :return: imputed_image: Model generated de-normalized images/imputations
                            Shape = (batch_size,256,256,3)
    """

    sentinel2_imgs = sentinel2_path_to_image_helper(sentinel2_paths)
    modis_imgs = np.array(paths_to_rgb_convertor(modis_path, isSentinel=False))

    geohashes = extract_geohash_from_path([i[0] for i in sentinel2_paths])
    if target_timestamps is None:
        target_timestamps = []
        recent_sentinel = [i[-1] for i in sentinel2_paths]
        for p in recent_sentinel:
            inputDate = p.rfind('_')
            # Assuming prediction is made for 5th day
            target_timestamps.append(convertDateToEpoch(p[inputDate+1:inputDate+9],
                                                        delta=5))

    target_geo, target_woy, target_soy = preprocess_inputs(encoder, target_timestamps, geohashes)
    sentinel2_imgs = tf.cast(sentinel2_imgs, tf.float32)
    modis_imgs = tf.cast(modis_imgs, tf.float32)
    start=time.time()
    imputed_image = unscale_images(model([sentinel2_imgs, tf.cast(target_geo, tf.float32), tf.cast(target_woy, tf.float32),
                                          tf.cast(target_soy, tf.float32), modis_imgs]))
    print("Time = ", time.time()-start)
    imputed_image = tf.cast(imputed_image, tf.uint8)
    return imputed_image


if __name__ == '__main__':
    # Random Example

    # shape=(2,3) if batch_size=2
    sentinel2_paths = [[
        "/s/lattice-176/a/nobackup/galileo/stip-images/laton-ca-20km/Sentinel-2/9q7j2/raw/L1C_T11SKA_A025092_20200411T185241-3.tif",
        "/s/lattice-176/a/nobackup/galileo/stip-images/laton-ca-20km/Sentinel-2/9q7j2/raw/L1C_T11SKA_A025235_20200421T184607-3.tif",
        "/s/lattice-176/a/nobackup/galileo/stip-images/laton-ca-20km/Sentinel-2/9q7j2/raw/L1C_T11SKA_A025378_20200501T184410-3.tif"],

        [
            "/s/lattice-176/a/nobackup/galileo/stip-images/laton-ca-20km/Sentinel-2/9q7j2/raw/L1C_T11SKA_A025092_20200411T185241-3.tif",
            "/s/lattice-176/a/nobackup/galileo/stip-images/laton-ca-20km/Sentinel-2/9q7j2/raw/L1C_T11SKA_A025235_20200421T184607-3.tif",
            "/s/lattice-176/a/nobackup/galileo/stip-images/laton-ca-20km/Sentinel-2/9q7j2/raw/L1C_T11SKA_A025378_20200501T184410-3.tif"]]

    # shape=(2,) if batch_size=2
    modis_path = [
        "/s/lattice-176/a/nobackup/galileo/stip-images/laton-ca-20km/MODIS/9q7j2/split/MCD43A4.A2020045.h08v05.006.2020056012811-1.tif",
        "/s/lattice-176/a/nobackup/galileo/stip-images/laton-ca-20km/MODIS/9q7j2/split/MCD43A4.A2020044.h08v05.006.2020056003627-1.tif"]

    # shape=(2,) if batch_size=2
    target_imputation_timestamp = ['1586350800', '1591621200']
    target_imputation_geohash = ['9qdb9', '9qdb9']

    model = load_model_from_disk()
    model.predict((tf.zeros((1,3,256,256,3)),
                            tf.zeros((1,1)),
                            tf.zeros((1,1)),
                            tf.zeros((1,1)),
                            tf.zeros((1,16,16,3))))
    encoder = get_encoding()

    generated_image = impute_image(sentinel2_paths, modis_path, model, encoder)

