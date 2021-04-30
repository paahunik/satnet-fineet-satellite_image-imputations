import random
from libtiff import TIFF
import libtiff
import matplotlib.image as matImg
import stippy
from time import strptime, mktime
import datetime
from sklearn.preprocessing import LabelEncoder
import socket
import numpy as np
import sys
from skimage import exposure
import cv2
libtiff.libtiff_ctypes.suppress_warnings()
np.set_printoptions(threshold=sys.maxsize)

class DatasetHandling():

    def __init__(self, w=256, h=256, no_of_timesteps=2, geoh='9q', platf=['Sentinel-2'],
                 recurse=True, pix_cover=1.0, startT='01-01-2018', endT='30-12-2020',
                 cloud_cov=0.2,album='laton-ca-20km'):
        self.targetH = h
        self.targetW = w
        self.max_geo = 32 * 32
        self.max_woy = 53
        self.max_season = 5
        self.startT = startT
        self.endT=endT
        self.pix_cover=pix_cover
        self.cloud_cov=cloud_cov
        self.recurse=recurse
        self.no_of_timesteps = no_of_timesteps
        self.platf = platf
        self.album = album
        self.geoh = geoh
        self.host_addr=socket.gethostbyname(socket.gethostname()) + ':15606'
        self.train_geohashes=self.load_images_path()
        self.metaModis = self.getModisTileAtGivenTime()
        self.len_train_geohash = len(list(self.train_geohashes))

        # print("No. of training geohashes {} ".format(self.len_train_geohash))
        self.geoEncodings = self.getEncoding(list(map(self.removeLastG, list(self.train_geohashes))))
        # count = 0
        # for g in self.train_geohashes:
        #     print("For {} found {} images ".format(g, len(self.train_geohashes.get(g))))
        #     count += len(self.train_geohashes.get(g))
        # print("Total training samples: ", count)

    def convertDateToEpoch(self, inputDate):
        dateF = strptime(inputDate + ' 00:00', '%d-%m-%Y %H:%M')
        epochTime=mktime(dateF)
        return int(epochTime)

    def removeLastG(self,g):
        return g[:-1]

    def convertEpochToDate(self, inputEpoch):
        return datetime.datetime.fromtimestamp(inputEpoch).strftime('%Y-%m-%d')

    def getTime(self, startT, endT):
        if startT is not None:
            startT = self.convertDateToEpoch(startT)
        if endT is not None:
            endT = self.convertDateToEpoch(endT)

        return startT, endT

    def scale_images_11(self, imgs):
        """ Returns normalized images between (-1 to 1) pixel value"""
        return imgs/127.5 - 1

    def unscale_images_11(self, imgs):
        return (imgs + 1.) * 127.5

    def accessFiles(self, paths, isSentinel=True):
        loadedImages = []
        if isSentinel:
            for p in paths:
                loaded_image = TIFF.open(p)
                image = loaded_image.read_image()
                image = cv2.resize(image, dsize=(self.targetW, self.targetW), interpolation=cv2.INTER_CUBIC)
                image = self.scale_images_11(image)

                loadedImages.append(image)
        else:
            scaleFactor = 0.0001
            fillValue = 32767
            gamma = 1.4
            minP = 0.0
            maxP = 4095.0

            for p in paths:
                loaded_image = TIFF.open(p)
                image = loaded_image.read_image()
                redB = image[:, :, 0] * scaleFactor
                greenB = image[:, :, 3] * scaleFactor
                blueB = image[:, :, 2] * scaleFactor

                rgbImage = np.dstack((redB, greenB, blueB))
                rgbImage[rgbImage == fillValue * scaleFactor] = 0
                rgbImage = exposure.rescale_intensity(rgbImage, in_range=(minP, maxP * scaleFactor))

                rgbStretched = exposure.adjust_gamma(rgbImage, gamma)
                rgbStretched = cv2.resize(rgbStretched, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)
                rgbStretched = self.scale_images_11(rgbStretched)
                loadedImages.append(rgbStretched)
        # print("loaded image shape: ", np.array(loadedImages).shape)
        # print(np.array(loadedImages).dtype)
        return loadedImages


    def getWeekOfYear(self, inpDateEpoch):
        #starts from 0 = Jan 1st week
        t = datetime.datetime.fromtimestamp(inpDateEpoch)
        return int(t.strftime("%W"))

    def get_season(self, woy):
        # starting month = 1
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

    def getSentiT(self, timeS):
        for i in timeS:
            yield i

    def getModisBetweenTwoSentinel(self,geo,startT,endT):
        metaModis=[]
        listImages = stippy.list_node_images(self.host_addr, album=self.album, geocode=geo, recurse=False,
                                             platform='MODIS',
                                             start_timestamp=startT, end_timestamp=endT)
        count = 0

        for (node, image) in listImages:
            for p in image.files:
                if p.path.endswith('-1.tif'):
                    metaModis.append(p.path)
                    count += 1
            if count > 5:
                break
        return self.accessFiles(metaModis, isSentinel=False)

    def getModisTileAtGivenTime(self):
        startT, endT = self.getTime(self.startT, self.endT)
        metaModis = {}

        for g in list(self.train_geohashes):

            listImages = stippy.list_node_images(self.host_addr,album=self.album, geocode=g, recurse=False,
                                            platform='MODIS',
                                            start_timestamp=startT, end_timestamp=endT)
            itr_time = self.getSentiT(self.train_geohashes.get(g)[:, 0])
            try:
                (node, b_image) = listImages.__next__()
                t = next(itr_time)
                while True:
                    if (abs(t - b_image.timestamp) < 86400) and g == b_image.geocode:
                        for p in b_image.files:
                            if p.path.endswith('-1.tif'):
                                   pathf = p.path
                                   metaModis[g + "_" + str(t)] = pathf
                            (_, b_image) = listImages.__next__()
                            t = next(itr_time)
                    elif t < b_image.timestamp:
                        t = next(itr_time)
                    elif t > b_image.timestamp:
                        (_, b_image) = listImages.__next__()
            except StopIteration:
                pass
        return metaModis



    def getEncoding(self, geohashes):
        le = LabelEncoder()
        return le.fit(geohashes)

    def get_non_random_image_iterator_testing(self, batch_size=1, no_of_timesteps=2, sendMetaInfo=True, includeModis=0):
                '''
                    :param includeModis:
                    0 = Only Target Modis
                    1 = Every day since sentinel prev and target date
                    2 = no modis
                '''
                targets, modisI, targetGeo, targetTimeStamp, targetSOY, samples = [], [], [], [], [], []
                for g in list(self.train_geohashes):
                        paths = self.train_geohashes.get(g)[:, 1]
                        timestamps = self.train_geohashes.get(g)[:, 0]
                        batch = self.accessFiles(paths.tolist(),True)  # Return non-cloudy normalized images for given geohash
                        # cloudcover=self.train_geohashes.get(g)[:, -1]
                        for i in range(0, len(batch) - no_of_timesteps):
                            sample = batch[i:i + no_of_timesteps]
                            targetTime = timestamps[i + no_of_timesteps]
                            inputTimeStamp = timestamps[i: i + no_of_timesteps]

                            if (targetTime - inputTimeStamp[0]) > 86400 * self.no_of_timesteps * 8:    #look only for past month images
                                continue

                            if includeModis == 0 and (g + "_" + str(targetTime) not in self.metaModis):
                                continue     # no modis tile found
                            elif includeModis == 1:
                                modisI = self.getModisBetweenTwoSentinel(g, inputTimeStamp[-1], targetTime - 86400)
                                if len(np.array(modisI).shape) < 3:
                                    continue # no modis tile found
                                modisI = np.expand_dims(modisI, axis=0)

                            samples.append(sample)
                            targets.append(batch[i + no_of_timesteps])

                            encoding = self.geoEncodings.transform([g[:-1]])[-1]
                            encoding = encoding / self.max_geo
                            targetGeo.append(encoding)
                            woy = self.getWeekOfYear(targetTime)

                            targetSOY.append(self.get_season(woy) / self.max_season)
                            woy = woy / self.max_woy
                            targetTimeStamp.append(woy)

                            if includeModis == 0:
                                path = self.metaModis.get(g + "_" + str(targetTime))
                                modisI.append(self.accessFiles([path], isSentinel=False))

                            if batch_size == len(targets):
                                if sendMetaInfo and includeModis != 2:
                                    if includeModis == 0:
                                        modisI = np.squeeze(np.array(modisI), axis=1)
                                    yield np.array(samples), np.array(targets), np.array(targetSOY), np.array(
                                        targetGeo), np.array(targetTimeStamp), np.array(modisI),g

                                elif not sendMetaInfo and includeModis == 2:
                                    yield np.array(samples), np.array(targets)

                                else:
                                    modisI = np.squeeze(np.array(modisI), axis=1)
                                    yield np.array(samples), np.array(targets), np.array(modisI)

                                samples, targets, targetGeo, targetTimeStamp, targetSOY, modisI = [], [], [], [], [], []

    '''
                       :param includeModis:
                       0 = Only Target Modis
                       1 = Every day since sentinel prev and target date
                       2 = no modis
    '''
    def get_non_random_image_iterator_new(self, batch_size=4, no_of_timesteps=2, sendMetaInfo=False, includeModis=0):
            targets, modisI, targetGeo, targetTimeStamp, targetSOY,samples = [], [], [], [], [], []
            while True:
                for g in list(self.train_geohashes):

                    paths = self.train_geohashes.get(g)[:, 1]
                    timestamps = self.train_geohashes.get(g)[:, 0]
                    batch = self.accessFiles(paths.tolist(),True)  # Return non-cloudy normalized images for given geohash
                    # batch contains all the images between given timestamp for a given geohash
                    for i in range(0, len(batch)-no_of_timesteps,no_of_timesteps+1):
                        sample = batch[i:i + no_of_timesteps]
                        targetTime=timestamps[i + no_of_timesteps]
                        inputTimeStamp = timestamps[i: i + no_of_timesteps]

                        if (targetTime - inputTimeStamp[0]) > 86400 * self.no_of_timesteps * 8:
                            continue

                        if includeModis == 0 and (g + "_" + str(targetTime) not in self.metaModis):
                            continue
                        elif includeModis == 1:
                            modisI = self.getModisBetweenTwoSentinel(g, inputTimeStamp[-1],targetTime-86400)
                            if len(np.array(modisI).shape) < 3:
                                continue
                            modisI = np.expand_dims(modisI,axis=0)

                        samples.append(sample)
                        targets.append(batch[i + no_of_timesteps])

                        encoding = self.geoEncodings.transform([g[:-1]])[-1]
                        encoding = encoding / self.max_geo
                        targetGeo.append(encoding)
                        woy=self.getWeekOfYear(targetTime)

                        targetSOY.append(self.get_season(woy) / self.max_season)
                        woy = woy / self.max_woy
                        targetTimeStamp.append(woy)

                        if includeModis == 0:
                            path = self.metaModis.get(g + "_" + str(targetTime))
                            modisI.append(self.accessFiles([path], isSentinel=False))

                        if (batch_size == len(targets)):
                            if sendMetaInfo and includeModis != 2:
                                if includeModis == 0:
                                    modisI = np.squeeze(np.array(modisI), axis=1)
                                yield np.array(samples), np.array(targets), np.array(targetSOY), np.array(targetGeo), np.array(
                                    targetTimeStamp), np.array(modisI)
                            elif (not sendMetaInfo and includeModis == 2):
                                yield np.array(samples), np.array(targets)
                            else:
                                modisI = np.squeeze(np.array(modisI), axis=1)
                                yield np.array(samples), np.array(targets), np.array(modisI)
                            samples, targets, targetGeo, targetTimeStamp, targetSOY, modisI=[],[],[],[],[],[]


if __name__ == '__main__':
    dataH = DatasetHandling(256, 256, startT='01-01-2018', platf=['Sentinel-2'], recurse=True, endT='01-06-2020', cloud_cov = 0.2, album='laton-ca-20km')
    batch_size = 8
    it = dataH.get_non_random_image_iterator_testing(batch_size, no_of_timesteps=2, sendMetaInfo=True, includeModis=0)    #Model 1
    # countI = 0
    # while True:
    #     try:
    #         samples, targets, targetSOY,targetGeo, targetTimeStamp, modisI = it.__next__()
    #         countI += batch_size
    #     except StopIteration:
    #         break
    # print("Total images found: {} ".format(countI))