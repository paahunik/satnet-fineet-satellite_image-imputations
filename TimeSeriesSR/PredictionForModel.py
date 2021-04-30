import argparse
import datetime
import os
import tensorflow as tf
import numpy as np
import socket
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json
import TimeSeriesSR_Final.data_loader_helpers as dataloaders
from tensorflow.python.ops import math_ops
import stippy
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.optimizers import Adam
import subprocess
import sys

class PM:
    def __init__(self,timesteps,folderI,trainLoss,includeModis,batch_size=1,album='laton-ca-20km',img_h=256,img_width=256,cloudC=0.2,startT='31-03-2018',endT='01-07-2018'):

        self.img_h = img_h
        self.img_w = img_width
        self.timesteps = timesteps
        self.trainLoss = trainLoss
        self.batch_size = batch_size
        self.host_addr=socket.gethostbyname(socket.gethostname()) + ':15606'
        self.album=album
        self.cloudC = float(cloudC)
        self.pathToModel = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/" + str(folderI)
        self.folderI=folderI
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        self.lstmOp = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.dataloader = dataloaders.DatasetHandling(self.img_w, self.img_h, no_of_timesteps=timesteps, startT=startT,
                                                      endT=endT,album=album,cloud_cov=self.cloudC)

        self.includeModis=includeModis

        # self.albums = albums
    def ssim_loss_custom(self, y_true, y_pred):
        ss = tf.image.ssim(y_true, y_pred, 2.0)
        ss = tf.where(tf.math.is_nan(ss), -K.ones_like(ss), ss)
        return -tf.reduce_mean(ss)

    def custom_loss_mse_ssim(self, y_true, y_pred):
        alpha = 0.84
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        ssimL = self.ssim_loss_custom(y_true, y_pred)
        return ((1- alpha)*mse) + (alpha*ssimL)

    def sample_image(self, imgs_unscaled_train,imgs_unscaled_target,fakeImg,imgs_unscaled_modis,psnr, mse,epoch):
        r, c = 2, 3
        fig, axarr = plt.subplots(r, c, figsize=(15, 12))
        np.vectorize(lambda axarr: axarr.axis('off'))(axarr)
        titles = ['T1', 'T2', 'T3', 'Original HR', 'Predicted ' + str(psnr) + "," + str(mse)]

        for row in range(r - 1):
            for col in range(self.timesteps):
                axarr[row, col].imshow(imgs_unscaled_train[0][col])
                axarr[row, col].set_title(titles[col], fontdict={'fontsize': 15})

        axarr[r - 1, c - 2].imshow(imgs_unscaled_target[-1])
        axarr[r - 1, c - 2].set_title(titles[-2], fontdict={'fontsize': 15})

        # Print Predicted
        axarr[r - 1, c - 1].imshow(fakeImg[-1])
        axarr[r - 1, c - 1].set_title(titles[-1], fontdict={'fontsize': 15})
        if True:
            axarr[r - 1, 0].imshow(imgs_unscaled_modis[-1])
            axarr[r - 1, 0].set_title('MODIS', fontdict={'fontsize': 15})

        # plt.suptitle("Target Sentinel Tile Season: {}, WeekOfYear: {}"
        #              .format(int(targetSOY[-1] * 5), int(targetTimeStamp[-1] * 53)), fontsize=20)

        fig.savefig(self.pathToModel + '/test/' + "%s.png" % (epoch))
        plt.close()

    def clusterbased(self,clusterS=16):
        dicG={}
        basep="/s/chopin/a/grad/paahuni/PycharmProjects/ImageSuperResolution/TimeSeriesSR_Final/"
        if clusterS==8:
            pathN="geohash-clusters-8.txt"
        elif clusterS==16:
            pathN="geohash-clusters-16.txt"
        elif clusterS==24:
            pathN="geohash-clusters-24.txt"
        else:
            return "Wrong number of cluster"
        with open(basep+pathN, 'r') as reader:
            for line in reader.readlines():
                line=line.strip()
                dicG[line.split(" ")[0]] = int(line.split(" ")[1])
        # print(dicG)
        return dicG

    def trainLossForGen(self):
        if self.trainLoss == 1:
            return 'huber'
        elif self.trainLoss == 3:
            return 'logcosh'
        elif self.trainLoss == 4:
            return self.custom_loss_mse_ssim
        elif self.trainLoss == 5:
            return 'mae'
        else:
            return 'mse'

    def scale_images_11(self, imgs):
        """ Returns normalized images between (-1 to 1) pixel value"""
        return imgs / 127.5 - 1

    def unscale_images_11(self, imgs):
        return (imgs + 1.) * 127.5

    def load_model(self):
        json_file = open(self.pathToModel + "/ModelCheckp/GeneratorModel.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        lstm_generator=model_from_json(loaded_model_json)
        lstm_generator.load_weights(self.pathToModel + "/ModelCheckp/GeneratorModel.h5")
        lstm_generator.compile(loss=self.trainLossForGen(), optimizer=self.lstmOp, experimental_run_tf_function=False,
                               metrics=["mae", "accuracy"])
        return lstm_generator

    def performaceAcc(self, targets, predicted):
        a = tf.image.convert_image_dtype(targets, tf.float64)
        b = tf.image.convert_image_dtype(predicted, tf.float64)
        mse = math_ops.reduce_mean(math_ops.squared_difference(a, b), [-3, -2, -1])
        psnr = tf.image.psnr(targets,predicted,255.)
        return mse.numpy(),psnr

    def mseOnly(self,targets,predicted):
        a = tf.image.convert_image_dtype(targets, tf.float64)
        b = tf.image.convert_image_dtype(predicted, tf.float64)
        mse = math_ops.reduce_mean(math_ops.squared_difference(a, b), [-3, -2,-1])
        return mse,None

    def saveMetrics(self, name, metric):
        with open(name, 'wb') as filehandle:
            pickle.dump(metric, filehandle)

    def perform_testing_model(self):
            # timeT=[]
            self.lstm_generator = self.load_model()
            imgs_iterTest = self.dataloader.get_non_random_image_iterator_testing(batch_size=self.batch_size,
                                                                         no_of_timesteps=self.timesteps,
                                                                         sendMetaInfo=True,
                                                                         includeModis=self.includeModis)
            psnrs,mses,count=[],[],0
            redMSE,blueMS,greenMS =[],[],[]
            while True:
                # print("Epoch : ", count)
                try:
                    count+=1

                    imgs, target, targetSOY, targetGeo, targetTimeStamp, modisT, _= next(imgs_iterTest)
                    # print("images = ", imgs)
                    imgs = tf.cast(imgs, tf.float32)
                    modisT = tf.cast(modisT, tf.float32)
                    # start = datetime.datetime.now()
                    fakeImg = self.unscale_images_11(
                        self.lstm_generator([imgs, tf.cast(targetGeo, tf.float32), tf.cast(targetTimeStamp, tf.float32),
                               tf.cast(targetSOY, tf.float32), modisT]))

                    # end = datetime.datetime.now()
                    # timeT.append(((end-start).microseconds)/self.batch_size)
                    fakeImg = tf.cast(fakeImg, tf.uint8)
                    # print("Fake image: ", fakeImg)
                    # fakeImg = (self.dataloader.unscale_images_11(self.lstm_generator.predict([imgs, targetGeo, targetTimeStamp, targetSOY, modisT]))).astype(np.uint8)
                    imgs_unscaled_target = self.dataloader.unscale_images_11(target).astype(np.uint8)
                    iC,ps = self.performaceAcc(imgs_unscaled_target, fakeImg)
                    psnrs = psnrs + list(K.eval(ps))
                    mses = mses + list(K.eval(iC))
                    # redM,_ = self.mseOnly(imgs_unscaled_targev[:,:,:,0], fakeImg[:,:,:,0])
                    # blueM,_ = self.mseOnly(imgs_unscaled_target[:,:,:,1], fakeImg[:,:,:,1])
                    # greenM, _ = self.mseOnly(imgs_unscaled_target[:,:,:,2], fakeImg[:,:,:,2])
                    # print("RedM: ", redMSE)
                    self.sample_image(tf.cast(self.unscale_images_11(imgs),tf.uint8),imgs_unscaled_target,fakeImg,self.unscale_images_11(modisT),K.eval(ps),K.eval(iC),count)

                    # redMSE.append(K.eval(redM))
                    # blueMS.append(K.eval(blueM))
                    # greenMS.append(K.eval(greenM))
                    # print("Predicting," ,count)
                except StopIteration:
                    break
            # print("Time: ", timeT)
            # print("Time taken for inference : microseconds", np.average(np.array(timeT)))
            if psnrs==[]:
                return "None","None","None"
            else:
                # print(greenMS)
                return np.mean(np.array(psnrs)), np.mean(np.array(mses)),np.std(np.array(mses)),len(psnrs)
                # return np.array(redMSE),np.array(blueMS), np.array(greenMS)

    def perform_testing_model_cluster(self):
        self.lstm_generator = self.load_model()
        imgs_iterTest = self.dataloader.get_non_random_image_iterator_testing(batch_size=1,
                                                                              no_of_timesteps=self.timesteps,
                                                                              sendMetaInfo=True,
                                                                              includeModis=self.includeModis)
        cou=0
        psnrs = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,
                     8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0}
        mses=  {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,
                     8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0}
        localC=self.clusterbased(16)
        finalClusCo={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,
                     8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0}
        totalI = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0,
                       8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
        localCn=set()
        while True:
            try:
                imgs, target, targetSOY, targetGeo, targetTimeStamp, modisT,g = next(imgs_iterTest)
                fakeImg = (self.dataloader.unscale_images_11(
                    self.lstm_generator.predict([imgs, targetGeo, targetTimeStamp, targetSOY, modisT]))).astype(
                    np.uint8)
                imgs_unscaled_target = self.dataloader.unscale_images_11(target).astype(np.uint8)
                iC, ps = self.performaceAcc(imgs_unscaled_target, fakeImg)
                clustFound = localC[g]
                psnrs[clustFound] += np.sum(K.eval(ps))
                mses[clustFound] += np.sum(K.eval(iC))
                totalI[clustFound] += 1
                localCn.add(g)
            except StopIteration:
                break
        for g in localCn:
                clustFound = localC[g]
                finalClusCo[clustFound]+=1
        for c in finalClusCo:
            print(c, finalClusCo[c],psnrs[c], mses[c], totalI[c])

    def targetNAIP(self):
        infoNAIP={}
        startT, endT = self.dataloader.getTime('01-01-2018', '01-01-2020')

        listImages = stippy.list_node_images(self.host_addr, album=self.album, geocode='9q', recurse=True,
                                             platform='NAIP', source='filled',
                                             start_timestamp=startT, end_timestamp=endT, min_pixel_coverage=1.0)
        count = 0

        for (node, image) in listImages:
            for p in image.files:
                if p.path.endswith('-0.tif'):
                    g = image.geocode
                    count += 1
                    if g not in infoNAIP:
                        infoNAIP[g] = np.array([[image.timestamp, p.path]], dtype=np.dtype(object))
                    else:
                        paths = infoNAIP.get(g)
                        paths = np.concatenate((paths, np.array([[image.timestamp, p.path]], dtype=np.dtype(object))))
                        infoNAIP[g]=paths
        return infoNAIP

    def runDistributedPredictionClust(self,worker=1):
        start,end=176,219
        countC={}
        psrnC={}
        mseC={}
        imageC={}
        for HOST in range(start,end):
            if HOST==192:
                continue
            bashCmd = ["ssh", "lattice-%s" % str(HOST), "python3",
                       "PycharmProjects/ImageSuperResolution/TimeSeriesSR_Final/PredictionForModel.py","--outputDir " + str(self.folderI),"--timesteps "+ str(self.timesteps),"--worker "+ str(worker) ,"--cloudCov " + str(self.cloudC)]

            process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
            result = process.stdout
            print("Testing on Machine : ", HOST)
            prs = result.readlines()
            for results in prs:
                pr = results.decode("utf-8")
                if pr.split(" ")[0].strip()!='None':
                    cls = int(pr.split(" ")[0].strip())
                    count = int(pr.split(" ")[1].strip())
                    machPsnr=float(pr.split(" ")[2].strip())
                    machMSE=float(pr.split(" ")[3].strip())
                    machI=int(pr.split(" ")[4].strip())
                    if cls in countC:
                        ps = psrnC.get(cls)
                        ps.append(machPsnr)
                        psrnC[cls]=ps
                        ms = mseC.get(cls)
                        ms.append(machMSE)
                        mseC[cls] = ms
                        countC[cls] += count
                        imageC[cls]+=machI
                    else:
                        psrnC[cls] = [machPsnr]
                        mseC[cls] = [machMSE]
                        imageC[cls]=machI
                        countC[cls] = count
            # print(psrnC)
            # print(countC)
        for g in countC:
            if imageC.get(g)<1:
                continue
            print("Geohash Cluster: {} Geohash counts: {} Average PSNR: {} Average MSE: {} CoungG: {}".format(g,countC.get(g),(np.sum(psrnC.get(g))/imageC.get(g)),np.sum(mseC.get(g))/imageC.get(g),imageC.get(g)))

    def runDistributedPrediction(self,worker=1):
        psnrsSum,msesSum,std,counts=0,0,0,0
        start,end=176,177
        redB,blueB,greenB=[],[],[]
        cou=0
        for HOST in range(start,end):
            if HOST==192:
                continue
            bashCmd = ["ssh", "lattice-%s" % str(HOST), "python3",
                       "PycharmProjects/ImageSuperResolution/TimeSeriesSR_Final/PredictionForModel.py","--outputDir " + str(self.folderI),"--timesteps "+ str(self.timesteps),"--worker "+ str(worker) ,"--cloudCov " + str(self.cloudC)]

            process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
            result = process.stdout
            print("Testing on Machine : ", HOST)
            pr = result.readlines()[0].decode("utf-8")

            if pr.split(" ")[0].strip()!='None':
                # psnrsSum+=float(pr.split(" ")[0].strip())
                # msesSum+=float(pr.split(" ")[1].strip())
                # std+=float(pr.split(" ")[2].strip())
                # counts += int(pr.split(" ")[3].strip())
                print("Got: ", pr.strip())
                redB.append(list(pr))
                # blueB = np.array(pr.split(" ")[1])
                # greenB = np.array(pr.split(" ")[2])
                cou +=1
        # print("Final Results ---> DIR: {} CloudCov: {} PSNR: {} MSE: {} stdDev: {} TotalCount: {}".format(str(self.folderI), self.cloudC, str(psnrsSum/float(cou)), str(msesSum/float(cou)),str(std/float(cou)),counts))
        print("Final r: ", redB)




if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        # Change the output directory
        parser.add_argument('--outputDir', type=int, default=26)
        parser.add_argument('--timesteps', type=int, default=3)
        parser.add_argument('--worker', type=int,default=1)
        parser.add_argument('--cloudCov', type=float, default=0.6)
        parser.add_argument('--album', type=str, default='laton-ca-20km')

        args = parser.parse_args()
        #
        predictingModel = PM(timesteps=args.timesteps,folderI=args.outputDir,trainLoss=1,batch_size=1,cloudC=args.cloudCov,
                             album='laton-ca-20km',includeModis=0,img_h=256,img_width=256,startT='01-04-2019',endT='01-10-2020')

        # predictingModel.perform_testing_model_cluster()
        if args.worker:
            predictingModel.perform_testing_model()
            # print(psnrs, mses, std, count)
            # print("Finished")
        else:
             predictingModel.runDistributedPrediction(1)

        # if args.worker:
        #     predictingModel.perform_testing_model_cluster()
        # else:
        #      predictingModel.runDistributedPredictionClust(1)

        # count=0
        # imgs_iterTest = predictingModel.dataloader.get_non_random_image_iterator_new(
        #     batch_size=4,
        #     no_of_timesteps=predictingModel.timesteps,
        #     sendMetaInfo=True,
        #     includeModis=predictingModel.includeModis)
        #
        # while True:
        #     try:
        #         imgs, target, targetSOY, targetGeo, targetTimeStamp, modisT = next(imgs_iterTest)
        #         count +=1
        #         print("vcount =" , count)
        #     except StopIteration:
        #         break

