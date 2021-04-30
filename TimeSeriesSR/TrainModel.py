import pickle
import sys
import TimeSeriesSR_Final.data_loader_helpers as dataloaders
import TimeSeriesSR_Final.model_helpers as models
import numpy as np
import datetime
import argparse
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
from tensorflow.keras.optimizers import Adam
from sewar.full_ref import ssim,uqi,vifp
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import horovod.tensorflow.keras as hvd
import math
from TrajGRU import MyModel

import socket

class TrainModels():
    def __init__(self,timesteps,includeAux,folderI,trainLoss,includeModis,includeVGG,disLoss,cloud_cov=0.4,istransfer=False,img_h=256,img_width=256,startT='01-01-2018',endT='01-05-2019'):

        self.img_h = img_h
        self.img_w = img_width
        self.timesteps = timesteps
        self.includeModis = includeModis
        hvd.init()

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.gen_schedule = ExponentialDecay(
            1e-4 * hvd.size(),
            decay_steps=10000,
            decay_rate=0.1,
            staircase=True
        )

        self.disc_schedule = ExponentialDecay(
            1e-4 * hvd.size() * 5,
            decay_steps=10000,
            decay_rate=0.1,
            staircase=True
        )
        self.istransfer=istransfer
        # self.disOp = hvd.DistributedOptimizer(tf.keras.optimizers.Adam(1e-4 * hvd.size(), 0.5))
        # self.lstmOp = hvd.DistributedOptimizer(Adam(lr=1e-4 * hvd.size(), beta_1=0.9, beta_2=0.999, epsilon=1e-08))
        self.disOp = hvd.DistributedOptimizer(Adam(learning_rate=self.disc_schedule))
        self.lstmOp = hvd.DistributedOptimizer(Adam(learning_rate=self.gen_schedule))

        self.model_helpers = models.LSTM_GAN_MODEL(disOp=self.disOp,lstmOp=self.lstmOp,h=self.img_h,w=self.img_w,timeStep=timesteps,includeAux=includeAux,trainLoss=trainLoss,disLoss=disLoss)

        # print("GOT MODIS======", includeModis)
        if includeVGG and includeModis==0:
            if istransfer:
                self.dataloader = dataloaders.DatasetHandling(self.img_w, self.img_h, no_of_timesteps=timesteps,
                                                              startT=startT, endT=endT, cloud_cov=cloud_cov,album='foco-co-20km')

                self.lstm_gan, self.vgg, self.disciminator, self.lstm_generator = self.model_helpers.lstm_gan_with_vgg_transfer(self.transferLear())
            else:
                self.dataloader = dataloaders.DatasetHandling(self.img_w, self.img_h, no_of_timesteps=timesteps,
                                                              startT=startT, endT=endT, cloud_cov=cloud_cov)

                self.lstm_gan, self.vgg, self.disciminator, self.lstm_generator = self.model_helpers.lstm_gan_with_vgg()
        elif not includeVGG and includeModis==0:
            self.lstm_gan,self.vgg, self.disciminator, self.lstm_generator = self.model_helpers.lstm_gan_no_vgg()
        elif includeModis==1:
            self.lstm_gan, self.vgg, self.disciminator, self.lstm_generator = self.model_helpers.lstm_gan_with_vgg_multi_modis()

        self.dirName = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/" + str(folderI) + "/"
        if not includeModis==2:
            self.img_itr = self.dataloader.get_non_random_image_iterator_new(batch_size=1,
                                                                             no_of_timesteps=self.timesteps,
                                                                             sendMetaInfo=True,
                                                                             includeModis=includeModis)
        else:
            self.dataloader = dataloaders.DatasetHandling(self.img_w, self.img_h, no_of_timesteps=timesteps,
                                                          startT=startT, endT=endT, cloud_cov=cloud_cov)
        self.includeVGG=includeVGG

    def saveMetrics(self, name, metric):
        with open(name, 'wb') as filehandle:
            pickle.dump(metric, filehandle)

    def psnrAndRmse(self, target, ref):
            rmseVU = self.rmseM(self.dataloader.scale_images_11(target), self.dataloader.scale_images_11(ref))
            rmseV = self.rmseM(target, ref)
            return round(20 * math.log10(255. / rmseV), 1), round(rmseVU,5)

    def rmseM(self, image1, image2):
        return np.sqrt(np.mean((image2.astype(np.float64) - image1.astype(np.float64)) ** 2))

    def psnr(self, target, ref):
            rmseV = self.rmseM(self.dataloader.unscale_images_11(target), self.dataloader.unscale_images_11(ref))
            return round(20 * math.log10(255. / rmseV), 1)

    def rmseMAEmse(self, image1, image2):
        image1 = self.dataloader.scale_images_11(image1)
        image2 =self.dataloader.scale_images_11(image2)
        rmse = np.sqrt(np.mean((image2 - image1) ** 2))
        mse = np.mean((image2- image1) ** 2)
        mae = np.mean(np.abs(image1 - image2))
        return rmse, mae, mse

    def saveModel(self,model=None):
        if self.includeModis!=2:
            model_json = self.lstm_generator.to_json()
            self.lstm_generator.save_weights(self.dirName + "ModelCheckp/GeneratorModel.h5")
        else:
            model_json=model.to_json()
            model.save_weights(self.dirName + "ModelCheckp/GeneratorModel.h5")

        with open(self.dirName + "ModelCheckp/GeneratorModel.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5

        print("Saved model to disk")
        return

    def load_model(self):
        json_file = open(self.dirName + "ModelCheckp/GeneratorModel.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.dirName + "ModelCheckp/GeneratorModel.h5")
        print("Loaded model from disk")
        self.lstm_generator = loaded_model
        return loaded_model

    def sample_images(self,epoch, with_Aux=False, is_train=True):
        if is_train:
            output_dir = self.dirName + 'train/'
        else:
            output_dir = self.dirName + 'test/'

        r, c = 2, 3
        imgs, target, targetSOY, targetGeo, targetTimeStamp, modisT= next(self.img_itr)

        fakeImg = (self.dataloader.unscale_images_11(self.lstm_generator.predict([imgs, targetGeo, targetTimeStamp, targetSOY, modisT])[-1])).astype(np.uint8)

        imgs_unscaled_train = np.array([self.dataloader.unscale_images_11(img).astype(np.uint8) for img in imgs])
        imgs_unscaled_target = self.dataloader.unscale_images_11(target[-1]).astype(np.uint8)
        imgs_unscaled_modis = self.dataloader.unscale_images_11(modisT[-1])

        titles = ['T1', 'T2', 'T3','Original HR', 'Predicted']

        fig, axarr = plt.subplots(r,c,  figsize=(15, 12))
        np.vectorize(lambda axarr: axarr.axis('off'))(axarr)
        for row in range(r-1):
            for col in range(self.timesteps):
                axarr[row, col].imshow(imgs_unscaled_train[0][col])
                axarr[row, col].set_title(titles[col] , fontdict={'fontsize':15})

        axarr[r-1, c-2].imshow(imgs_unscaled_target)
        axarr[r-1, c-2].set_title(titles[-2],  fontdict={'fontsize':15})

        #Print Predicted
        axarr[r-1, c-1].imshow(fakeImg)
        axarr[r-1, c-1].set_title(titles[-1],  fontdict={'fontsize':15})
        if with_Aux:
            axarr[r - 1, 0].imshow(imgs_unscaled_modis)
            axarr[r - 1, 0].set_title('MODIS',  fontdict={'fontsize':15})

        plt.suptitle("Target Sentinel Tile Season: {}, WeekOfYear: {}"
                     .format(int(targetSOY[-1] * 5), int(targetTimeStamp[-1] * 53)), fontsize=20)

        fig.savefig(output_dir + "%s.png" % (epoch))
        plt.close()

    def trainOnlyLSTM(self, epochs, batch_size=4, sample_interval=50):
        train_itr= self.dataloader.get_non_random_image_iterator_with_aux(self.dataloader.train_geohash, batch_size=batch_size, no_of_timesteps=self.timesteps)
        # val_itr = self.dataloader.get_random_image_iterator(self.metaInfoTest, batch_size=batch_size, no_of_timesteps=3)
        # self.lstm_generator.fit(train_itr, epochs=epochs, steps_per_epoch=len(self.metaInfoTrain) // batch_size, verbose=1,
        #                          validation_data=val_itr,
        #                          validation_steps=self.data_loader.num_test_samples // batch_size, callbacks = [LambdaCallback(on_epoch_end=self.sample_images())])
        # self.saveModel()

        for epoch in range(epochs):
            (X_train, y_train) = next(train_itr)
            start_time = datetime.datetime.now()
            g_loss = self.lstm_generator.train_on_batch(X_train, y_train)
            print_losses = {"G": []}
            print_losses['G'].append(g_loss)
            g_avg_loss = np.array(print_losses['G']).mean(axis=0)
            timeE = ((datetime.datetime.now() - start_time).microseconds) * 0.001

            finalEpochRes = "\nEpoch {}/{} | Time: {}ms\n>>LSTM Generator: {}".format(
                    epoch, epochs,
                    timeE,
                    ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.lstm_generator.metrics_names, g_avg_loss)]))
            print(finalEpochRes)

            if epoch % sample_interval == 0:
                self.sample_images(epoch, with_Aux=False)
                self.saveModel()

        self.perform_testing_model(epoches=5001, with_Aux=False, sampleTestImage=500)

    def named_logs(self, model, logs):
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

    def trainGAN(self, epochs, batch_size=4, sample_interval=51):
        disciminator_output_shape = list(self.disciminator.output_shape)
        disciminator_output_shape[0] = batch_size
        disciminator_output_shape = tuple(disciminator_output_shape)

        print_losses = {"G": [], "D": []}
        valid = np.ones(disciminator_output_shape)
        fake = np.zeros(disciminator_output_shape)

        verbose=1 if hvd.rank() == 0 else 0

        if not self.istransfer:
            callbacks1 = hvd.callbacks.BroadcastGlobalVariablesCallback(0)
            callbacks1.set_model(self.lstm_gan)
            callbacks3 = hvd.callbacks.BroadcastGlobalVariablesCallback(0)
            callbacks3.set_model(self.lstm_generator)
            callbacks1.on_train_begin()
            callbacks3.on_train_begin()

        callbacks2 = hvd.callbacks.BroadcastGlobalVariablesCallback(0)
        callbacks2.set_model(self.disciminator)
        callbacks2.on_train_begin()


        for epoch in range(epochs):
            img_itr = self.dataloader.get_non_random_image_iterator_new(batch_size=batch_size,
                                                                  no_of_timesteps=self.timesteps,sendMetaInfo=True,includeModis=self.includeModis)
            count=0
            elapsedT = datetime.datetime.now()
            while count<=250:
                try:
                    (X_train, y_train, targetSOY, targetGeo, targetTimeStamp, modisT) = next(img_itr)
                    start_time = datetime.datetime.now()
                    count += 1
                    # print(X_train.shape, y_train.shape, targetSOY.shape, targetGeo.shape, targetTimeStamp.shape, modisT.shape)
                    fake_img = self.lstm_generator.predict([X_train, targetGeo, targetTimeStamp, targetSOY, modisT])

                    d_loss_real = self.disciminator.train_on_batch(y_train, valid)
                    d_loss_fake = self.disciminator.train_on_batch(fake_img, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    if self.includeVGG:
                        image_features = self.vgg.predict(self.model_helpers.preprocess_vgg(y_train))
                        g_loss = self.lstm_gan.train_on_batch([X_train, targetGeo, targetTimeStamp, targetSOY, modisT], [valid, image_features])
                    else:
                        g_loss = self.lstm_gan.train_on_batch([X_train, targetGeo, targetTimeStamp, targetSOY, modisT], [valid, y_train])
                    if verbose:
                        timeTotal = ((datetime.datetime.now() - elapsedT).seconds) / 60
                        timeE = ((datetime.datetime.now() - start_time).microseconds) * 0.001
                        print("\nEpoch {}/{} | Sample: {} |  Total Time Elapsed: {} min | epoch time: {} ms".format(
                                    epoch+1, epochs, count * batch_size, timeTotal, timeE))
                        # print("g_loss diction = ", self.lstm_gan.metrics_names)
                        # print("d_loss diction = ", self.disciminator.metrics_names)
                        # self.lstm_generator.save(self.dirName + "ModelCheckp/")

                except StopIteration:
                    break
            if verbose:
                print_losses['G'].append(g_loss)
                print_losses['D'].append(d_loss)
                self.saveMetrics(self.dirName + "TrainingLoss", print_losses)

            if epoch % sample_interval == 0:
                self.saveModel()
                self.sample_images(str(epoch + 1) + "-" + str(count), with_Aux=False)

    def trainGEN(self, epochs, batch_size = 4, sample_interval = 50):
        print_losses = {"G": []}
        # self.lstm_generator.compile(loss='mse', optimizer=self.lstmOp, experimental_run_tf_function=False, metrics=["mae", "accuracy"])

        self.lstm_generator.compile(optimizer=self.lstmOp, loss='mse', metrics=['mae', 'accuracy'])
        verbose=1 if hvd.rank() == 0 else 0
        callbacks3 = hvd.callbacks.BroadcastGlobalVariablesCallback(0)
        callbacks3.set_model(self.lstm_generator)
        callbacks3.on_train_begin()

        for epoch in range(epochs):
            img_itr = self.dataloader.get_non_random_image_iterator_new(batch_size=batch_size,
                                                                  no_of_timesteps=self.timesteps,sendMetaInfo=True,includeModis=self.includeModis)
            count=0
            elapsedT = datetime.datetime.now()
            while count <= 152:
                try:
                    (X_train, y_train, targetSOY, targetGeo, targetTimeStamp, modisT) = next(img_itr)
                    count += 1
                    start_time = datetime.datetime.now()

                    g_loss = self.lstm_generator.train_on_batch([X_train, targetGeo, targetTimeStamp, targetSOY, modisT], y_train)
                    if verbose:
                        timeTotal = ((datetime.datetime.now() - elapsedT).seconds) / 60
                        timeE = ((datetime.datetime.now() - start_time).microseconds) * 0.001
                        print("\nEpoch {}/{} | Sample: {} |  Total Time Elapsed: {} min | epoch time: {} ms".format(
                                    epoch+1, epochs, count * batch_size, timeTotal, timeE))
                        # print("g_loss diction = ", self.lstm_gan.metrics_names)
                        # print("d_loss diction = ", self.disciminator.metrics_names)
                        print_losses['G'].append(g_loss)
                        self.saveMetrics(self.dirName + "TrainingLoss", print_losses)
                        # print("g_loss= " , g_loss[0])
                    if count % sample_interval == 0 and verbose:
                        # self.lstm_generator.save(self.dirName + "ModelCheckp/")
                        self.sample_images(str(epoch + 1) + "-" + str(count), with_Aux=False)
                    self.saveModel()
                except StopIteration:
                    break

    def trainDF(self, epochs, batch_size):
        img_itr = self.dataloader.get_non_random_image_iterator_new(batch_size=batch_size,
                                                                         no_of_timesteps=self.timesteps, sendMetaInfo=False,includeModis=2)
        #
        #
        # img_valid = self.dataloader.get_non_random_image_iterator_with_aux(self.dataloader.test_geohash,
        #                                                                    self.dataloader.metaModisTest,
        #                                                                    batch_size=1,
        #                                                                    no_of_timesteps=self.timesteps, sendMetaInfo=False)
        #
        # callbacks = [tf.keras.callbacks.ModelCheckpoint(self.dirName + 'ModelCheckp/GeneratorModel.h5', monitor='accuracy',
        #                                        mode='max', save_best_only=True)]
        #
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, initial_lr=0.0088, verbose=1),
            # tf.keras.callbacks.ReduceLROnPlateau(patience=30, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(self.dirName + 'ModelCheckp/GeneratorModel.h5')
        ]
        # model = self.model_helpers.train_dfm()
        model = self.model_helpers.ModelJustConv()
        # model = MyModel(3, 1, self.img_h, self.img_w)
        model.compile(loss='mse', optimizer=self.lstmOp, experimental_run_tf_function=False, metrics=["mae", "accuracy"])
        # trainH = model.fit(img_itr, epochs=epochs,
        #                    steps_per_epoch=5707, verbose=1,
        #                    validation_data=None,
        #                    validation_steps=1417, workers=1, max_queue_size=100,
        #                    use_multiprocessing=False, callbacks=callbacks)
        # if hvd.rank() == 0:
        #     callbacks.append(tf.keras.callbacks.ModelCheckpoint(self.dirName + 'ModelCheckp/GeneratorModel.h5'))
        if hvd.rank() == 0 :
           verbose=1
        else:
           verbose=0
        trainH=model.fit(img_itr,epochs=epochs,
                           steps_per_epoch=41, verbose=verbose,
                           validation_data=None,
                             callbacks=callbacks)
        # workers = 1, max_queue_size = 100,
        # use_multiprocessing = False,


        with open(self.dirName + 'TrainingLoss', 'wb') as file_pi:
            pickle.dump(trainH.history, file_pi)

        self.saveModel(model)



    def test_df(self):
        newModel = self.model_helpers.train_dfm()
        # newModel = tf.keras.models.load_model(self.dirName + "ModelCheckp/GeneratorModel.h5")
        newModel.load_weights(self.dirName + "ModelCheckp/GeneratorModel.h5")

        newModel.compile(loss='mse', optimizer=self.disOp, experimental_run_tf_function=False,
                         metrics=['mse', 'mae', 'accuracy'])


        img_test = self.dataloader.get_non_random_image_iterator_testing(
                                                                          batch_size=1,
                                                                          no_of_timesteps=self.timesteps,
                                                                          sendMetaInfo=False, includeModis=2)

        newModel.evaluate(img_test, steps=20, verbose=1)

        epoch = 0
        r, c = 2, max(2, self.timesteps)
        output_dir = self.dirName + 'test/'
        # mymodel.load_weights(self.dirName + 'ModelCheckp/')
        rgbMetric = []
        for x, y in img_test:
            if epoch == 1418:
                break
            # x_com = tf.keras.backend.permute_dimensions(x, (1, 0, 2, 3, 4))  # <B,S,H,W,C>-><S,B,H,W,C>

            prediction = newModel.predict(x)

            imgs_unscaled_train = np.array(
                [self.dataloader.unscale_images_11(img).astype(np.uint8) for img in x])
            imgs_unscaled_target = self.dataloader.unscale_images_11(y[-1]).astype(np.uint8)
            predS = self.dataloader.unscale_images_11(np.array(prediction[-1])).astype(np.uint8)

            # Save generated SRGeneratedImages and the high resolution originals

            iC = self.measureM(imgs_unscaled_target, predS)
            rgbMetric.append(iC)

            if (epoch % 100 == 0):
                print("Testing Sample: ", epoch)
                # titles = ['T1 WeekOfYear-', 'T2 WeekOfYear-', 'T3 WeekOfYear-', 'Original HR',
                #           'Predicted PSNR, RMSE: ' + str(self.psnrAndRmse(imgs_unscaled_target, fakeImg))]
                titles = ['T1', 'T2 ', 'T3', 'Original HR',
                          'Predicted PSNR, RMSE -' + str(self.psnrAndRmse(imgs_unscaled_target, predS))]

                fig, axarr = plt.subplots(r, c, figsize=(15, 12))
                for row in range(r - 1):
                    for col in range(self.timesteps):
                        axarr[row, col].imshow(imgs_unscaled_train[0][col])
                        axarr[row, col].set_title(titles[col], fontdict={'fontsize': 15})
                        axarr[row, col].axis('off')
                # axarr[0, c - 1].axis('off')
                # axarr[0, c-1].axis('off')
                axarr[r - 1, c - 2].imshow(imgs_unscaled_target)
                axarr[r - 1, c - 2].set_title(titles[-2], fontdict={'fontsize': 15})
                axarr[r - 1, c - 2].axis('off')

                # Print Predicted

                axarr[r - 1, c - 1].imshow(predS)
                axarr[r - 1, c - 1].set_title(titles[-1], fontdict={'fontsize': 15})
                axarr[r - 1, c - 1].axis('off')

                for i in range(self.timesteps - 2):
                    axarr[r - 1, i].axis('off')

                # plt.suptitle("Target Sentinel Tile Geohash: {}, Season: {}, WeekOfYear: {}"
                #              .format(g, int(targetSOY[-1] * 5), int(targetTimeStamp[-1] * 53)), fontsize=20)

                fig.savefig(output_dir + "%d.png" % (epoch))
                plt.close()
            epoch += 1

        self.saveMetrics(output_dir + 'imageBMetrics.data', rgbMetric)
        self.loadMetrics(None)

        img_test = self.dataloader.get_non_random_image_iterator_with_aux(self.dataloader.test_geohash,
                                                                          self.dataloader.metaModisTest,
                                                                          batch_size=1,
                                                                          no_of_timesteps=self.timesteps,
                                                                          sendMetaInfo=False, includeModis=False)

        newModel.evaluate(img_test, steps=1418, verbose=1)
        return


    def transferLear(self):
            json_file = open("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/26/ModelCheckp/GeneratorModel.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            lstm_generator = model_from_json(loaded_model_json)
            lstm_generator.load_weights("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/26/ModelCheckp/GeneratorModel.h5")
            # lstm_generator.compile(loss=self.trainLossForGen(), optimizer=self.lstmOp,
            #                        experimental_run_tf_function=False,
            #                        metrics=["mae", "accuracy"])
            return lstm_generator

    def trainTraj(self, epochs, batch_size):
        mymodel = MyModel(3, 1, self.img_h, self.img_w)
        loss_fn = tf.keras.losses.MeanSquaredError()
        losses = []
        # maes = []
        # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=.0002, decay_steps=2000, decay_rate=.96)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        for epoch in range(epochs):
            i = 0
            try:
                img_itr = self.dataloader.get_non_random_image_iterator_testing(batch_size=batch_size,
                                                                            no_of_timesteps=self.timesteps,
                                                                            sendMetaInfo=False, includeModis=2)
                for x, y in img_itr:
                    i += 1
                    with tf.GradientTape() as tape:
                        print("Input shape: ", x.shape)
                        print("Type: ", type(x))
                        x = tf.keras.backend.permute_dimensions(x,
                                                                    (1, 0, 2, 3,
                                                                     4))  # <B,S,H,W,C>-><S,B,H,W,C>
                        prediction = mymodel(x)
                        print("Prediction shape: ", prediction.shape)
                        print("Y shape: ", y.shape)
                        lossMSE = loss_fn(prediction, y)
                        # mae = tf.keras.losses.mean_absolute_error(prediction, y)
                        # psnr = self.psnr(np.array(prediction), np.array(y))
                    gradients = tape.gradient(lossMSE, mymodel.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, mymodel.trainable_variables))
                print("Epoch: {}/{} Batch size: {} MSE: {}".format(epoch+1, epochs, i, lossMSE.numpy()))
                losses.append(lossMSE.numpy())
                self.saveModel(mymodel)
                # mymodel.save_weights(self.dirName + "ModelCheckp/")
                # self.saveMetrics("MSE Training", losses)
                with open(self.dirName + 'TrainingLoss', 'wb') as file_pi:
                    pickle.dump(losses, file_pi)
                # self.saveMetrics("mae training", maes)
            except StopIteration:
                continue

        #
        # epoch = 0
        # r,c = 2, max(3, self.timesteps)
        # output_dir = self.dirName + 'test/'
        # mymodel.load_weights(self.dirName + 'ModelCheckp/')
        # rgbMetric =[]
        # for x, y in img_valid:
        #
        #         x_com = tf.keras.backend.permute_dimensions(x,(1, 0, 2, 3, 4))  # <B,S,H,W,C>-><S,B,H,W,C>
        #
        #         prediction = mymodel(x_com)
        #
        #         imgs_unscaled_train = np.array(
        #             [self.dataloader.unscale_images_11(img).astype(np.uint8) for img in x])
        #         imgs_unscaled_target = self.dataloader.unscale_images_11(y[-1]).astype(np.uint8)
        #         predS = self.dataloader.unscale_images_11(np.array(prediction[-1])).astype(np.uint8)
        #
        #
        #         # Save generated SRGeneratedImages and the high resolution originals
        #
        #         iC = self.measureM(imgs_unscaled_target, predS)
        #         rgbMetric.append(iC)
        #
        #         if (epoch % sampleTestImage == 0):
        #             print("Testing Sample: ", epoch)
        #             # titles = ['T1 WeekOfYear-', 'T2 WeekOfYear-', 'T3 WeekOfYear-', 'Original HR',
        #             #           'Predicted PSNR, RMSE: ' + str(self.psnrAndRmse(imgs_unscaled_target, fakeImg))]
        #             titles = ['T1', 'T2 ', 'T3', 'Original HR',
        #                       'Predicted PSNR, RMSE -' + str(self.psnrAndRmse(imgs_unscaled_target, predS) )]
        #
        #             fig, axarr = plt.subplots(r, c, figsize=(15, 12))
        #             for row in range(r - 1):
        #                 for col in range(self.timesteps):
        #                     axarr[row, col].imshow(imgs_unscaled_train[0][col])
        #                     axarr[row, col].set_title(titles[col] , fontdict={'fontsize': 15})
        #                     axarr[row, col].axis('off')
        #             # axarr[0, c - 1].axis('off')
        #             # axarr[0, c-1].axis('off')
        #             axarr[r - 1, c - 2].imshow(imgs_unscaled_target)
        #             axarr[r - 1, c - 2].set_title(titles[-2], fontdict={'fontsize': 15})
        #             axarr[r - 1, c - 2].axis('off')
        #
        #             # Print Predicted
        #
        #
        #             axarr[r - 1, c - 1].imshow(predS)
        #             axarr[r - 1, c - 1].set_title(titles[-1], fontdict={'fontsize': 15})
        #             axarr[r - 1, c - 1].axis('off')
        #
        #             for i in range(self.timesteps - 2):
        #                 axarr[r - 1, i].axis('off')
        #
        #             # plt.suptitle("Target Sentinel Tile Geohash: {}, Season: {}, WeekOfYear: {}"
        #             #              .format(g, int(targetSOY[-1] * 5), int(targetTimeStamp[-1] * 53)), fontsize=20)
        #
        #             fig.savefig(output_dir + "%d.png" % (epoch))
        #             plt.close()
        #         epoch += 1
        #
        # self.saveMetrics(output_dir + 'imageBMetrics.data', rgbMetric)
        # self.loadMetrics(None)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDir', type=int, default=930)
    parser.add_argument('--epochs', type=int, default=144)
    parser.add_argument('--timesteps', type=int, default=3)
    parser.add_argument('--includeAux', type=bool, default=False)
    parser.add_argument('--trainLoss', type=int, default=1)
    parser.add_argument('--disLoss', type=int, default=0)
    parser.add_argument('--includeModis', type=int, default=0)
    parser.add_argument('--includeVGG', type=bool, default=False)
    parser.add_argument('--cloudCov',type=float,default=0.4)
    # parser.add_argument('--onlyGenerator', type=bool, default=True)
    args = parser.parse_args()
    print("Training on Directory ---------> ", args.outputDir)
    train_cl = TrainModels(img_h=256, img_width=256, timesteps=args.timesteps, includeAux=args.includeAux,
                           folderI=args.outputDir, trainLoss=args.trainLoss, includeVGG=args.includeVGG, includeModis=args.includeModis,
                           disLoss=args.disLoss,cloud_cov=args.cloudCov,istransfer=False)
    # if args.onlyGenerator:
    #     train_cl.trainGEN()
    # else:
    # if args.onlyGenerator:
    # train_cl.trainGEN(epochs=args.epochs, batch_size=1, sample_interval=300)
    # else:
    if args.timesteps>4:
        batch_size=3
    else:
        batch_size=4
    train_cl.trainGAN(epochs=args.epochs, batch_size=batch_size, sample_interval=101)
    # train_cl.test_df()
    # train_cl.trainDF(epochs=args.epochs, batch_size=batch_size)

