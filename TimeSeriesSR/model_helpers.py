from sklearn import preprocessing
from tensorflow.keras.layers import Flatten, TimeDistributed, UpSampling2D, RepeatVector,Input,Conv3DTranspose, Conv2D,Bidirectional, Conv2DTranspose, LeakyReLU,Reshape, Dense, Conv3D,\
    ConvLSTM2D, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from sewar.full_ref import msssim

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Lambda
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import plot_model
import TrajGRU, CustomDFN

class LSTM_GAN_MODEL():

    def __init__(self,disOp,lstmOp,timeStep,includeAux,trainLoss,disLoss=0,w=256,h=256):
        self.targetH = h
        self.targetW = w
        self.targetShape = (self.targetW, self.targetH, 3)
        self.inputShape = (timeStep, self.targetW, self.targetH, 3)
        self.no_of_timesteps = timeStep
        self.baseModDFN = CustomDFN.CustomDFN(filter_size=(6, 6), inp_shape=(self.targetH, self.targetW, 3))
        self.max_mod = 256 + 1
        self.includeAux = includeAux
        self.max_geo = (32 * 32 * 32) + 1
        self.max_woy = 53 + 1
        self.max_season = 5 + 1
        self.disOp = disOp
        self.lstmOp = lstmOp
        self.trainLoss = trainLoss
        self.disLoss=disLoss

    def convertGeohToNumber(self, X, col_no = 1):
        le = preprocessing.LabelEncoder()
        X[:, col_no] = le.fit_transform(X[:, col_no])
        return X

    def ssim_loss(self, y_true, y_pred):
            ssims = []
            for c in range(y_pred.shape[2]):
                ssim = tf.reduce_mean(tf.image.ssim(y_true[:,:,c], y_pred[:,:,c], 2.0))
                ssims.append(ssim)
            return tf.reduce_mean(ssims)

    def mssim_loss(self, y_true, y_pred):
        if not tf.is_tensor(y_true):
            y_true = K.variable(y_true)
            y_pred = K.variable(y_pred)
        return tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 2.0))

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def ssim_loss_custom(self, y_true, y_pred):
        ss = tf.image.ssim(y_true, y_pred, 2.0)
        ss = tf.where(tf.math.is_nan(ss), -K.ones_like(ss), ss)
        return -tf.reduce_mean(ss)


    def my_custom_loss_mse_ssim(self, y_true, y_pred):
        alpha = 0.84
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        ssimL = self.ssim_loss_custom(y_true, y_pred)
        return ((1- alpha)*mse) + (alpha*ssimL)

    def trainLossForGen(self):
        if self.trainLoss == 1:
            return 'huber'
        elif self.trainLoss == 3:
            return 'logcosh'
        elif self.trainLoss == 4:
            return self.my_custom_loss_mse_ssim
        elif self.trainLoss == 5:
            return 'mae'
        else:
            return 'mse'

    def calculate_veg_index(self, red_B4=None, nir_B8=None, img=None ,isSentinel=False):
        if not isSentinel:
            red_B4 = img[:0]
            nir_B8 = img[:3]
        vi = (nir_B8 - red_B4) / (nir_B8 + red_B4)
        if np.isinf(vi).any():
            print("Replacing infinite value with NaN")
            vi[np.isinf(vi)] = np.nan
        if np.isnan(vi).any():
            vi = np.ma.masked_invalid(vi)
        return vi

    def preprocess_vgg(self, x):
        """Take a generated scaled image between [-1, 1] and convert back to [0, 255], then feed as input to VGG network"""
        if isinstance(x, np.ndarray):
            return preprocess_input((x + 1) * 127.5)
        else:
            return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x)

    def train_dfm(self):
        inp = Input(shape=self.inputShape)
        hidden_state = np.zeros((1, self.targetH// 2, self.targetW // 2, 128), dtype='float32')
        newM = self.baseModDFN.mymodel(hidden_state, self.targetH, self.targetW, 3)
        # input_frames = tf.unstack(inp, axis=-1)
        for i in range(self.no_of_timesteps):
            input_frame = tf.reshape(inp[0][i], [1, 256, 256, 3])
            prediction, hidden_state = newM(input_frame, hidden_state)
        finModel = Model(inputs=inp, outputs=prediction)
        # finModel.compile(loss='mse', optimizer=self.disOp, experimental_run_tf_function=False, metrics=["mse", "accuracy"])
        return finModel

    def ModelT(self, colourS=3, batch_size=3,height=256,width=256,i2h_kernel=(3,3)):
        trajgru = TrajGRU.TrajGRU(num_filter=colourS, b_h_w=(batch_size, height, width), zoneout=0.0, L=5,
                               i2h_kernel=i2h_kernel, i2h_stride=(1, 1), i2h_pad=(1, 1), h2h_kernel=(5, 5),
                               h2h_dilate=(1, 1),
                               act_type=LeakyReLU)

        trajgru.compile(loss='mse', optimizer=self.disOp, experimental_run_tf_function=False, metrics=["mse", "accuracy"])
        return trajgru


    def getDiscriminatorModel(self):
        def discriminator_block(model, filters, kernel_size, strides=1, bn=True):
            model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(
                model)
            if bn:
                model = BatchNormalization(momentum=0.9)(model)
            model = LeakyReLU(alpha=0.2)(model)
            return model

        inp = Input(shape=self.targetShape)
        model = discriminator_block(inp, 32, 3, 1)
        model = discriminator_block(model, 32, 3, 2)
        # model = discriminator_block(model, 32, 3, 1)
        model = discriminator_block(model, 32, 3, 2)
        # model = discriminator_block(model, 32, 3, 1)
        model = discriminator_block(model, 32, 3, 2)
        # model = discriminator_block(model, 32, 3, 1)
        model = discriminator_block(model, 32, 3, 2)
        model = Flatten()(model)
        model = Dense(1)(model)
        # model = LeakyReLU(alpha=0.2)(model)
        # model = Dense(1, activation='sigmoid')(model)
        # model = Conv2D(1, kernel_size=1, strides=1, padding='same')(model)

        final_dis = Model(inputs=inp, outputs=model, name='Discriminator')
        if self.disLoss==1:
            final_dis.compile(loss=self.wasserstein_loss, optimizer=self.disOp, metrics=['accuracy'])
        else:
            final_dis.compile(loss=tf.losses.BinaryCrossentropy(), optimizer=self.disOp, metrics=['accuracy'])
        return final_dis

    def getVGGModel(self):
        modelN = VGG19(weights="imagenet", input_shape=self.targetShape, include_top=False)
        # modelN.trainable=False
        modelN_out = modelN.get_layer('block3_conv4').output
        for layer in modelN.layers:
            layer.trainable = False
        feature_model = Model(inputs=modelN.input, outputs=modelN_out, name="Image-Features-Extractor")
        return feature_model

    def ModelWithAuxSingleModis2(self):
        inp = Input(shape=self.inputShape,name='Sentinel')
        inp_geo = Input(shape=(1,), name='Geohash')
        inp_woy = Input(shape=(1,), name='WOY')
        inp_season = Input(shape=(1,), name='SeasonOfTheYear')
        modisT = Input(shape=(16, 16, 3), name="ModisTile")

        if self.includeAux:
            inp_geo1 = RepeatVector(86)(inp_geo)
            inp_woy1 = RepeatVector(85)(inp_woy)
            inp_season1 = RepeatVector(85)(inp_season)
            merged = Concatenate(axis=1)([inp_season1, inp_woy1, inp_geo1])
            merged = Reshape((16,16,1))(merged)
            merged = Concatenate(axis=-1)([merged, modisT])
            merged = Conv2D(filters=16,kernel_size=(3, 3), strides=(1, 1), padding="same")(merged)
        else:
            merged = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(modisT)
        merged = Conv2DTranspose(filters=1, kernel_size=(3, 3), padding='same', strides=(2, 2))(merged)
        merged = LeakyReLU(0.2)(merged)
        merged = Conv2DTranspose(filters=1, kernel_size=(3, 3), padding='same', strides=(2, 2))(merged)
        merged = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="same")(merged)
        merged = LeakyReLU(0.2)(merged)

        #Here Sentinel-2 is preprocessed
        model = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same"))(inp)
        model = LeakyReLU(0.2)(model)
        model = Bidirectional(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                                          padding='same', return_sequences=True))(model)

        model = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same"))(model)
        model = LeakyReLU(0.2)(model)
        model = Bidirectional(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                                            padding='same', return_sequences=True))(model)

        model = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))(model)
        model = LeakyReLU(0.2)(model)
        model = Bidirectional(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                                            padding='same', return_sequences=False))(model)

        # model = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same"))(model)
        # model = LeakyReLU(0.2)(model)
        # model = Bidirectional(ConvLSTM2D(filters=8, kernel_size=(3, 3),
        #                                    padding='same', return_sequences=False), name='LstmLayer32')(model)
        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")(model)
        model = LeakyReLU(0.2)(model)
        model = BatchNormalization()(model)
        model = Concatenate(axis=-1)([model, merged])


        model = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2))(model)
        # model = Concatenate(axis=-1)([model, layer64])
        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = LeakyReLU(0.2)(model)

        model = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2))(model)
        model = BatchNormalization()(model)
        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = LeakyReLU(0.2)(model)

        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")(model)
        model = LeakyReLU(0.2)(model)

        model = Conv2D(32, kernel_size=(1, 1), strides=(1, 1))(model)
        model = Conv2D(3, kernel_size=(1, 1), strides=(1, 1), activation='tanh')(model)

        genModel = Model(inputs=[inp, inp_geo, inp_woy, inp_season, modisT], outputs=[model], name='Generator')
        # genModel.compile(loss='mse', optimizer=self.lstmOp, experimental_run_tf_function=False,
        #                  metrics=["mse", "accuracy"])
        # plot_model(genModel, to_file='ModelWithAuxSingleModis.png', rankdir='TB', show_shapes=True, show_layer_names=True)
        # print(genModel.summary())
        return genModel

    def ModelWithAuxSingleModis(self):
        inp = Input(shape=self.inputShape,name='Sentinel')
        inp_geo = Input(shape=(1,), name='Geohash')
        inp_woy = Input(shape=(1,), name='WOY')
        inp_season = Input(shape=(1,), name='SeasonOfTheYear')
        modisT = Input(shape=(16, 16, 3), name="ModisTile")

        if self.includeAux:
            inp_geo1 = RepeatVector(86)(inp_geo)
            inp_woy1 = RepeatVector(85)(inp_woy)
            inp_season1 = RepeatVector(85)(inp_season)
            merged = Concatenate(axis=1)([inp_season1, inp_woy1, inp_geo1])
            merged = Reshape((16,16,1))(merged)
            merged = Concatenate(axis=-1)([merged, modisT])
            merged = Conv2D(filters=16,kernel_size=(3, 3), strides=(1, 1), padding="same")(merged)
        else:
            merged = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(modisT)
        merged = Conv2DTranspose(filters=16, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')(merged)

        #Here Sentinel-2 is preprocessed
        model = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))(inp)
        model = LeakyReLU(0.2)(model)
        layer256 = Bidirectional(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                                          padding='same', return_sequences=False), name='LstmLayer256')(model)

        model = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same"))(model)
        model = LeakyReLU(0.2)(model)
        layer128 = Bidirectional(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                                            padding='same', return_sequences=False), name='LstmLayer128')(model)

        model = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same"))(model)
        model = LeakyReLU(0.2)(model)
        layer64 = Bidirectional(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                                            padding='same', return_sequences=False), name='LstmLayer64')(model)

        model = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same"))(model)
        model = LeakyReLU(0.2)(model)
        model = Bidirectional(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                                           padding='same', return_sequences=False), name='LstmLayer32')(model)

        model = BatchNormalization()(model)
        model = Concatenate(axis=-1)([model, merged])
        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = LeakyReLU(0.2)(model)

        model = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')(model)
        model = Concatenate(axis=-1)([model, layer64])
        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = LeakyReLU(0.2)(model)

        model = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')(model)
        model = BatchNormalization()(model)
        model = Concatenate(axis=-1)([model, layer128])
        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = LeakyReLU(0.2)(model)

        model = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')(model)
        model = Concatenate(axis=-1)([model, layer256])
        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = LeakyReLU(0.2)(model)

        model = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='tanh')(model)
        model = Conv2D(16, kernel_size=(1, 1), strides=(1, 1), activation='tanh')(model)
        model = Conv2D(3, kernel_size=(1, 1), strides=(1, 1), activation='tanh')(model)

        genModel = Model(inputs=[inp, inp_geo, inp_woy, inp_season, modisT], outputs=[model], name='Generator')
        # genModel.compile(loss='mse', optimizer=self.lstmOp, experimental_run_tf_function=False,
        #                  metrics=["mse", "accuracy"])
        # plot_model(genModel, to_file='ModelWithAuxSingleModis.png', rankdir='TB', show_shapes=True, show_layer_names=True)
        # print(genModel.summary())
        return genModel

    def ModelJustConv(self):
        inp = Input(shape=self.inputShape,name='Sentinel')
        # inp_geo = Input(shape=(1,), name='Geohash')
        # inp_woy = Input(shape=(1,), name='WOY')
        # inp_season = Input(shape=(1,), name='SeasonOfTheYear')
        # modisT = Input(shape=(16, 16, 3), name="ModisTile")

        #Here Sentinel-2 is preprocessed
        model = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))(inp)
        model = LeakyReLU(0.2)(model)
        model = Bidirectional(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                                          padding='same', return_sequences=True), name='LstmLayer256')(model)

        model = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same"))(model)
        model = LeakyReLU(0.2)(model)
        model = Bidirectional(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                                            padding='same', return_sequences=False), name='LstmLayer128')(model)


        model = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='tanh')(model)
        model = Conv2D(16, kernel_size=(1, 1), strides=(1, 1), activation='tanh')(model)
        model = Conv2D(3, kernel_size=(1, 1), strides=(1, 1), activation='tanh')(model)

        genModel = Model(inputs=inp, outputs=model, name='Generator')
        # genModel.compile(loss='mse', optimizer=self.lstmOp, experimental_run_tf_function=False,
        #                  metrics=["mse", "accuracy"])
        # plot_model(genModel, to_file='ModelWithAuxSingleModis.png', rankdir='TB', show_shapes=True, show_layer_names=True)
        # print(genModel.summary())
        return genModel

    def ModelWithAuxSingleModisSingleSent(self):
        inp1 = Input(shape=self.inputShape,name='Sentinel')
        inp = Reshape((self.targetH, self.targetW, 3))(inp1)
        inp_geo = Input(shape=(1,), name='Geohash')
        inp_woy = Input(shape=(1,), name='WOY')
        inp_season = Input(shape=(1,), name='SeasonOfTheYear')
        modisT = Input(shape=(16, 16, 3), name="ModisTile")

        inp_geo1 = RepeatVector(86)(inp_geo)
        inp_woy1 = RepeatVector(85)(inp_woy)
        inp_season1 = RepeatVector(85)(inp_season)
        merged1 = Concatenate(axis=1)([inp_season1, inp_woy1, inp_geo1])
        merged = Reshape((16,16,1))(merged1)
        mergedM = Concatenate(axis=-1)([merged, modisT])

        mergedM = Conv2DTranspose(filters=16, kernel_size=(3, 3), padding='same', strides=(2, 2),activation=None)(mergedM)
        mergedM = LeakyReLU(0.2)(mergedM)

        mergedM = Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same', strides=(2, 2),activation=None)(mergedM)
        mergedM = LeakyReLU(0.2)(mergedM)

        model = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same")(inp)
        model2 = LeakyReLU(0.2)(model)

        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")(model2)
        model = LeakyReLU(0.2)(model)

        model = Concatenate(axis=-1)([model, mergedM])
        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = LeakyReLU(0.2)(model)

        model = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2),activation=None)(model)
        model = LeakyReLU(0.2)(model)
        model = BatchNormalization()(model)

        model = Concatenate()([model2, model])
        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = LeakyReLU(0.2)(model)

        model = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2),activation=None)(model)
        model = LeakyReLU(0.2)(model)
        # model = Concatenate()([inp, model])
        model = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='tanh')(model)
        model = Conv2D(16, kernel_size=(1, 1), strides=(1, 1), activation='tanh')(model)
        model = Conv2D(3, kernel_size=(1, 1), strides=(1, 1), activation='tanh')(model)

        genModel = Model(inputs=[inp1, inp_geo, inp_woy, inp_season, modisT], outputs=[model], name='Generator')
        # genModel.compile(loss='mse', optimizer=self.lstmOp, experimental_run_tf_function=False,
        #                  metrics=["mse", "accuracy"])
        # plot_model(genModel, to_file='ModelWithAuxSingleModis.png', rankdir='TB', show_shapes=True, show_layer_names=True)
        # print(genModel.summary())
        return genModel

    def ModelSingleSentMultipleModis(self):
        inp1 = Input(shape=self.inputShape, name='Sentinel')
        inp = Reshape((self.targetH, self.targetW, 3))(inp1)
        inp_geo = Input(shape=(1,), name='Geohash')
        inp_woy = Input(shape=(1,), name='WOY')
        inp_season = Input(shape=(1,), name='SeasonOfTheYear')
        modisT = Input(shape=(None, 16, 16, 3), name="ModisTile")

        inp_geo1 = RepeatVector(86)(inp_geo)
        inp_woy1 = RepeatVector(85)(inp_woy)
        inp_season1 = RepeatVector(85)(inp_season)
        merged1 = Concatenate(axis=1)([inp_season1, inp_woy1, inp_geo1])
        merged = Reshape((16, 16, 1))(merged1)

        #Apply LSTM

        model = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu')(modisT)

        model = ConvLSTM2D(filters=64, kernel_size=(3, 3),padding='same', return_sequences=True)(model)
        model = BatchNormalization()(model)
        model = ConvLSTM2D(filters=64, kernel_size=(3, 3),padding='same', return_sequences=True)(model)
        model = ConvLSTM2D(filters=64, kernel_size=(3, 3),padding='same', return_sequences=True)(model)
        model = BatchNormalization()(model)
        model = ConvLSTM2D(filters=64, kernel_size=(3, 3),padding='same', return_sequences=False)(model)
        model = Conv2D(64, kernel_size=3, strides=1, activation='tanh', padding='same')(model)

        mergedM = Concatenate(axis=-1)([merged, model])
        mergedM = Conv2D(64, kernel_size=9, strides=1, activation='tanh', padding='same')(mergedM)
        mergedM = LeakyReLU(0.2)(mergedM)

        mergedM = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2), activation=None)(mergedM)
        mergedM = LeakyReLU(0.2)(mergedM)

        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")(inp)
        model2 = LeakyReLU(0.2)(model)

        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")(model2)
        model3 = LeakyReLU(0.2)(model)

        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")(model3)
        model = LeakyReLU(0.2)(model)

        model = Concatenate(axis=-1)([model, mergedM])
        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = LeakyReLU(0.2)(model)

        model = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2), activation=None)(model)
        model = LeakyReLU(0.2)(model)
        model = BatchNormalization()(model)

        model = Concatenate()([model3, model])
        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = LeakyReLU(0.2)(model)

        model = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2), activation=None)(model)
        model = LeakyReLU(0.2)(model)
        model = BatchNormalization()(model)

        model = Concatenate()([model2, model])
        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = LeakyReLU(0.2)(model)

        model = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2), activation=None)(model)
        model = LeakyReLU(0.2)(model)
        model = BatchNormalization()(model)

        # model = Concatenate()([inp, model])
        model = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='tanh')(model)
        model = Conv2D(16, kernel_size=(1, 1), strides=(1, 1), activation='tanh')(model)
        model = Conv2D(3, kernel_size=(1, 1), strides=(1, 1), activation='tanh')(model)

        genModel = Model(inputs=[inp1, inp_geo, inp_woy, inp_season, modisT], outputs=[model], name='Generator')
        # genModel.compile(loss='mse', optimizer=self.lstmOp, experimental_run_tf_function=False,
        #                  metrics=["mse", "accuracy"])
        # plot_model(genModel, to_file='ModelWithAuxSingleModis.png', rankdir='TB', show_shapes=True, show_layer_names=True)
        # print(genModel.summary())
        return genModel

    def lstm_gan_with_vgg_transfer(self,lstmGen):
        inputLayer = lstmGen.input
        target_img = lstmGen.output
        vgg = self.getVGGModel()
        target_vgg_features = vgg(self.preprocess_vgg(target_img))
        discriminator = self.getDiscriminatorModel()
        if self.disLoss == 1:
            l = self.wasserstein_loss
        else:
            l = 'binary_crossentropy'
        discriminator.compile(loss=l, optimizer=self.disOp, metrics=['accuracy'],
                                  experimental_run_tf_function=False)
        for layer in discriminator.layers:
            layer.trainable = False
        adversial_network = discriminator(target_img)
        gan_model = Model(inputs=[inputLayer], outputs=[adversial_network, target_vgg_features], name='LSTM-GAN-WITH-VGG')
        gan_model.compile(loss=[l, self.trainLossForGen()], loss_weights=[0.001, 1], optimizer=self.lstmOp, experimental_run_tf_function=False, metrics=["mae", "accuracy"])
        return gan_model, vgg, discriminator, lstmGen

    def lstm_gan_with_vgg(self):
        if self.no_of_timesteps > 1:
            lstmGen = self.ModelWithAuxSingleModis()
        else:
            lstmGen = self.ModelWithAuxSingleModisSingleSent()
        inputLayer = lstmGen.input
        target_img = lstmGen.output
        vgg = self.getVGGModel()
        target_vgg_features = vgg(self.preprocess_vgg(target_img))
        discriminator = self.getDiscriminatorModel()
        if self.disLoss == 1:
            l = self.wasserstein_loss
        else:
            l = 'binary_crossentropy'
        discriminator.compile(loss=l, optimizer=self.disOp, metrics=['accuracy'],
                                  experimental_run_tf_function=False)
        for layer in discriminator.layers:
            layer.trainable = False
        adversial_network = discriminator(target_img)
        gan_model = Model(inputs=[inputLayer], outputs=[adversial_network, target_vgg_features], name='LSTM-GAN-WITH-VGG')
        gan_model.compile(loss=[l, self.trainLossForGen()], loss_weights=[0.001, 1], optimizer=self.lstmOp, experimental_run_tf_function=False, metrics=["mae", "accuracy"])
        return gan_model, vgg, discriminator, lstmGen

    def lstm_gan_with_vgg_multi_modis(self):
        lstmGen = self.ModelSingleSentMultipleModis()
        inputLayer = lstmGen.input
        target_img = lstmGen.output
        vgg = self.getVGGModel()
        target_vgg_features = vgg(self.preprocess_vgg(target_img))
        discriminator = self.getDiscriminatorModel()
        discriminator.compile(loss='binary_crossentropy', optimizer=self.disOp, metrics=['accuracy'])
        for layer in discriminator.layers:
            layer.trainable = False
        adversial_network = discriminator(target_img)
        gan_model = Model(inputs=[inputLayer], outputs=[adversial_network, target_vgg_features], name='LSTM-GAN-WITH-VGG')
        gan_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[0.001, 1], optimizer=self.lstmOp, experimental_run_tf_function=False, metrics=["mae", "accuracy"])
        return gan_model, vgg, discriminator, lstmGen

    def lstm_gan_no_vgg(self):
        lstmGen = self.ModelWithAuxSingleModis()
        inputLayer = lstmGen.input
        target_img = lstmGen.output
        discriminator = self.getDiscriminatorModel()
        if self.disLoss==1:
            l = self.wasserstein_loss
        else:
            l = 'binary_crossentropy'
        discriminator.compile(loss=l, optimizer=self.disOp, metrics=['accuracy'],
                              experimental_run_tf_function=False)
        for layer in discriminator.layers:
            layer.trainable = False
        adversial_network = discriminator(target_img)
        gan_model = Model(inputs=[inputLayer], outputs=[adversial_network, target_img],name='LSTM-GAN-NO-VGG-NO-AUX-WITH-DIS')

        gan_model.compile(loss=[l, self.trainLossForGen()], optimizer=self.lstmOp, loss_weights=[0.001, 1], experimental_run_tf_function=False, metrics=["mae", "accuracy"])
        return gan_model, None, discriminator, lstmGen

    def get_model_memory_usage(self,batch_size, model):



        shapes_mem_count = 0
        internal_model_mem_count = 0
        for l in model.layers:
            layer_type = l.__class__.__name__
            if layer_type == 'Model':
                internal_model_mem_count += self.get_model_memory_usage(batch_size, l)
            single_layer_mem = 1
            out_shape = l.output_shape
            if type(out_shape) is list:
                out_shape = out_shape[0]
            for s in out_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

        number_size = 4.0
        if K.floatx() == 'float16':
            number_size = 2.0
        if K.floatx() == 'float64':
            number_size = 8.0

        total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
        return gbytes

if __name__ == '__main__':
    train_cl = LSTM_GAN_MODEL(Adam(0.0002, 0.5), Adam(0.0002, 0.5),includeAux=True,trainLoss=2 ,disLoss=1,w=128, h=128, timeStep=2)
    model = train_cl.ModelWithAuxSingleModis()
    train_cl.get_model_memory_usage(6,model)


