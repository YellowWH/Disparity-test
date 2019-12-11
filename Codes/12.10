from __future__ import print_function
import PIL.Image as Image
from requests import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
np.random.seed(1337)
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Concatenate, Conv2D, Reshape, AveragePooling2D, Add, Flatten, Conv1D
from keras.optimizers import Adam, Adadelta, RMSprop
from keras.models import Model
import h5py
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt


# 曲线
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.losses1 = {'batch':[], 'epoch':[]}
        self.losses2 = {'batch':[], 'epoch':[]}
        self.losses3 = {'batch':[], 'epoch':[]}
        # self.accuracy = {'batch':[], 'epoch':[]}
        # self.val_loss = {'batch':[], 'epoch':[]}
        # self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.losses1['batch'].append(logs.get('main_reshape_loss'))
        self.losses2['batch'].append(logs.get('side_layerallsum_loss'))
        self.losses3['batch'].append(logs.get('wavesum_loss'))
        # self.accuracy['batch'].append(logs.get('acc'))
        # self.val_loss['batch'].append(logs.get('val_loss'))
        # self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.losses['epoch'].append(logs.get('loss'))
        self.losses1['epoch'].append(logs.get('main_reshape_loss'))
        self.losses2['epoch'].append(logs.get('side_layerallsum_loss'))
        self.losses3['epoch'].append(logs.get('wavesum_loss'))
        # self.accuracy['epoch'].append(logs.get('acc'))
        # self.val_loss['epoch'].append(logs.get('val_loss'))
        # self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        # plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'r', label='train loss')
        plt.plot(iters, self.losses1[loss_type] * alpha, 'g', label='alpha loss')
        plt.plot(iters, self.losses2[loss_type] * beta, 'b', label='beta loss')
        plt.plot(iters, self.losses3[loss_type] * gamma, 'k', label='gamma loss')
        # if loss_type == 'epoch':
        #     # val_acc
        #     plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
        #     # val_loss
        #     plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


history = LossHistory()

iml = np.array(Image.open("pooliml.png"))
imr = np.array(Image.open("poolimr.png"))
alpha = 1
beta = 0.02
gamma = 0.005
kaisu = 100
M = 185
N = 223
q = 16
LD = 16
startm = 0
startn = LD
# print(iml.shape)
# (185, 223, 3)
testtarget = np.zeros((kaisu, q, q, 3))
for k in range(kaisu):
    for i in range(q):
        for j in range(q):
            testtarget[k, i, j,] = iml[startm + i, startn + j,]

input_grey = np.ones((kaisu, 1), dtype=np.uint8)
probability = np.ones(LD)
weight = np.zeros((LD, 3))
side_weight = np.zeros((LD, LD))
sum_weight = np.zeros(LD)
test1 = np.zeros(kaisu)
a = np.array([1])
sideConv_weight = np.array([1])

x_list = list()
y_list = list()
z_list = list()

for i in range(0, LD):
    side_weight[i, i] = 1
for i in range(LD):
    sum_weight[i] = i * 8

def pp(x):
    return x-K.square(x)

filterLL = [[1/32, -2/32, -6/32, -2/32, 1/32],
            [-2/32, 4/32, 12/32, 4/32, -2/32],
            [-6/32, 12/32, 36/32, 12/32, -6/32],
            [-2/32, 4/32, 12/32, 4/32, -2/32],
            [1/32, -2/32, -6/32, -2/32, 1/32]]

filterLH = [[1/16, -2/16, -6/16, -2/16, 1/16],
            [-2/16, 4/16, 12/16, 4/16, -2/16],
            [1/16, -2/16, -6/16, -2/16, 1/16]]

filterHL = [[1/16, -2/16, 1/16],
            [-2/16, 4/16, -2/16],
            [-6/16, 12/16, -6/16],
            [-2/16, 4/16, -2/16],
            [1/16, -2/16, 1/16]]

filterHH = [[1/8, -2/8, 1/8],
            [-2/8, 4/8, -2/8],
            [1/8, -2/8, 1/8]]

filterNLL = [[-1/32, 2/32, 6/32, 2/32, -1/32],
             [2/32, -4/32, -12/32, -4/32, 2/32],
             [6/32, -12/32, -36/32, -12/32, 6/32],
             [2/32, -4/32, -12/32, -4/32, 2/32],
             [-1/32, 2/32, 6/32, 2/32, -1/32]]

filterNLH = [[-1/16, 2/16, 6/16, 2/16, -1/16],
             [2/16, -4/16, -12/16, -4/16, 2/16],
             [-1/16, 2/16, 6/16, 2/16, -1/16]]

filterNHL = [[-1/16, 2/16, -1/16],
             [2/16, -4/16, 2/16],
             [6/16, -12/16, 6/16],
             [2/16, -4/16, 2/16],
             [-1/16, 2/16, -1/16]]

filterNHH = [[-1/8, 2/8, -1/8],
             [2/8, -4/8, 2/8],
             [-1/8, 2/8, -1/8]]

# create a net
main_input = Input(shape=(1,), name='main_input')
# for m in range(M):
#     for n in range(32, N):
for m in range(startm, startm + q):
    for n in range(startn, startn + q):
        main_layer_name = 'main_layer' + str(1000 * m + n)
        main_layer1_name = 'main1_layer' + str(1000 * m + n)
        side_layer_name = 'side_layer' + str(1000 * m + n)
        side_layer1_name = 'side1_layer' + str(1000 * m + n)
        shisalayer_name = 'shisalayer' + str(1000 * m + n)
        reshape_name = 'reshape_name' + str(1000 * m + n)
        reshape1_name = 'reshape1_name' + str(1000 * m + n)
        for i in range(LD):
            weight[i, ] = imr[m, n - i]
        main_layer = Dense(LD,
                           kernel_initializer=keras.initializers.constant(value=probability),
                           use_bias=False,
                           input_shape=(1,),
                           name=main_layer_name,
                           activation='softmax')(main_input)
        main_layer1 = Dense(3,
                            kernel_initializer=keras.initializers.constant(value=weight),
                            use_bias=False,
                            trainable=False,
                            activation='linear',
                            name=main_layer1_name)(main_layer)
        # side_layer = Dense(LD,
        #                    use_bias=False,
        #                    kernel_initializer=keras.initializers.constant(value=side_weight),
        #                    trainable=False,
        #                    activation=pp,
        #                    name=side_layer_name)(main_layer)
        reshape = Reshape(target_shape=(LD, 1), name=reshape_name)(main_layer)
        side_layer_Conv = Conv1D(filters=1,
                                 kernel_size=1,
                                 activation=pp,
                                 trainable=False,
                                 use_bias=False,
                                 strides=1,
                                 input_shape=(1,),
                                 padding='same',
                                 kernel_initializer=keras.initializers.constant(value=1),
                                 name=side_layer_name)(reshape)
        reshape1 = Flatten(name=reshape1_name)(side_layer_Conv)
        side_layer1 = Dense(1,
                            use_bias=False,
                            trainable=False,
                            activation='linear',
                            kernel_initializer=keras.initializers.ones(),
                            name=side_layer1_name)(reshape1)
        shisalayer = Dense(1,
                           use_bias=False,
                           trainable=False,
                           activation='linear',
                           kernel_initializer=keras.initializers.constant(value=sum_weight),
                           name=shisalayer_name)(main_layer)
        x_list.append(main_layer1)
        y_list.append(side_layer1)
        z_list.append(shisalayer)

print(x_list)
print(y_list)
print(z_list)
main_layerall = Concatenate(axis=-1, name='main_layerall')(x_list)
side_layerall = Concatenate(axis=-1, name='side_layerall')(y_list)
wave_layerall = Concatenate(axis=-1, name='wave_layerall')(z_list)
print(main_layerall)
print(side_layerall)
print(wave_layerall)
main_reshape = Reshape(input_shape=(q * q * 3, -1), target_shape=(q, q, 3), name='main_reshape')(main_layerall)
print(main_reshape)

side_layerallsum = Dense(1,
                         use_bias=False,
                         trainable=False,
                         activation='linear',
                         kernel_initializer=keras.initializers.ones(),
                         name='side_layerallsum')(side_layerall)
print(side_layerallsum)

wave_reshape = Reshape(input_shape=(q * q, -1), target_shape=(q, q, 1), name='wave_reshape')(wave_layerall)
print(wave_reshape)
wavelet_LL = Conv2D(filters=1,
                    kernel_size=(5, 5),
                    kernel_initializer=keras.initializers.constant(filterLL),
                    padding='same',
                    strides=(1, 1),
                    # data_format='channels_first',
                    trainable=False,
                    use_bias=False,
                    name='wavelet_LL',
                    activation='relu')(wave_reshape)
print(wavelet_LL)
wavelet_LH = Conv2D(filters=1,
                    kernel_size=(3, 5),
                    kernel_initializer=keras.initializers.constant(filterLH),
                    padding='same',
                    strides=(1, 1),
                    trainable=False,
                    use_bias=False,
                    name='wavelet_LH',
                    activation='relu')(wave_reshape)

wavelet_HL = Conv2D(filters=1,
                    kernel_size=(5, 3),
                    kernel_initializer=keras.initializers.constant(filterHL),
                    padding='same',
                    strides=(1, 1),
                    trainable=False,
                    use_bias=False,
                    name='wavelet_HL',
                    activation='relu')(wave_reshape)

wavelet_HH = Conv2D(filters=1,
                    kernel_size=(3, 3),
                    kernel_initializer=keras.initializers.constant(filterHH),
                    padding='same',
                    strides=(1, 1),
                    trainable=False,
                    use_bias=False,
                    name='wavelet_HH',
                    activation='relu')(wave_reshape)

wavelet_NLL = Conv2D(filters=1,
                     kernel_size=(5, 5),
                     kernel_initializer=keras.initializers.constant(filterNLL),
                     padding='same',
                     strides=(1, 1),
                     trainable=False,
                     use_bias=False,
                     name='wavelet_NLL',
                     activation='relu')(wave_reshape)

wavelet_NLH = Conv2D(filters=1,
                     kernel_size=(3, 5),
                     kernel_initializer=keras.initializers.constant(filterLH),
                     padding='same',
                     strides=(1, 1),
                     trainable=False,
                     use_bias=False,
                     name='wavelet_NLH',
                     activation='relu')(wave_reshape)

wavelet_NHL = Conv2D(filters=1,
                     kernel_size=(5, 3),
                     kernel_initializer=keras.initializers.constant(filterHL),
                     padding='same',
                     strides=(1, 1),
                     trainable=False,
                     use_bias=False,
                     name='wavelet_NHL',
                     activation='relu')(wave_reshape)

wavelet_NHH = Conv2D(filters=1,
                     kernel_size=(3, 3),
                     kernel_initializer=keras.initializers.constant(filterHH),
                     padding='same',
                     strides=(1, 1),
                     trainable=False,
                     use_bias=False,
                     name='wavelet_NHH',
                     activation='relu')(wave_reshape)

waveLL = Flatten()(wavelet_LL)
waveLH = Flatten()(wavelet_LH)
waveHL = Flatten()(wavelet_HL)
waveHH = Flatten()(wavelet_HH)
waveNLL = Flatten()(wavelet_NLL)
waveNLH = Flatten()(wavelet_NLH)
waveNHL = Flatten()(wavelet_NHL)
waveNHH = Flatten()(wavelet_NHH)

print(waveLL)
flaten1 = Concatenate(name='flaten1')(
            [waveLL, waveHH, waveHL, waveLH, waveNLL, waveNHL, waveNLH, waveNHH])
wavesum = Dense(1,
                use_bias=False,
                trainable=False,
                activation='linear',
                kernel_initializer=keras.initializers.ones(),
                name='wavesum')(flaten1)
print(flaten1)

model = Model([main_input], [main_reshape, side_layerallsum, wavesum])
print('1111111111111111111111111111111')
adam = Adam(lr=1)
model.compile(optimizer=adam,
              loss={'main_reshape': 'mse',
                    'side_layerallsum': 'mae',
                    'wavesum': 'mae'},
              metrics=['accuracy'],
              loss_weights={'main_reshape': alpha,
                            'side_layerallsum': beta,
                            'wavesum': gamma})
print('2222222222222222222222222222222')
model.fit([input_grey],
          [testtarget,
           test1,
           test1],
          callbacks=[history],
          epochs=10,
          batch_size=10)
print('333333333333333333333333333333')
dense1_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('wave_reshape').output)
dense1_output = dense1_layer_model.predict(a)
print(dense1_output.shape)
print(dense1_output[0])
shisamap = np.uint8(dense1_output)
sa = np.reshape(shisamap, (q, q))
img = Image.fromarray(sa, 'L')
img.save('12102.png')
print(sa)

dense1_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('main_reshape').output)
dense1_output = dense1_layer_model.predict(a)
print(dense1_output.shape)
print(dense1_output[0])

check = np.ones((q, q, 3))
for i in range(q):
    for j in range(q):
        check[i, j, ] = iml[startm + i, startn + j, ]
print(check)
# model.summary()
# print(model.get_weights())
# shisa = 0
# print(shisa)
# for i in range(0, 32):
#     shisa += i * dense1_output[0, i]
# print(shisa)
# shisamap[m, n] = np.uint8(shisa*8)

history.loss_plot('epoch')

