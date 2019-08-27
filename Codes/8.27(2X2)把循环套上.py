from __future__ import print_function
import PIL.Image as Image
from pylab import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
np.random.seed(1337)
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Concatenate, Conv2D, Reshape, AveragePooling2D, Add
from keras.optimizers import Adam, Adadelta, RMSprop
from keras.models import Model
import h5py
from keras.models import load_model

iml = array(Image.open("Laundry-7views\\Laundry\\view0.png"))
imr = array(Image.open("Laundry-7views\\Laundry\\view1.png"))
m = 184
n = 319
m = 42
n = 42
# reshape_imr = np.reshape(imr, (165390, 3))
kaisu = 10000
input_grey = np.ones((kaisu, 1), dtype=np.uint8)
target_pixel1 = iml[m, n, ]
target_pixel2 = iml[m, n + 1, ]
target_pixel3 = iml[m + 1, n, ]
target_pixel4 = iml[m + 1, n + 1, ]

gailv = np.zeros(32)
gailv[31] = 1

target_pixel1_100 = np.zeros((kaisu, 3))
target_pixel2_100 = np.zeros((kaisu, 3))
target_pixel3_100 = np.zeros((kaisu, 3))
target_pixel4_100 = np.zeros((kaisu, 3))
target_side_100 = np.zeros((kaisu, 1))
a = array([1])
b = [a,a,a,a]

layer1_weights = np.zeros((32, 32))
layer12_weights = np.zeros((32, 3))
layer22_weights = np.zeros((32, 3))
layer32_weights = np.zeros((32, 3))
layer42_weights = np.zeros((32, 3))
sidelayer1_weights = np.zeros((32, 64))
sidelayer2_weights = np.zeros((64, 32))
sidelayer1_bias = np.zeros(64)
sidelayer3_weights = np.ones(32)
shisasum_weights = np.ones(32)

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

test1 = np.zeros((kaisu, 1, 2, 2))


for i in range(0, 32):
    sidelayer1_weights[i, 2*i] = 1
    sidelayer1_weights[i, 2*i+1] = 1

for i in range(0, 32):
    sidelayer2_weights[2*i, i] = 1
    sidelayer2_weights[2*i+1, i] = -2

for i in range(0, 32):
    sidelayer1_bias[2*i+1] = -0.5



for i in range(0, kaisu):
    target_pixel1_100[i, ] = target_pixel1
    target_pixel2_100[i, ] = target_pixel2
    target_pixel3_100[i, ] = target_pixel3
    target_pixel4_100[i, ] = target_pixel4

# 四个网络都用同一个1-1

for i in range(0, 32):
    layer1_weights[i, i] = 1

for i in range(0, 32):
    shisasum_weights[i] = i * 8


main_input1 = Input(shape=(1, ), name='main_input1')
main_layer11 = Dense(32,
                     kernel_initializer=keras.initializers.constant(value=gailv),
                     use_bias=False,
                     input_shape=(1, ),
                     name='main_layer11',
                     activation='softmax')(main_input1)
main_layer12 = Dense(32,
                     kernel_initializer=keras.initializers.constant(value=layer1_weights),
                     use_bias=False,
                     trainable=False,
                     name='main_layer12',
                     activation='linear')(main_layer11)


side_layer11 = Dense(64,
                     kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                     use_bias=True,
                     trainable=False,
                     bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                     name='side_layer11',
                     activation='relu')(main_layer12)
side_layer12 = Dense(32,
                     kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                     use_bias=False,
                     trainable=False,
                     name='side_layer12',
                     activation='linear')(side_layer11)
side_layer13 = Dense(1,
                     kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                     use_bias=False,
                     trainable=False,
                     name='side_layer13',
                     activation='linear')(side_layer12)
# 左上

# 右上
main_input2 = Input(shape=(1, ), name='main_input2')
main_layer21 = Dense(32,
                     kernel_initializer=keras.initializers.constant(value=gailv),
                     use_bias=False,
                     input_shape=(1, ),
                     name='main_layer21',
                     activation='softmax')(main_input2)
main_layer22 = Dense(32,
                     kernel_initializer=keras.initializers.constant(value=layer1_weights),
                     use_bias=False,
                     trainable=False,
                     name='main_layer22',
                     activation='linear')(main_layer21)


side_layer21 = Dense(64,
                     kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                     use_bias=True,
                     trainable=False,
                     bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                     name='side_layer21',
                     activation='relu')(main_layer22)
side_layer22 = Dense(32,
                     kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                     use_bias=False,
                     trainable=False,
                     name='side_layer22',
                     activation='linear')(side_layer21)
side_layer23 = Dense(1,
                     kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                     use_bias=False,
                     trainable=False,
                     name='side_layer23',
                     activation='linear')(side_layer22)
# 右上

# 左下
main_input3 = Input(shape=(1, ), name='main_input3')
main_layer31 = Dense(32,
                     kernel_initializer=keras.initializers.constant(value=gailv),
                     use_bias=False,
                     input_shape=(1, ),
                     name='main_layer31',
                     activation='softmax')(main_input3)
main_layer32 = Dense(32,
                     kernel_initializer=keras.initializers.constant(value=layer1_weights),
                     use_bias=False,
                     trainable=False,
                     name='main_layer32',
                     activation='linear')(main_layer31)


side_layer31 = Dense(64,
                     kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                     use_bias=True,
                     trainable=False,
                     bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                     name='side_layer31',
                     activation='relu')(main_layer32)
side_layer32 = Dense(32,
                     kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                     use_bias=False,
                     trainable=False,
                     name='side_layer32',
                     activation='linear')(side_layer31)
side_layer33 = Dense(1,
                     kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                     use_bias=False,
                     trainable=False,
                     name='side_layer33',
                     activation='linear')(side_layer32)
# 左下

# 右下
main_input4 = Input(shape=(1, ), name='main_input4')
main_layer41 = Dense(32,
                     kernel_initializer=keras.initializers.constant(value=gailv),
                     use_bias=False,
                     input_shape=(1, ),
                     name='main_layer41',
                     activation='softmax')(main_input4)
main_layer42 = Dense(32,
                     kernel_initializer=keras.initializers.constant(value=layer1_weights),
                     use_bias=False,
                     trainable=False,
                     name='main_layer42',
                     activation='linear')(main_layer41)


side_layer41 = Dense(64,
                     kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                     use_bias=True,
                     trainable=False,
                     bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                     name='side_layer41',
                     activation='relu')(main_layer42)
side_layer42 = Dense(32,
                     kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                     use_bias=False,
                     trainable=False,
                     name='side_layer42',
                     activation='linear')(side_layer41)
side_layer43 = Dense(1,
                     kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                     use_bias=False,
                     trainable=False,
                     name='side_layer43',
                     activation='linear')(side_layer42)
# 右下

shisasum1 = Dense(1,
                  kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                  use_bias=False,
                  trainable=False,
                  name='shisasum1',
                  activation='linear')(main_layer12)

shisasum2 = Dense(1,
                  kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                  use_bias=False,
                  trainable=False,
                  name='shisasum2',
                  activation='linear')(main_layer22)

shisasum3 = Dense(1,
                  kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                  use_bias=False,
                  trainable=False,
                  name='shisasum3',
                  activation='linear')(main_layer32)

shisasum4 = Dense(1,
                  kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                  use_bias=False,
                  trainable=False,
                  name='shisasum4',
                  activation='linear')(main_layer42)

concatenated1 = Concatenate(axis=1, name='concatenated1')([shisasum1, shisasum2, shisasum3, shisasum4])

reshaped = Reshape(input_shape=(4, -1), target_shape=(1, 2, 2), name='reshaped')(concatenated1)
# CNN1 = Conv2D(input_shape=(0, 4), filters=8, kernel_size=2)(reshaped)

wavelet_LL = Conv2D(filters=1,
                    kernel_size=(5, 5),
                    kernel_initializer=keras.initializers.constant(filterLL),
                    padding='same',
                    strides=(1, 1),
                    data_format='channels_first',
                    trainable=False,
                    use_bias=False,
                    name='wavelet_LL',
                    activation='relu')(reshaped)

wavelet_LH = Conv2D(filters=1,
                    kernel_size=(3, 5),
                    kernel_initializer=keras.initializers.constant(filterLH),
                    padding='same',
                    strides=(1, 1),
                    data_format='channels_first',
                    trainable=False,
                    use_bias=False,
                    name='wavelet_LH',
                    activation='relu')(reshaped)

wavelet_HL = Conv2D(filters=1,
                    kernel_size=(5, 3),
                    kernel_initializer=keras.initializers.constant(filterHL),
                    padding='same',
                    strides=(1, 1),
                    data_format='channels_first',
                    trainable=False,
                    use_bias=False,
                    name='wavelet_HL',
                    activation='relu')(reshaped)

wavelet_HH = Conv2D(filters=1,
                    kernel_size=(3, 3),
                    kernel_initializer=keras.initializers.constant(filterHH),
                    padding='same',
                    strides=(1, 1),
                    data_format='channels_first',
                    trainable=False,
                    use_bias=False,
                    name='wavelet_HH',
                    activation='relu')(reshaped)

wavelet_NLL = Conv2D(filters=1,
                     kernel_size=(5, 5),
                     kernel_initializer=keras.initializers.constant(filterNLL),
                     padding='same',
                     strides=(1, 1),
                     data_format='channels_first',
                     trainable=False,
                     use_bias=False,
                     name='wavelet_NLL',
                     activation='relu')(reshaped)

wavelet_NLH = Conv2D(filters=1,
                     kernel_size=(3, 5),
                     kernel_initializer=keras.initializers.constant(filterLH),
                     padding='same',
                     strides=(1, 1),
                     data_format='channels_first',
                     trainable=False,
                     use_bias=False,
                     name='wavelet_NLH',
                     activation='relu')(reshaped)

wavelet_NHL = Conv2D(filters=1,
                     kernel_size=(5, 3),
                     kernel_initializer=keras.initializers.constant(filterHL),
                     padding='same',
                     strides=(1, 1),
                     data_format='channels_first',
                     trainable=False,
                     use_bias=False,
                     name='wavelet_NHL',
                     activation='relu')(reshaped)

wavelet_NHH = Conv2D(filters=1,
                     kernel_size=(3, 3),
                     kernel_initializer=keras.initializers.constant(filterHH),
                     padding='same',
                     strides=(1, 1),
                     data_format='channels_first',
                     trainable=False,
                     use_bias=False,
                     name='wavelet_NHH',
                     activation='relu')(reshaped)

wavelet_LL_averagepooling = AveragePooling2D(pool_size=2,
                                             strides=None,
                                             padding='same',
                                             data_format='channels_first',
                                             name='avgpoolingLL')(wavelet_LL)

wavelet_LH_averagepooling = AveragePooling2D(pool_size=2,
                                             strides=None,
                                             padding='same',
                                             data_format='channels_first',
                                             name='avgpoolingLH')(wavelet_LH)

wavelet_HL_averagepooling = AveragePooling2D(pool_size=2,
                                             strides=None,
                                             padding='same',
                                             data_format='channels_first',
                                             name='avgpoolingHL')(wavelet_HL)

wavelet_HH_averagepooling = AveragePooling2D(pool_size=2,
                                             strides=None,
                                             padding='same',
                                             data_format='channels_first',
                                             name='avgpoolingHH')(wavelet_HH)

wavelet_NLL_averagepooling = AveragePooling2D(pool_size=2,
                                              strides=None,
                                              padding='same',
                                              data_format='channels_first',
                                              name='avgpoolingNLL')(wavelet_NLL)

wavelet_NLH_averagepooling = AveragePooling2D(pool_size=2,
                                              strides=None,
                                              padding='same',
                                              data_format='channels_first',
                                              name='avgpoolingNLH')(wavelet_NLH)

wavelet_NHL_averagepooling = AveragePooling2D(pool_size=2,
                                              strides=None,
                                              padding='same',
                                              data_format='channels_first',
                                              name='avgpoolingNHL')(wavelet_NHL)

wavelet_NHH_averagepooling = AveragePooling2D(pool_size=2,
                                              strides=None,
                                              padding='same',
                                              data_format='channels_first',
                                              name='avgpoolingNHH')(wavelet_NHH)

addLL = Add()([wavelet_LL_averagepooling, wavelet_NLL_averagepooling])
addLH = Add()([wavelet_LH_averagepooling, wavelet_NLH_averagepooling])
addHL = Add()([wavelet_HL_averagepooling, wavelet_NHL_averagepooling])
addHH = Add()([wavelet_HH_averagepooling, wavelet_NHH_averagepooling])

concatenated2 = Concatenate(axis=1, name='concatenated2')([addLL, addLH, addHL, addHH])

reshaped1 = Reshape(input_shape=(4, -1), target_shape=(1, 2, 2), name='reshaped1')(concatenated2)



for i in range(0, 32):
    layer12_weights[i, ] = imr[m, n - i]
    layer22_weights[i, ] = imr[m, n + 1 - i]
    layer32_weights[i, ] = imr[m + 1, n - i]
    layer42_weights[i, ] = imr[m + 1, n + 1 - i]
# 左上

# print(model1.get_weights())
# keras.initializers.Initializer()

main_layer13 = Dense(3,
                     kernel_initializer=keras.initializers.constant(value=layer12_weights),
                     use_bias=False,
                     trainable=False,
                     name='main_layer13',
                     activation='linear')(main_layer12)
main_layer23 = Dense(3,
                     kernel_initializer=keras.initializers.constant(value=layer22_weights),
                     use_bias=False,
                     trainable=False,
                     name='main_layer23',
                     activation='linear')(main_layer22)
main_layer33 = Dense(3,
                     kernel_initializer=keras.initializers.constant(value=layer32_weights),
                     use_bias=False,
                     trainable=False,
                     name='main_layer33',
                     activation='linear')(main_layer32)
main_layer43 = Dense(3,
                     kernel_initializer=keras.initializers.constant(value=layer42_weights),
                     use_bias=False,
                     trainable=False,
                     name='main_layer43',
                     activation='linear')(main_layer42)
model1 = Model([main_input1, main_input2,
                main_input3, main_input4],
               [main_layer13, side_layer13, main_layer23, side_layer23,
                main_layer33, side_layer33, main_layer43, side_layer43, reshaped1])
rMSprop = RMSprop(lr=1e-1)
adadelta = Adadelta(lr=1e-1)
adam = Adam(lr=10, decay=0)
model1.compile(optimizer=adam,
               loss='mse',
               metrics=['accuracy'],
               loss_weights={'main_layer13': 1, 'side_layer13': 30,
                             'main_layer23': 1, 'side_layer23': 30,
                             'main_layer33': 1, 'side_layer33': 30,
                             'main_layer43': 1, 'side_layer43': 30, 'reshaped1': 0})

model1.fit([input_grey, input_grey, input_grey, input_grey],
           [target_pixel1_100, target_side_100, target_pixel2_100, target_side_100,
            target_pixel3_100, target_side_100, target_pixel4_100, target_side_100, test1],
           epochs=5, batch_size=10)

# print(model1.get_output_at(0))

print(model1.get_weights())

model1.summary()


dense5_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer12').output)

dense5_output = dense5_layer_model.predict(b)

print(dense5_output.shape)
print(dense5_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense5_output[0, i]
print(shisa)

dense5_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer22').output)

dense5_output = dense5_layer_model.predict([a, a, a, a])

print(dense5_output.shape)
print(dense5_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense5_output[0, i]
print(shisa)

dense5_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer32').output)

dense5_output = dense5_layer_model.predict([a, a, a, a])

print(dense5_output.shape)
print(dense5_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense5_output[0, i]
print(shisa)

dense5_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer42').output)

dense5_output = dense5_layer_model.predict([a, a, a, a])

print(dense5_output.shape)
print(dense5_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense5_output[0, i]
print(shisa)

dense5_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer13').output)
dense5_output = dense5_layer_model.predict([a, a, a, a])
print(dense5_output[0])

dense5_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer23').output)
dense5_output = dense5_layer_model.predict([a, a, a, a])
print(dense5_output[0])

dense5_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer33').output)
dense5_output = dense5_layer_model.predict([a, a, a, a])
print(dense5_output[0])

dense5_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer43').output)
dense5_output = dense5_layer_model.predict([a, a, a, a])
print(dense5_output[0])

dense5_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('reshaped').output)
dense5_output = dense5_layer_model.predict([a, a, a, a])
print(dense5_output[0])

print(iml[m, n, ])
print(iml[m, n+1, ])
print(iml[m+1, n, ])
print(iml[m+1, n+1, ])
print(layer12_weights)
print(layer22_weights)
print(layer32_weights)
print(layer42_weights)
