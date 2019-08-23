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
alpha = 1
beta = 20
gamma = 1
# m = 184
# n = 319
m = 40
n = 40
# reshape_imr = np.reshape(imr, (165390, 3))
kaisu = 5000
probability = np.ones(32)
# probability[31] = 10
# probability[23] = 10
# probability[15] = 10
# probability[7] = 10
# probability[0] = 10
input_grey = np.ones((kaisu, 1), dtype=np.uint8)
target_pixel11 = iml[m, n, ]
target_pixel12 = iml[m, n + 1, ]
target_pixel13 = iml[m, n + 2, ]
target_pixel14 = iml[m, n + 3, ]

target_pixel21 = iml[m + 1, n, ]
target_pixel22 = iml[m + 1, n + 1, ]
target_pixel23 = iml[m + 1, n + 2, ]
target_pixel24 = iml[m + 1, n + 3, ]

target_pixel31 = iml[m + 2, n, ]
target_pixel32 = iml[m + 2, n + 1, ]
target_pixel33 = iml[m + 2, n + 2, ]
target_pixel34 = iml[m + 2, n + 3, ]

target_pixel41 = iml[m + 3, n, ]
target_pixel42 = iml[m + 3, n + 1, ]
target_pixel43 = iml[m + 3, n + 2, ]
target_pixel44 = iml[m + 3, n + 3, ]

target_pixel11_100 = np.zeros((kaisu, 3))
target_pixel12_100 = np.zeros((kaisu, 3))
target_pixel13_100 = np.zeros((kaisu, 3))
target_pixel14_100 = np.zeros((kaisu, 3))

target_pixel21_100 = np.zeros((kaisu, 3))
target_pixel22_100 = np.zeros((kaisu, 3))
target_pixel23_100 = np.zeros((kaisu, 3))
target_pixel24_100 = np.zeros((kaisu, 3))

target_pixel31_100 = np.zeros((kaisu, 3))
target_pixel32_100 = np.zeros((kaisu, 3))
target_pixel33_100 = np.zeros((kaisu, 3))
target_pixel34_100 = np.zeros((kaisu, 3))

target_pixel41_100 = np.zeros((kaisu, 3))
target_pixel42_100 = np.zeros((kaisu, 3))
target_pixel43_100 = np.zeros((kaisu, 3))
target_pixel44_100 = np.zeros((kaisu, 3))

target_side_100 = np.zeros((kaisu, 1))
a = array([1])
b = [a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a]

layer1_weights = np.zeros((32, 32))

layer112_weights = np.zeros((32, 3))
layer122_weights = np.zeros((32, 3))
layer132_weights = np.zeros((32, 3))
layer142_weights = np.zeros((32, 3))

layer212_weights = np.zeros((32, 3))
layer222_weights = np.zeros((32, 3))
layer232_weights = np.zeros((32, 3))
layer242_weights = np.zeros((32, 3))

layer312_weights = np.zeros((32, 3))
layer322_weights = np.zeros((32, 3))
layer332_weights = np.zeros((32, 3))
layer342_weights = np.zeros((32, 3))

layer412_weights = np.zeros((32, 3))
layer422_weights = np.zeros((32, 3))
layer432_weights = np.zeros((32, 3))
layer442_weights = np.zeros((32, 3))

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

test1 = np.zeros((kaisu, 1, 4, 4))

for i in range(0, 32):
    sidelayer1_weights[i, 2*i] = 1
    sidelayer1_weights[i, 2*i+1] = 1

for i in range(0, 32):
    sidelayer2_weights[2*i, i] = 1
    sidelayer2_weights[2*i+1, i] = -2

for i in range(0, 32):
    sidelayer1_bias[2*i+1] = -0.5

for i in range(0, 32):
    layer112_weights[i, ] = imr[m, n - i]
    layer122_weights[i, ] = imr[m, n + 1 - i]
    layer132_weights[i, ] = imr[m, n + 2 - i]
    layer142_weights[i, ] = imr[m, n + 3 - i]

    layer212_weights[i, ] = imr[m + 1, n - i]
    layer222_weights[i, ] = imr[m + 1, n + 1 - i]
    layer232_weights[i, ] = imr[m + 1, n + 2 - i]
    layer242_weights[i, ] = imr[m + 1, n + 3 - i]

    layer312_weights[i, ] = imr[m + 2, n - i]
    layer322_weights[i, ] = imr[m + 2, n + 1 - i]
    layer332_weights[i, ] = imr[m + 2, n + 2 - i]
    layer342_weights[i, ] = imr[m + 2, n + 3 - i]

    layer412_weights[i, ] = imr[m + 3, n - i]
    layer422_weights[i, ] = imr[m + 3, n + 1 - i]
    layer432_weights[i, ] = imr[m + 3, n + 2 - i]
    layer442_weights[i, ] = imr[m + 3, n + 3 - i]

for i in range(0, kaisu):
    target_pixel11_100[i, ] = target_pixel11
    target_pixel12_100[i, ] = target_pixel12
    target_pixel13_100[i, ] = target_pixel13
    target_pixel14_100[i, ] = target_pixel14

    target_pixel21_100[i, ] = target_pixel21
    target_pixel22_100[i, ] = target_pixel22
    target_pixel23_100[i, ] = target_pixel23
    target_pixel24_100[i, ] = target_pixel24

    target_pixel31_100[i, ] = target_pixel31
    target_pixel32_100[i, ] = target_pixel32
    target_pixel33_100[i, ] = target_pixel33
    target_pixel34_100[i, ] = target_pixel34

    target_pixel41_100[i, ] = target_pixel41
    target_pixel42_100[i, ] = target_pixel42
    target_pixel43_100[i, ] = target_pixel43
    target_pixel44_100[i, ] = target_pixel44



# 四个网络都用同一个1-1

for i in range(0, 32):
    layer1_weights[i, i] = 1

for i in range(0, 32):
    shisasum_weights[i] = i * 8

# 11
main_input11 = Input(shape=(1, ), name='main_input11')
main_layer111 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer111',
                      activation='softmax')(main_input11)
main_layer112 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer112',
                      activation='linear')(main_layer111)
main_layer113 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer112_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer113',
                      activation='linear')(main_layer112)
side_layer111 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer111',
                      activation='relu')(main_layer112)
side_layer112 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer112',
                      activation='linear')(side_layer111)
side_layer113 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer113',
                      activation='linear')(side_layer112)
# 11

# 12
main_input12 = Input(shape=(1, ), name='main_input12')
main_layer121 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer121',
                      activation='softmax')(main_input12)
main_layer122 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer122',
                      activation='linear')(main_layer121)
main_layer123 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer122_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer123',
                      activation='linear')(main_layer122)
side_layer121 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer121',
                      activation='relu')(main_layer122)
side_layer122 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer122',
                      activation='linear')(side_layer121)
side_layer123 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer123',
                      activation='linear')(side_layer122)
# 12

# 13
main_input13 = Input(shape=(1, ), name='main_input13')
main_layer131 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer131',
                      activation='softmax')(main_input13)
main_layer132 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer132',
                      activation='linear')(main_layer131)
main_layer133 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer132_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer133',
                      activation='linear')(main_layer132)
side_layer131 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer131',
                      activation='relu')(main_layer132)
side_layer132 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer132',
                      activation='linear')(side_layer131)
side_layer133 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer133',
                      activation='linear')(side_layer132)
# 13

# 14
main_input14 = Input(shape=(1, ), name='main_input14')
main_layer141 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer141',
                      activation='softmax')(main_input14)
main_layer142 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer142',
                      activation='linear')(main_layer141)
main_layer143 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer142_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer143',
                      activation='linear')(main_layer142)
side_layer141 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer141',
                      activation='relu')(main_layer142)
side_layer142 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer142',
                      activation='linear')(side_layer141)
side_layer143 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer143',
                      activation='linear')(side_layer142)
# 14

# 21
main_input21 = Input(shape=(1, ), name='main_input21')
main_layer211 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer211',
                      activation='softmax')(main_input21)
main_layer212 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer212',
                      activation='linear')(main_layer211)
main_layer213 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer212_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer213',
                      activation='linear')(main_layer212)
side_layer211 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer211',
                      activation='relu')(main_layer212)
side_layer212 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer212',
                      activation='linear')(side_layer211)
side_layer213 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer213',
                      activation='linear')(side_layer212)
# 21

# 22
main_input22 = Input(shape=(1, ), name='main_input22')
main_layer221 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer221',
                      activation='softmax')(main_input22)
main_layer222 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer222',
                      activation='linear')(main_layer221)
main_layer223 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer222_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer223',
                      activation='linear')(main_layer222)
side_layer221 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer221',
                      activation='relu')(main_layer222)
side_layer222 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer222',
                      activation='linear')(side_layer221)
side_layer223 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer223',
                      activation='linear')(side_layer222)
# 22

# 23
main_input23 = Input(shape=(1, ), name='main_input23')
main_layer231 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer231',
                      activation='softmax')(main_input23)
main_layer232 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer232',
                      activation='linear')(main_layer231)
main_layer233 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer232_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer233',
                      activation='linear')(main_layer232)
side_layer231 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer231',
                      activation='relu')(main_layer232)
side_layer232 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer232',
                      activation='linear')(side_layer231)
side_layer233 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer233',
                      activation='linear')(side_layer232)
# 23

# 24
main_input24 = Input(shape=(1, ), name='main_input24')
main_layer241 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer241',
                      activation='softmax')(main_input24)
main_layer242 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer242',
                      activation='linear')(main_layer241)
main_layer243 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer242_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer243',
                      activation='linear')(main_layer242)
side_layer241 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer241',
                      activation='relu')(main_layer242)
side_layer242 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer242',
                      activation='linear')(side_layer241)
side_layer243 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer243',
                      activation='linear')(side_layer242)
# 24

# 31
main_input31 = Input(shape=(1, ), name='main_input31')
main_layer311 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer311',
                      activation='softmax')(main_input31)
main_layer312 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer312',
                      activation='linear')(main_layer311)
main_layer313 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer312_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer313',
                      activation='linear')(main_layer312)
side_layer311 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer311',
                      activation='relu')(main_layer312)
side_layer312 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer312',
                      activation='linear')(side_layer311)
side_layer313 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer313',
                      activation='linear')(side_layer312)
# 31

# 32
main_input32 = Input(shape=(1, ), name='main_input32')
main_layer321 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer321',
                      activation='softmax')(main_input32)
main_layer322 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer322',
                      activation='linear')(main_layer321)
main_layer323 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer322_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer323',
                      activation='linear')(main_layer322)
side_layer321 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer321',
                      activation='relu')(main_layer322)
side_layer322 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer322',
                      activation='linear')(side_layer321)
side_layer323 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer323',
                      activation='linear')(side_layer322)
# 32

# 33
main_input33 = Input(shape=(1, ), name='main_input33')
main_layer331 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer331',
                      activation='softmax')(main_input33)
main_layer332 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer332',
                      activation='linear')(main_layer331)
main_layer333 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer332_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer333',
                      activation='linear')(main_layer332)
side_layer331 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer331',
                      activation='relu')(main_layer332)
side_layer332 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer332',
                      activation='linear')(side_layer331)
side_layer333 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer333',
                      activation='linear')(side_layer332)
# 33

# 34
main_input34 = Input(shape=(1, ), name='main_input34')
main_layer341 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer341',
                      activation='softmax')(main_input34)
main_layer342 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer342',
                      activation='linear')(main_layer341)
main_layer343 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer342_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer343',
                      activation='linear')(main_layer342)
side_layer341 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer341',
                      activation='relu')(main_layer342)
side_layer342 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer342',
                      activation='linear')(side_layer341)
side_layer343 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer343',
                      activation='linear')(side_layer342)
# 34

# 41
main_input41 = Input(shape=(1, ), name='main_input41')
main_layer411 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer411',
                      activation='softmax')(main_input41)
main_layer412 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer412',
                      activation='linear')(main_layer411)
main_layer413 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer412_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer413',
                      activation='linear')(main_layer412)
side_layer411 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer411',
                      activation='relu')(main_layer412)
side_layer412 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer412',
                      activation='linear')(side_layer411)
side_layer413 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer413',
                      activation='linear')(side_layer412)
# 41

# 42
main_input42 = Input(shape=(1, ), name='main_input42')
main_layer421 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer421',
                      activation='softmax')(main_input42)
main_layer422 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer422',
                      activation='linear')(main_layer421)
main_layer423 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer422_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer423',
                      activation='linear')(main_layer422)
side_layer421 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer421',
                      activation='relu')(main_layer422)
side_layer422 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer422',
                      activation='linear')(side_layer421)
side_layer423 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer423',
                      activation='linear')(side_layer422)
# 42

# 43
main_input43 = Input(shape=(1, ), name='main_input43')
main_layer431 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer431',
                      activation='softmax')(main_input43)
main_layer432 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer432',
                      activation='linear')(main_layer431)
main_layer433 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer432_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer433',
                      activation='linear')(main_layer432)
side_layer431 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer431',
                      activation='relu')(main_layer432)
side_layer432 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer432',
                      activation='linear')(side_layer431)
side_layer433 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer433',
                      activation='linear')(side_layer432)
# 43

# 44
main_input44 = Input(shape=(1, ), name='main_input44')
main_layer441 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=probability),
                      use_bias=False,
                      input_shape=(1, ),
                      name='main_layer441',
                      activation='softmax')(main_input44)
main_layer442 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=layer1_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer442',
                      activation='linear')(main_layer441)
main_layer443 = Dense(3,
                      kernel_initializer=keras.initializers.constant(value=layer442_weights),
                      use_bias=False,
                      trainable=False,
                      name='main_layer443',
                      activation='linear')(main_layer442)
side_layer441 = Dense(64,
                      kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                      use_bias=True,
                      trainable=False,
                      bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                      name='side_layer441',
                      activation='relu')(main_layer442)
side_layer442 = Dense(32,
                      kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer442',
                      activation='linear')(side_layer441)
side_layer443 = Dense(1,
                      kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                      use_bias=False,
                      trainable=False,
                      name='side_layer443',
                      activation='linear')(side_layer442)
# 44

shisasum11 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum11',
                   activation='linear')(main_layer112)
shisasum12 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum12',
                   activation='linear')(main_layer122)
shisasum13 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum13',
                   activation='linear')(main_layer132)
shisasum14 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum14',
                   activation='linear')(main_layer142)

shisasum21 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum21',
                   activation='linear')(main_layer212)
shisasum22 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum22',
                   activation='linear')(main_layer222)
shisasum23 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum23',
                   activation='linear')(main_layer232)
shisasum24 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum24',
                   activation='linear')(main_layer242)

shisasum31 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum31',
                   activation='linear')(main_layer312)
shisasum32 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum32',
                   activation='linear')(main_layer322)
shisasum33 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum33',
                   activation='linear')(main_layer332)
shisasum34 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum34',
                   activation='linear')(main_layer342)

shisasum41 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum41',
                   activation='linear')(main_layer412)
shisasum42 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum42',
                   activation='linear')(main_layer422)
shisasum43 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum43',
                   activation='linear')(main_layer432)
shisasum44 = Dense(1,
                   kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                   use_bias=False,
                   trainable=False,
                   name='shisasum44',
                   activation='linear')(main_layer442)

concatenated1 = Concatenate(axis=1, name='concatenated1')([shisasum11, shisasum12, shisasum13, shisasum14,
                                                           shisasum21, shisasum22, shisasum23, shisasum24,
                                                           shisasum31, shisasum32, shisasum33, shisasum31,
                                                           shisasum41, shisasum42, shisasum43, shisasum44,])

reshaped = Reshape(input_shape=(16, -1), target_shape=(1, 4, 4), name='reshaped')(concatenated1)
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

reshaped1 = Reshape(input_shape=(16, -1), target_shape=(1, 4, 4), name='reshaped1')(concatenated2)

model1 = Model([main_input11, main_input12, main_input13, main_input14,
                main_input21, main_input22, main_input23, main_input24,
                main_input31, main_input32, main_input33, main_input34,
                main_input41, main_input42, main_input43, main_input44,],
               [main_layer113, side_layer113, main_layer123, side_layer123, main_layer133, side_layer133, main_layer143, side_layer143,
                main_layer213, side_layer213, main_layer223, side_layer223, main_layer233, side_layer233, main_layer243, side_layer243,
                main_layer313, side_layer313, main_layer323, side_layer323, main_layer333, side_layer333, main_layer343, side_layer343,
                main_layer413, side_layer413, main_layer423, side_layer423, main_layer433, side_layer433, main_layer443, side_layer443,  reshaped1])

# print(model1.get_weights())

rMSprop = RMSprop(lr=1e-1)
adadelta = Adadelta(lr=1e-1)
adam = Adam()
# entropy权重0
model1.compile(optimizer=adam,
               loss='mse',
               metrics=['accuracy'],
               loss_weights={'main_layer113': alpha, 'side_layer113': beta, 'main_layer123': alpha, 'side_layer123': beta, 'main_layer133': alpha, 'side_layer133': beta, 'main_layer143': alpha, 'side_layer143': beta,
                             'main_layer213': alpha, 'side_layer213': beta, 'main_layer223': alpha, 'side_layer223': beta, 'main_layer233': alpha, 'side_layer233': beta, 'main_layer243': alpha, 'side_layer243': beta,
                             'main_layer313': alpha, 'side_layer313': beta, 'main_layer323': alpha, 'side_layer323': beta, 'main_layer333': alpha, 'side_layer333': beta, 'main_layer343': alpha, 'side_layer343': beta,
                             'main_layer413': alpha, 'side_layer413': beta, 'main_layer423': alpha, 'side_layer423': beta, 'main_layer433': alpha, 'side_layer433': beta, 'main_layer443': alpha, 'side_layer443': beta, 'reshaped1': 0}
               )

model1.fit([input_grey, input_grey, input_grey, input_grey,
            input_grey, input_grey, input_grey, input_grey,
            input_grey, input_grey, input_grey, input_grey,
            input_grey, input_grey, input_grey, input_grey],
           [target_pixel11_100, target_side_100, target_pixel12_100, target_side_100, target_pixel13_100, target_side_100, target_pixel14_100, target_side_100,
            target_pixel21_100, target_side_100, target_pixel22_100, target_side_100, target_pixel23_100, target_side_100, target_pixel24_100, target_side_100,
            target_pixel31_100, target_side_100, target_pixel32_100, target_side_100, target_pixel33_100, target_side_100, target_pixel34_100, target_side_100,
            target_pixel41_100, target_side_100, target_pixel42_100, target_side_100, target_pixel43_100, target_side_100, target_pixel44_100, target_side_100, test1],
           epochs=100, batch_size=10)

# print(model1.get_output_at(0))

# print(model1.get_weights())

model1.summary()

#
# dense5_layer_model = Model(inputs=model1.input,
#                            outputs=model1.get_layer('main_layer12').output)
#
# dense5_output = dense5_layer_model.predict([a, a, a, a])
#
# print(dense5_output.shape)
# print(dense5_output[0])
# shisa = 0
# print(shisa)
# for i in range(0, 32):
#     shisa += i * dense5_output[0, i]
# print(shisa)
#
# dense5_layer_model = Model(inputs=model1.input,
#                            outputs=model1.get_layer('main_layer22').output)
#
# dense5_output = dense5_layer_model.predict([a, a, a, a])
#
# print(dense5_output.shape)
# print(dense5_output[0])
# shisa = 0
# print(shisa)
# for i in range(0, 32):
#     shisa += i * dense5_output[0, i]
# print(shisa)
#
# dense5_layer_model = Model(inputs=model1.input,
#                            outputs=model1.get_layer('main_layer32').output)
#
# dense5_output = dense5_layer_model.predict([a, a, a, a])
#
# print(dense5_output.shape)
# print(dense5_output[0])
# shisa = 0
# print(shisa)
# for i in range(0, 32):
#     shisa += i * dense5_output[0, i]
# print(shisa)
#
# dense5_layer_model = Model(inputs=model1.input,
#                            outputs=model1.get_layer('main_layer42').output)
#
# dense5_output = dense5_layer_model.predict([a, a, a, a])
#
# print(dense5_output.shape)
# print(dense5_output[0])
# shisa = 0
# print(shisa)
# for i in range(0, 32):
#     shisa += i * dense5_output[0, i]
# print(shisa)
#
# dense5_layer_model = Model(inputs=model1.input,
#                            outputs=model1.get_layer('main_layer13').output)
# dense5_output = dense5_layer_model.predict([a, a, a, a])
# print(dense5_output[0])
#
# dense5_layer_model = Model(inputs=model1.input,
#                            outputs=model1.get_layer('main_layer23').output)
# dense5_output = dense5_layer_model.predict([a, a, a, a])
# print(dense5_output[0])
#
# dense5_layer_model = Model(inputs=model1.input,
#                            outputs=model1.get_layer('main_layer33').output)
# dense5_output = dense5_layer_model.predict([a, a, a, a])
# print(dense5_output[0])
#
# dense5_layer_model = Model(inputs=model1.input,
#                            outputs=model1.get_layer('main_layer43').output)
# dense5_output = dense5_layer_model.predict([a, a, a, a])
# print(dense5_output[0])
#
# dense5_layer_model = Model(inputs=model1.input,
#                            outputs=model1.get_layer('reshaped').output)
# dense5_output = dense5_layer_model.predict([a, a, a, a])
# print(dense5_output[0])

# print(iml[m, n, ])
# print(iml[m, n+1, ])
# print(iml[m+1, n, ])
# print(iml[m+1, n+1, ])

# 11111
# print(layer112_weights)
# print(layer122_weights)
# print(layer132_weights)
# print(layer142_weights)
# print(layer152_weights)
# print(layer162_weights)
# print(layer172_weights)
# print(layer182_weights)
#
# print(layer212_weights)
# print(layer222_weights)
# print(layer232_weights)
# print(layer242_weights)
# print(layer252_weights)
# print(layer262_weights)
# print(layer272_weights)
# print(layer282_weights)
#
# print(layer312_weights)
# print(layer322_weights)
# print(layer332_weights)
# print(layer342_weights)
# print(layer352_weights)
# print(layer362_weights)
# print(layer372_weights)
# print(layer382_weights)
#
# print(layer412_weights)
# print(layer422_weights)
# print(layer432_weights)
# print(layer442_weights)
# print(layer452_weights)
# print(layer462_weights)
# print(layer472_weights)
# print(layer482_weights)
#
# print(layer512_weights)
# print(layer522_weights)
# print(layer532_weights)
# print(layer542_weights)
# print(layer552_weights)
# print(layer562_weights)
# print(layer572_weights)
# print(layer582_weights)
#
# print(layer612_weights)
# print(layer622_weights)
# print(layer632_weights)
# print(layer642_weights)
# print(layer652_weights)
# print(layer662_weights)
# print(layer672_weights)
# print(layer682_weights)
#
# print(layer712_weights)
# print(layer722_weights)
# print(layer732_weights)
# print(layer742_weights)
# print(layer752_weights)
# print(layer762_weights)
# print(layer772_weights)
# print(layer782_weights)
#
# print(layer812_weights)
# print(layer822_weights)
# print(layer832_weights)
# print(layer842_weights)
# print(layer852_weights)
# print(layer862_weights)
# print(layer872_weights)
# print(layer882_weights)
# # 1111111


# 1
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer112').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer122').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer132').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer142').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

# 2
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer212').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer222').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer232').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer242').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

# 3
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer312').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer322').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer332').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

# 4
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer412').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer422').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer432').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer442').output)

dense1_output = dense1_layer_model.predict(b)

print(dense1_output.shape)
print(dense1_output[0])
shisa = 0
print(shisa)
for i in range(0, 32):
    shisa += i * dense1_output[0, i]
print(shisa)

# ------------------------
# 1
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer113').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer123').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer133').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer143').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])

# 2
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer213').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer223').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer233').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer243').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])

# 3
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer313').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer323').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer333').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer343').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])

# 4
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer413').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer423').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer433').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])
dense1_layer_model = Model(inputs=model1.input,
                           outputs=model1.get_layer('main_layer443').output)
dense1_output = dense1_layer_model.predict(b)
print(dense1_output[0])


print(iml[m, n, ], iml[m, n+1, ], iml[m, n+2, ], iml[m, n+3, ])
print(iml[m+1, n, ], iml[m+1, n+1, ], iml[m+1, n+2, ], iml[m+1, n+3, ])
print(iml[m+2, n, ], iml[m+2, n+1, ], iml[m+2, n+2, ], iml[m+2, n+3, ])
print(iml[m+3, n, ], iml[m+3, n+1, ], iml[m+3, n+2, ], iml[m+3, n+3, ])

