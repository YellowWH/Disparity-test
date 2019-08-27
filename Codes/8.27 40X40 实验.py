from __future__ import print_function
import PIL.Image as Image
from pylab import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
np.random.seed(1337)
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Concatenate, Conv2D, Reshape, AveragePooling2D, Add, Flatten
from keras.optimizers import Adam, Adadelta, RMSprop
from keras.models import Model
import h5py
from keras.models import load_model

iml = array(Image.open("Laundry-7views\\Laundry\\view0.png"))
imr = array(Image.open("Laundry-7views\\Laundry\\view1.png"))
alpha = 1
beta = 30
gamma = 0
# m = 184
# n = 319
m = 184
n = 319
# reshape_imr = np.reshape(imr, (165390, 3))
kaisu = 5000
probability = np.ones(32)
# probability[31] = 10
# probability[23] = 10
# probability[15] = 10
# probability[7] = 10
# probability[0] = 10
# probability = np.random.normal(loc=0.0, scale=1.0, size=32)
wave = np.ones(512)
input_grey = np.ones((kaisu, 1), dtype=np.uint8)
shisamap = np.ones((447, 370))
for i in range(0, 447):
    for j in range(0, 370):
        shisamap[i, j] = 255

target_pixel11_100 = np.zeros((kaisu, 3))
target_pixel12_100 = np.zeros((kaisu, 3))
target_pixel13_100 = np.zeros((kaisu, 3))
target_pixel14_100 = np.zeros((kaisu, 3))
target_pixel15_100 = np.zeros((kaisu, 3))
target_pixel16_100 = np.zeros((kaisu, 3))
target_pixel17_100 = np.zeros((kaisu, 3))
target_pixel18_100 = np.zeros((kaisu, 3))

target_pixel21_100 = np.zeros((kaisu, 3))
target_pixel22_100 = np.zeros((kaisu, 3))
target_pixel23_100 = np.zeros((kaisu, 3))
target_pixel24_100 = np.zeros((kaisu, 3))
target_pixel25_100 = np.zeros((kaisu, 3))
target_pixel26_100 = np.zeros((kaisu, 3))
target_pixel27_100 = np.zeros((kaisu, 3))
target_pixel28_100 = np.zeros((kaisu, 3))

target_pixel31_100 = np.zeros((kaisu, 3))
target_pixel32_100 = np.zeros((kaisu, 3))
target_pixel33_100 = np.zeros((kaisu, 3))
target_pixel34_100 = np.zeros((kaisu, 3))
target_pixel35_100 = np.zeros((kaisu, 3))
target_pixel36_100 = np.zeros((kaisu, 3))
target_pixel37_100 = np.zeros((kaisu, 3))
target_pixel38_100 = np.zeros((kaisu, 3))

target_pixel41_100 = np.zeros((kaisu, 3))
target_pixel42_100 = np.zeros((kaisu, 3))
target_pixel43_100 = np.zeros((kaisu, 3))
target_pixel44_100 = np.zeros((kaisu, 3))
target_pixel45_100 = np.zeros((kaisu, 3))
target_pixel46_100 = np.zeros((kaisu, 3))
target_pixel47_100 = np.zeros((kaisu, 3))
target_pixel48_100 = np.zeros((kaisu, 3))

target_pixel51_100 = np.zeros((kaisu, 3))
target_pixel52_100 = np.zeros((kaisu, 3))
target_pixel53_100 = np.zeros((kaisu, 3))
target_pixel54_100 = np.zeros((kaisu, 3))
target_pixel55_100 = np.zeros((kaisu, 3))
target_pixel56_100 = np.zeros((kaisu, 3))
target_pixel57_100 = np.zeros((kaisu, 3))
target_pixel58_100 = np.zeros((kaisu, 3))

target_pixel61_100 = np.zeros((kaisu, 3))
target_pixel62_100 = np.zeros((kaisu, 3))
target_pixel63_100 = np.zeros((kaisu, 3))
target_pixel64_100 = np.zeros((kaisu, 3))
target_pixel65_100 = np.zeros((kaisu, 3))
target_pixel66_100 = np.zeros((kaisu, 3))
target_pixel67_100 = np.zeros((kaisu, 3))
target_pixel68_100 = np.zeros((kaisu, 3))

target_pixel71_100 = np.zeros((kaisu, 3))
target_pixel72_100 = np.zeros((kaisu, 3))
target_pixel73_100 = np.zeros((kaisu, 3))
target_pixel74_100 = np.zeros((kaisu, 3))
target_pixel75_100 = np.zeros((kaisu, 3))
target_pixel76_100 = np.zeros((kaisu, 3))
target_pixel77_100 = np.zeros((kaisu, 3))
target_pixel78_100 = np.zeros((kaisu, 3))

target_pixel81_100 = np.zeros((kaisu, 3))
target_pixel82_100 = np.zeros((kaisu, 3))
target_pixel83_100 = np.zeros((kaisu, 3))
target_pixel84_100 = np.zeros((kaisu, 3))
target_pixel85_100 = np.zeros((kaisu, 3))
target_pixel86_100 = np.zeros((kaisu, 3))
target_pixel87_100 = np.zeros((kaisu, 3))
target_pixel88_100 = np.zeros((kaisu, 3))

# target_pixel1_100 = np.zeros((kaisu, 3))
# target_pixel2_100 = np.zeros((kaisu, 3))
# target_pixel3_100 = np.zeros((kaisu, 3))
# target_pixel4_100 = np.zeros((kaisu, 3))
target_side_100 = np.zeros((kaisu, 1))
a = array([1])
b = [a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a,
     a, a, a, a, a, a, a, a,]

layer1_weights = np.zeros((32, 32))

layer112_weights = np.zeros((32, 3))
layer122_weights = np.zeros((32, 3))
layer132_weights = np.zeros((32, 3))
layer142_weights = np.zeros((32, 3))
layer152_weights = np.zeros((32, 3))
layer162_weights = np.zeros((32, 3))
layer172_weights = np.zeros((32, 3))
layer182_weights = np.zeros((32, 3))

layer212_weights = np.zeros((32, 3))
layer222_weights = np.zeros((32, 3))
layer232_weights = np.zeros((32, 3))
layer242_weights = np.zeros((32, 3))
layer252_weights = np.zeros((32, 3))
layer262_weights = np.zeros((32, 3))
layer272_weights = np.zeros((32, 3))
layer282_weights = np.zeros((32, 3))

layer312_weights = np.zeros((32, 3))
layer322_weights = np.zeros((32, 3))
layer332_weights = np.zeros((32, 3))
layer342_weights = np.zeros((32, 3))
layer352_weights = np.zeros((32, 3))
layer362_weights = np.zeros((32, 3))
layer372_weights = np.zeros((32, 3))
layer382_weights = np.zeros((32, 3))

layer412_weights = np.zeros((32, 3))
layer422_weights = np.zeros((32, 3))
layer432_weights = np.zeros((32, 3))
layer442_weights = np.zeros((32, 3))
layer452_weights = np.zeros((32, 3))
layer462_weights = np.zeros((32, 3))
layer472_weights = np.zeros((32, 3))
layer482_weights = np.zeros((32, 3))

layer512_weights = np.zeros((32, 3))
layer522_weights = np.zeros((32, 3))
layer532_weights = np.zeros((32, 3))
layer542_weights = np.zeros((32, 3))
layer552_weights = np.zeros((32, 3))
layer562_weights = np.zeros((32, 3))
layer572_weights = np.zeros((32, 3))
layer582_weights = np.zeros((32, 3))

layer612_weights = np.zeros((32, 3))
layer622_weights = np.zeros((32, 3))
layer632_weights = np.zeros((32, 3))
layer642_weights = np.zeros((32, 3))
layer652_weights = np.zeros((32, 3))
layer662_weights = np.zeros((32, 3))
layer672_weights = np.zeros((32, 3))
layer682_weights = np.zeros((32, 3))

layer712_weights = np.zeros((32, 3))
layer722_weights = np.zeros((32, 3))
layer732_weights = np.zeros((32, 3))
layer742_weights = np.zeros((32, 3))
layer752_weights = np.zeros((32, 3))
layer762_weights = np.zeros((32, 3))
layer772_weights = np.zeros((32, 3))
layer782_weights = np.zeros((32, 3))

layer812_weights = np.zeros((32, 3))
layer822_weights = np.zeros((32, 3))
layer832_weights = np.zeros((32, 3))
layer842_weights = np.zeros((32, 3))
layer852_weights = np.zeros((32, 3))
layer862_weights = np.zeros((32, 3))
layer872_weights = np.zeros((32, 3))
layer882_weights = np.zeros((32, 3))

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

test1 = np.zeros(kaisu)

for i in range(0, 32):
    sidelayer1_weights[i, 2*i] = 1
    sidelayer1_weights[i, 2*i+1] = 1

for i in range(0, 32):
    sidelayer2_weights[2*i, i] = 1
    sidelayer2_weights[2*i+1, i] = -2

for i in range(0, 32):
    sidelayer1_bias[2*i+1] = -0.5

# 四个网络都用同一个1-1

for i in range(0, 32):
    layer1_weights[i, i] = 1

for i in range(0, 32):
    shisasum_weights[i] = i * 8
# 最大是55*46

for j in range(7, 12):
    for k in range(7, 12):
        m = j * 8
        n = k * 8
        target_pixel11 = iml[m, n,]
        target_pixel12 = iml[m, n + 1,]
        target_pixel13 = iml[m, n + 2,]
        target_pixel14 = iml[m, n + 3,]
        target_pixel15 = iml[m, n + 4,]
        target_pixel16 = iml[m, n + 5,]
        target_pixel17 = iml[m, n + 6,]
        target_pixel18 = iml[m, n + 7,]

        target_pixel21 = iml[m + 1, n,]
        target_pixel22 = iml[m + 1, n + 1,]
        target_pixel23 = iml[m + 1, n + 2,]
        target_pixel24 = iml[m + 1, n + 3,]
        target_pixel25 = iml[m + 1, n + 4,]
        target_pixel26 = iml[m + 1, n + 5,]
        target_pixel27 = iml[m + 1, n + 6,]
        target_pixel28 = iml[m + 1, n + 7,]

        target_pixel31 = iml[m + 2, n,]
        target_pixel32 = iml[m + 2, n + 1,]
        target_pixel33 = iml[m + 2, n + 2,]
        target_pixel34 = iml[m + 2, n + 3,]
        target_pixel35 = iml[m + 2, n + 4,]
        target_pixel36 = iml[m + 2, n + 5,]
        target_pixel37 = iml[m + 2, n + 6,]
        target_pixel38 = iml[m + 2, n + 7,]

        target_pixel41 = iml[m + 3, n,]
        target_pixel42 = iml[m + 3, n + 1,]
        target_pixel43 = iml[m + 3, n + 2,]
        target_pixel44 = iml[m + 3, n + 3,]
        target_pixel45 = iml[m + 3, n + 4,]
        target_pixel46 = iml[m + 3, n + 5,]
        target_pixel47 = iml[m + 3, n + 6,]
        target_pixel48 = iml[m + 3, n + 7,]

        target_pixel51 = iml[m + 4, n,]
        target_pixel52 = iml[m + 4, n + 1,]
        target_pixel53 = iml[m + 4, n + 2,]
        target_pixel54 = iml[m + 4, n + 3,]
        target_pixel55 = iml[m + 4, n + 4,]
        target_pixel56 = iml[m + 4, n + 5,]
        target_pixel57 = iml[m + 4, n + 6,]
        target_pixel58 = iml[m + 4, n + 7,]

        target_pixel61 = iml[m + 5, n,]
        target_pixel62 = iml[m + 5, n + 1,]
        target_pixel63 = iml[m + 5, n + 2,]
        target_pixel64 = iml[m + 5, n + 3,]
        target_pixel65 = iml[m + 5, n + 4,]
        target_pixel66 = iml[m + 5, n + 5,]
        target_pixel67 = iml[m + 5, n + 6,]
        target_pixel68 = iml[m + 5, n + 7,]

        target_pixel71 = iml[m + 6, n,]
        target_pixel72 = iml[m + 6, n + 1,]
        target_pixel73 = iml[m + 6, n + 2,]
        target_pixel74 = iml[m + 6, n + 3,]
        target_pixel75 = iml[m + 6, n + 4,]
        target_pixel76 = iml[m + 6, n + 5,]
        target_pixel77 = iml[m + 6, n + 6,]
        target_pixel78 = iml[m + 6, n + 7,]

        target_pixel81 = iml[m + 7, n,]
        target_pixel82 = iml[m + 7, n + 1,]
        target_pixel83 = iml[m + 7, n + 2,]
        target_pixel84 = iml[m + 7, n + 3,]
        target_pixel85 = iml[m + 7, n + 4,]
        target_pixel86 = iml[m + 7, n + 5,]
        target_pixel87 = iml[m + 7, n + 6,]
        target_pixel88 = iml[m + 7, n + 7,]

        for i in range(0, 32):
            layer112_weights[i,] = imr[m, n - i]
            layer122_weights[i,] = imr[m, n + 1 - i]
            layer132_weights[i,] = imr[m, n + 2 - i]
            layer142_weights[i,] = imr[m, n + 3 - i]
            layer152_weights[i,] = imr[m, n + 4 - i]
            layer162_weights[i,] = imr[m, n + 5 - i]
            layer172_weights[i,] = imr[m, n + 6 - i]
            layer182_weights[i,] = imr[m, n + 7 - i]

            layer212_weights[i,] = imr[m + 1, n - i]
            layer222_weights[i,] = imr[m + 1, n + 1 - i]
            layer232_weights[i,] = imr[m + 1, n + 2 - i]
            layer242_weights[i,] = imr[m + 1, n + 3 - i]
            layer252_weights[i,] = imr[m + 1, n + 4 - i]
            layer262_weights[i,] = imr[m + 1, n + 5 - i]
            layer272_weights[i,] = imr[m + 1, n + 6 - i]
            layer282_weights[i,] = imr[m + 1, n + 7 - i]

            layer312_weights[i,] = imr[m + 2, n - i]
            layer322_weights[i,] = imr[m + 2, n + 1 - i]
            layer332_weights[i,] = imr[m + 2, n + 2 - i]
            layer342_weights[i,] = imr[m + 2, n + 3 - i]
            layer352_weights[i,] = imr[m + 2, n + 4 - i]
            layer362_weights[i,] = imr[m + 2, n + 5 - i]
            layer372_weights[i,] = imr[m + 2, n + 6 - i]
            layer382_weights[i,] = imr[m + 2, n + 7 - i]

            layer412_weights[i,] = imr[m + 3, n - i]
            layer422_weights[i,] = imr[m + 3, n + 1 - i]
            layer432_weights[i,] = imr[m + 3, n + 2 - i]
            layer442_weights[i,] = imr[m + 3, n + 3 - i]
            layer452_weights[i,] = imr[m + 3, n + 4 - i]
            layer462_weights[i,] = imr[m + 3, n + 5 - i]
            layer472_weights[i,] = imr[m + 3, n + 6 - i]
            layer482_weights[i,] = imr[m + 3, n + 7 - i]

            layer512_weights[i,] = imr[m + 4, n - i]
            layer522_weights[i,] = imr[m + 4, n + 1 - i]
            layer532_weights[i,] = imr[m + 4, n + 2 - i]
            layer542_weights[i,] = imr[m + 4, n + 3 - i]
            layer552_weights[i,] = imr[m + 4, n + 4 - i]
            layer562_weights[i,] = imr[m + 4, n + 5 - i]
            layer572_weights[i,] = imr[m + 4, n + 6 - i]
            layer582_weights[i,] = imr[m + 4, n + 7 - i]

            layer612_weights[i,] = imr[m + 5, n - i]
            layer622_weights[i,] = imr[m + 5, n + 1 - i]
            layer632_weights[i,] = imr[m + 5, n + 2 - i]
            layer642_weights[i,] = imr[m + 5, n + 3 - i]
            layer652_weights[i,] = imr[m + 5, n + 4 - i]
            layer662_weights[i,] = imr[m + 5, n + 5 - i]
            layer672_weights[i,] = imr[m + 5, n + 6 - i]
            layer682_weights[i,] = imr[m + 5, n + 7 - i]

            layer712_weights[i,] = imr[m + 6, n - i]
            layer722_weights[i,] = imr[m + 6, n + 1 - i]
            layer732_weights[i,] = imr[m + 6, n + 2 - i]
            layer742_weights[i,] = imr[m + 6, n + 3 - i]
            layer752_weights[i,] = imr[m + 6, n + 4 - i]
            layer762_weights[i,] = imr[m + 6, n + 5 - i]
            layer772_weights[i,] = imr[m + 6, n + 6 - i]
            layer782_weights[i,] = imr[m + 6, n + 7 - i]

            layer812_weights[i,] = imr[m + 7, n - i]
            layer822_weights[i,] = imr[m + 7, n + 1 - i]
            layer832_weights[i,] = imr[m + 7, n + 2 - i]
            layer842_weights[i,] = imr[m + 7, n + 3 - i]
            layer852_weights[i,] = imr[m + 7, n + 4 - i]
            layer862_weights[i,] = imr[m + 7, n + 5 - i]
            layer872_weights[i,] = imr[m + 7, n + 6 - i]
            layer882_weights[i,] = imr[m + 7, n + 7 - i]

        for i in range(0, kaisu):
            target_pixel11_100[i,] = target_pixel11
            target_pixel12_100[i,] = target_pixel12
            target_pixel13_100[i,] = target_pixel13
            target_pixel14_100[i,] = target_pixel14
            target_pixel15_100[i,] = target_pixel15
            target_pixel16_100[i,] = target_pixel16
            target_pixel17_100[i,] = target_pixel17
            target_pixel18_100[i,] = target_pixel18

            target_pixel21_100[i,] = target_pixel21
            target_pixel22_100[i,] = target_pixel22
            target_pixel23_100[i,] = target_pixel23
            target_pixel24_100[i,] = target_pixel24
            target_pixel25_100[i,] = target_pixel25
            target_pixel26_100[i,] = target_pixel26
            target_pixel27_100[i,] = target_pixel27
            target_pixel28_100[i,] = target_pixel28

            target_pixel31_100[i,] = target_pixel31
            target_pixel32_100[i,] = target_pixel32
            target_pixel33_100[i,] = target_pixel33
            target_pixel34_100[i,] = target_pixel34
            target_pixel35_100[i,] = target_pixel35
            target_pixel36_100[i,] = target_pixel36
            target_pixel37_100[i,] = target_pixel37
            target_pixel38_100[i,] = target_pixel38

            target_pixel41_100[i,] = target_pixel41
            target_pixel42_100[i,] = target_pixel42
            target_pixel43_100[i,] = target_pixel43
            target_pixel44_100[i,] = target_pixel44
            target_pixel45_100[i,] = target_pixel45
            target_pixel46_100[i,] = target_pixel46
            target_pixel47_100[i,] = target_pixel47
            target_pixel48_100[i,] = target_pixel48

            target_pixel51_100[i,] = target_pixel51
            target_pixel52_100[i,] = target_pixel52
            target_pixel53_100[i,] = target_pixel53
            target_pixel54_100[i,] = target_pixel54
            target_pixel55_100[i,] = target_pixel55
            target_pixel56_100[i,] = target_pixel56
            target_pixel57_100[i,] = target_pixel57
            target_pixel58_100[i,] = target_pixel58

            target_pixel61_100[i,] = target_pixel61
            target_pixel62_100[i,] = target_pixel62
            target_pixel63_100[i,] = target_pixel63
            target_pixel64_100[i,] = target_pixel64
            target_pixel65_100[i,] = target_pixel65
            target_pixel66_100[i,] = target_pixel66
            target_pixel67_100[i,] = target_pixel67
            target_pixel68_100[i,] = target_pixel68

            target_pixel71_100[i,] = target_pixel71
            target_pixel72_100[i,] = target_pixel72
            target_pixel73_100[i,] = target_pixel73
            target_pixel74_100[i,] = target_pixel74
            target_pixel75_100[i,] = target_pixel75
            target_pixel76_100[i,] = target_pixel76
            target_pixel77_100[i,] = target_pixel77
            target_pixel78_100[i,] = target_pixel78

            target_pixel81_100[i,] = target_pixel81
            target_pixel82_100[i,] = target_pixel82
            target_pixel83_100[i,] = target_pixel83
            target_pixel84_100[i,] = target_pixel84
            target_pixel85_100[i,] = target_pixel85
            target_pixel86_100[i,] = target_pixel86
            target_pixel87_100[i,] = target_pixel87
            target_pixel88_100[i,] = target_pixel88

        # 11
        main_input11 = Input(shape=(1,), name='main_input11')
        main_layer111 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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
        main_input12 = Input(shape=(1,), name='main_input12')
        main_layer121 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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
        main_input13 = Input(shape=(1,), name='main_input13')
        main_layer131 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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
        main_input14 = Input(shape=(1,), name='main_input14')
        main_layer141 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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

        # 15
        main_input15 = Input(shape=(1,), name='main_input15')
        main_layer151 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer151',
                              activation='softmax')(main_input15)
        main_layer152 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer152',
                              activation='linear')(main_layer151)
        main_layer153 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer152_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer153',
                              activation='linear')(main_layer152)
        side_layer151 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer151',
                              activation='relu')(main_layer152)
        side_layer152 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer152',
                              activation='linear')(side_layer151)
        side_layer153 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer153',
                              activation='linear')(side_layer152)
        # 15

        # 16
        main_input16 = Input(shape=(1,), name='main_input16')
        main_layer161 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer161',
                              activation='softmax')(main_input16)
        main_layer162 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer162',
                              activation='linear')(main_layer161)
        main_layer163 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer162_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer163',
                              activation='linear')(main_layer162)
        side_layer161 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer161',
                              activation='relu')(main_layer162)
        side_layer162 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer162',
                              activation='linear')(side_layer161)
        side_layer163 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer163',
                              activation='linear')(side_layer162)
        # 16

        # 17
        main_input17 = Input(shape=(1,), name='main_input17')
        main_layer171 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer171',
                              activation='softmax')(main_input17)
        main_layer172 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer172',
                              activation='linear')(main_layer171)
        main_layer173 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer172_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer173',
                              activation='linear')(main_layer172)
        side_layer171 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer171',
                              activation='relu')(main_layer172)
        side_layer172 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer172',
                              activation='linear')(side_layer171)
        side_layer173 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer173',
                              activation='linear')(side_layer172)
        # 17

        # 18
        main_input18 = Input(shape=(1,), name='main_input18')
        main_layer181 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer181',
                              activation='softmax')(main_input18)
        main_layer182 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer182',
                              activation='linear')(main_layer181)
        main_layer183 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer182_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer183',
                              activation='linear')(main_layer182)
        side_layer181 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer181',
                              activation='relu')(main_layer182)
        side_layer182 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer182',
                              activation='linear')(side_layer181)
        side_layer183 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer183',
                              activation='linear')(side_layer182)
        # 18

        # 21
        main_input21 = Input(shape=(1,), name='main_input21')
        main_layer211 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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
        main_input22 = Input(shape=(1,), name='main_input22')
        main_layer221 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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
        main_input23 = Input(shape=(1,), name='main_input23')
        main_layer231 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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
        main_input24 = Input(shape=(1,), name='main_input24')
        main_layer241 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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

        # 25
        main_input25 = Input(shape=(1,), name='main_input25')
        main_layer251 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer251',
                              activation='softmax')(main_input25)
        main_layer252 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer252',
                              activation='linear')(main_layer251)
        main_layer253 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer252_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer253',
                              activation='linear')(main_layer252)
        side_layer251 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer251',
                              activation='relu')(main_layer252)
        side_layer252 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer252',
                              activation='linear')(side_layer251)
        side_layer253 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer253',
                              activation='linear')(side_layer252)
        # 25

        # 26
        main_input26 = Input(shape=(1,), name='main_input26')
        main_layer261 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer261',
                              activation='softmax')(main_input26)
        main_layer262 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer262',
                              activation='linear')(main_layer261)
        main_layer263 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer262_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer263',
                              activation='linear')(main_layer262)
        side_layer261 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer261',
                              activation='relu')(main_layer262)
        side_layer262 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer262',
                              activation='linear')(side_layer261)
        side_layer263 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer263',
                              activation='linear')(side_layer262)
        # 26

        # 27
        main_input27 = Input(shape=(1,), name='main_input27')
        main_layer271 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer271',
                              activation='softmax')(main_input27)
        main_layer272 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer272',
                              activation='linear')(main_layer271)
        main_layer273 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer272_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer273',
                              activation='linear')(main_layer272)
        side_layer271 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer271',
                              activation='relu')(main_layer272)
        side_layer272 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer272',
                              activation='linear')(side_layer271)
        side_layer273 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer273',
                              activation='linear')(side_layer272)
        # 27

        # 28
        main_input28 = Input(shape=(1,), name='main_input28')
        main_layer281 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer281',
                              activation='softmax')(main_input28)
        main_layer282 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer282',
                              activation='linear')(main_layer281)
        main_layer283 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer282_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer283',
                              activation='linear')(main_layer282)
        side_layer281 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer281',
                              activation='relu')(main_layer282)
        side_layer282 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer282',
                              activation='linear')(side_layer281)
        side_layer283 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer283',
                              activation='linear')(side_layer282)
        # 28

        # 31
        main_input31 = Input(shape=(1,), name='main_input31')
        main_layer311 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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
        main_input32 = Input(shape=(1,), name='main_input32')
        main_layer321 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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
        main_input33 = Input(shape=(1,), name='main_input33')
        main_layer331 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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
        main_input34 = Input(shape=(1,), name='main_input34')
        main_layer341 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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

        # 35
        main_input35 = Input(shape=(1,), name='main_input35')
        main_layer351 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer351',
                              activation='softmax')(main_input35)
        main_layer352 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer352',
                              activation='linear')(main_layer351)
        main_layer353 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer352_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer353',
                              activation='linear')(main_layer352)
        side_layer351 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer351',
                              activation='relu')(main_layer352)
        side_layer352 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer352',
                              activation='linear')(side_layer351)
        side_layer353 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer353',
                              activation='linear')(side_layer352)
        # 35

        # 36
        main_input36 = Input(shape=(1,), name='main_input36')
        main_layer361 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer361',
                              activation='softmax')(main_input36)
        main_layer362 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer362',
                              activation='linear')(main_layer361)
        main_layer363 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer362_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer363',
                              activation='linear')(main_layer362)
        side_layer361 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer361',
                              activation='relu')(main_layer362)
        side_layer362 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer362',
                              activation='linear')(side_layer361)
        side_layer363 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer363',
                              activation='linear')(side_layer362)
        # 36

        # 37
        main_input37 = Input(shape=(1,), name='main_input37')
        main_layer371 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer371',
                              activation='softmax')(main_input37)
        main_layer372 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer372',
                              activation='linear')(main_layer371)
        main_layer373 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer372_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer373',
                              activation='linear')(main_layer372)
        side_layer371 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer371',
                              activation='relu')(main_layer372)
        side_layer372 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer372',
                              activation='linear')(side_layer371)
        side_layer373 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer373',
                              activation='linear')(side_layer372)
        # 37

        # 38
        main_input38 = Input(shape=(1,), name='main_input38')
        main_layer381 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer381',
                              activation='softmax')(main_input38)
        main_layer382 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer382',
                              activation='linear')(main_layer381)
        main_layer383 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer382_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer383',
                              activation='linear')(main_layer382)
        side_layer381 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer381',
                              activation='relu')(main_layer382)
        side_layer382 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer382',
                              activation='linear')(side_layer381)
        side_layer383 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer383',
                              activation='linear')(side_layer382)
        # 38

        # 41
        main_input41 = Input(shape=(1,), name='main_input41')
        main_layer411 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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
        main_input42 = Input(shape=(1,), name='main_input42')
        main_layer421 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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
        main_input43 = Input(shape=(1,), name='main_input43')
        main_layer431 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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
        main_input44 = Input(shape=(1,), name='main_input44')
        main_layer441 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
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

        # 45
        main_input45 = Input(shape=(1,), name='main_input45')
        main_layer451 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer451',
                              activation='softmax')(main_input45)
        main_layer452 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer452',
                              activation='linear')(main_layer451)
        main_layer453 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer452_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer453',
                              activation='linear')(main_layer452)
        side_layer451 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer451',
                              activation='relu')(main_layer452)
        side_layer452 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer452',
                              activation='linear')(side_layer451)
        side_layer453 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer453',
                              activation='linear')(side_layer452)
        # 45

        # 46
        main_input46 = Input(shape=(1,), name='main_input46')
        main_layer461 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer461',
                              activation='softmax')(main_input46)
        main_layer462 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer462',
                              activation='linear')(main_layer461)
        main_layer463 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer462_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer463',
                              activation='linear')(main_layer462)
        side_layer461 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer461',
                              activation='relu')(main_layer462)
        side_layer462 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer462',
                              activation='linear')(side_layer461)
        side_layer463 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer463',
                              activation='linear')(side_layer462)
        # 46

        # 47
        main_input47 = Input(shape=(1,), name='main_input47')
        main_layer471 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer471',
                              activation='softmax')(main_input47)
        main_layer472 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer472',
                              activation='linear')(main_layer471)
        main_layer473 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer472_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer473',
                              activation='linear')(main_layer472)
        side_layer471 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer471',
                              activation='relu')(main_layer472)
        side_layer472 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer472',
                              activation='linear')(side_layer471)
        side_layer473 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer473',
                              activation='linear')(side_layer472)
        # 47

        # 48
        main_input48 = Input(shape=(1,), name='main_input48')
        main_layer481 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer481',
                              activation='softmax')(main_input48)
        main_layer482 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer482',
                              activation='linear')(main_layer481)
        main_layer483 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer482_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer483',
                              activation='linear')(main_layer482)
        side_layer481 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer481',
                              activation='relu')(main_layer482)
        side_layer482 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer482',
                              activation='linear')(side_layer481)
        side_layer483 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer483',
                              activation='linear')(side_layer482)
        # 48

        # 51
        main_input51 = Input(shape=(1,), name='main_input51')
        main_layer511 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer511',
                              activation='softmax')(main_input51)
        main_layer512 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer512',
                              activation='linear')(main_layer511)
        main_layer513 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer512_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer513',
                              activation='linear')(main_layer512)
        side_layer511 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer511',
                              activation='relu')(main_layer512)
        side_layer512 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer512',
                              activation='linear')(side_layer511)
        side_layer513 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer513',
                              activation='linear')(side_layer512)
        # 51

        # 52
        main_input52 = Input(shape=(1,), name='main_input52')
        main_layer521 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer521',
                              activation='softmax')(main_input52)
        main_layer522 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer522',
                              activation='linear')(main_layer521)
        main_layer523 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer522_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer523',
                              activation='linear')(main_layer522)
        side_layer521 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer521',
                              activation='relu')(main_layer522)
        side_layer522 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer522',
                              activation='linear')(side_layer521)
        side_layer523 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer523',
                              activation='linear')(side_layer522)
        # 52

        # 53
        main_input53 = Input(shape=(1,), name='main_input53')
        main_layer531 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer531',
                              activation='softmax')(main_input53)
        main_layer532 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer532',
                              activation='linear')(main_layer531)
        main_layer533 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer532_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer533',
                              activation='linear')(main_layer532)
        side_layer531 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer531',
                              activation='relu')(main_layer532)
        side_layer532 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer532',
                              activation='linear')(side_layer531)
        side_layer533 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer533',
                              activation='linear')(side_layer532)
        # 53

        # 54
        main_input54 = Input(shape=(1,), name='main_input54')
        main_layer541 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer541',
                              activation='softmax')(main_input54)
        main_layer542 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer542',
                              activation='linear')(main_layer541)
        main_layer543 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer542_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer543',
                              activation='linear')(main_layer542)
        side_layer541 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer541',
                              activation='relu')(main_layer542)
        side_layer542 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer542',
                              activation='linear')(side_layer541)
        side_layer543 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer543',
                              activation='linear')(side_layer542)
        # 54

        # 55
        main_input55 = Input(shape=(1,), name='main_input55')
        main_layer551 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer551',
                              activation='softmax')(main_input55)
        main_layer552 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer552',
                              activation='linear')(main_layer551)
        main_layer553 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer552_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer553',
                              activation='linear')(main_layer552)
        side_layer551 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer551',
                              activation='relu')(main_layer552)
        side_layer552 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer552',
                              activation='linear')(side_layer551)
        side_layer553 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer553',
                              activation='linear')(side_layer552)
        # 55

        # 56
        main_input56 = Input(shape=(1,), name='main_input56')
        main_layer561 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer561',
                              activation='softmax')(main_input56)
        main_layer562 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer562',
                              activation='linear')(main_layer561)
        main_layer563 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer562_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer563',
                              activation='linear')(main_layer562)
        side_layer561 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer561',
                              activation='relu')(main_layer562)
        side_layer562 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer562',
                              activation='linear')(side_layer561)
        side_layer563 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer563',
                              activation='linear')(side_layer562)
        # 56

        # 57
        main_input57 = Input(shape=(1,), name='main_input57')
        main_layer571 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer571',
                              activation='softmax')(main_input57)
        main_layer572 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer572',
                              activation='linear')(main_layer571)
        main_layer573 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer572_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer573',
                              activation='linear')(main_layer572)
        side_layer571 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer571',
                              activation='relu')(main_layer572)
        side_layer572 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer572',
                              activation='linear')(side_layer571)
        side_layer573 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer573',
                              activation='linear')(side_layer572)
        # 57

        # 58
        main_input58 = Input(shape=(1,), name='main_input58')
        main_layer581 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer581',
                              activation='softmax')(main_input58)
        main_layer582 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer582',
                              activation='linear')(main_layer581)
        main_layer583 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer582_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer583',
                              activation='linear')(main_layer582)
        side_layer581 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer581',
                              activation='relu')(main_layer582)
        side_layer582 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer582',
                              activation='linear')(side_layer581)
        side_layer583 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer583',
                              activation='linear')(side_layer582)
        # 58

        # 61
        main_input61 = Input(shape=(1,), name='main_input61')
        main_layer611 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer611',
                              activation='softmax')(main_input61)
        main_layer612 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer612',
                              activation='linear')(main_layer611)
        main_layer613 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer612_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer613',
                              activation='linear')(main_layer612)
        side_layer611 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer611',
                              activation='relu')(main_layer612)
        side_layer612 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer612',
                              activation='linear')(side_layer611)
        side_layer613 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer613',
                              activation='linear')(side_layer612)
        # 61

        # 62
        main_input62 = Input(shape=(1,), name='main_input62')
        main_layer621 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer621',
                              activation='softmax')(main_input62)
        main_layer622 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer622',
                              activation='linear')(main_layer621)
        main_layer623 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer622_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer623',
                              activation='linear')(main_layer622)
        side_layer621 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer621',
                              activation='relu')(main_layer622)
        side_layer622 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer622',
                              activation='linear')(side_layer621)
        side_layer623 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer623',
                              activation='linear')(side_layer622)
        # 62

        # 63
        main_input63 = Input(shape=(1,), name='main_input63')
        main_layer631 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer631',
                              activation='softmax')(main_input63)
        main_layer632 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer632',
                              activation='linear')(main_layer631)
        main_layer633 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer632_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer633',
                              activation='linear')(main_layer632)
        side_layer631 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer631',
                              activation='relu')(main_layer632)
        side_layer632 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer632',
                              activation='linear')(side_layer631)
        side_layer633 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer633',
                              activation='linear')(side_layer632)
        # 63

        # 64
        main_input64 = Input(shape=(1,), name='main_input64')
        main_layer641 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer641',
                              activation='softmax')(main_input64)
        main_layer642 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer642',
                              activation='linear')(main_layer641)
        main_layer643 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer642_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer643',
                              activation='linear')(main_layer642)
        side_layer641 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer641',
                              activation='relu')(main_layer642)
        side_layer642 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer642',
                              activation='linear')(side_layer641)
        side_layer643 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer643',
                              activation='linear')(side_layer642)
        # 64

        # 65
        main_input65 = Input(shape=(1,), name='main_input65')
        main_layer651 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer651',
                              activation='softmax')(main_input65)
        main_layer652 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer652',
                              activation='linear')(main_layer651)
        main_layer653 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer652_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer653',
                              activation='linear')(main_layer652)
        side_layer651 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer651',
                              activation='relu')(main_layer652)
        side_layer652 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer652',
                              activation='linear')(side_layer651)
        side_layer653 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer653',
                              activation='linear')(side_layer652)
        # 65

        # 66
        main_input66 = Input(shape=(1,), name='main_input66')
        main_layer661 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer661',
                              activation='softmax')(main_input66)
        main_layer662 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer662',
                              activation='linear')(main_layer661)
        main_layer663 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer662_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer663',
                              activation='linear')(main_layer662)
        side_layer661 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer661',
                              activation='relu')(main_layer662)
        side_layer662 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer662',
                              activation='linear')(side_layer661)
        side_layer663 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer663',
                              activation='linear')(side_layer662)
        # 66

        # 67
        main_input67 = Input(shape=(1,), name='main_input67')
        main_layer671 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer671',
                              activation='softmax')(main_input67)
        main_layer672 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer672',
                              activation='linear')(main_layer671)
        main_layer673 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer672_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer673',
                              activation='linear')(main_layer672)
        side_layer671 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer671',
                              activation='relu')(main_layer672)
        side_layer672 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer672',
                              activation='linear')(side_layer671)
        side_layer673 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer673',
                              activation='linear')(side_layer672)
        # 67

        # 68
        main_input68 = Input(shape=(1,), name='main_input68')
        main_layer681 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer681',
                              activation='softmax')(main_input68)
        main_layer682 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer682',
                              activation='linear')(main_layer681)
        main_layer683 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer682_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer683',
                              activation='linear')(main_layer682)
        side_layer681 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer681',
                              activation='relu')(main_layer682)
        side_layer682 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer682',
                              activation='linear')(side_layer681)
        side_layer683 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer683',
                              activation='linear')(side_layer682)
        # 68

        # 71
        main_input71 = Input(shape=(1,), name='main_input71')
        main_layer711 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer711',
                              activation='softmax')(main_input71)
        main_layer712 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer712',
                              activation='linear')(main_layer711)
        main_layer713 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer712_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer713',
                              activation='linear')(main_layer712)
        side_layer711 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer711',
                              activation='relu')(main_layer712)
        side_layer712 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer712',
                              activation='linear')(side_layer711)
        side_layer713 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer713',
                              activation='linear')(side_layer712)
        # 71

        # 72
        main_input72 = Input(shape=(1,), name='main_input72')
        main_layer721 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer721',
                              activation='softmax')(main_input72)
        main_layer722 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer722',
                              activation='linear')(main_layer721)
        main_layer723 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer722_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer723',
                              activation='linear')(main_layer722)
        side_layer721 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer721',
                              activation='relu')(main_layer722)
        side_layer722 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer722',
                              activation='linear')(side_layer721)
        side_layer723 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer723',
                              activation='linear')(side_layer722)
        # 72

        # 73
        main_input73 = Input(shape=(1,), name='main_input73')
        main_layer731 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer731',
                              activation='softmax')(main_input73)
        main_layer732 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer732',
                              activation='linear')(main_layer731)
        main_layer733 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer732_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer733',
                              activation='linear')(main_layer732)
        side_layer731 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer731',
                              activation='relu')(main_layer732)
        side_layer732 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer732',
                              activation='linear')(side_layer731)
        side_layer733 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer733',
                              activation='linear')(side_layer732)
        # 73

        # 74
        main_input74 = Input(shape=(1,), name='main_input74')
        main_layer741 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer741',
                              activation='softmax')(main_input74)
        main_layer742 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer742',
                              activation='linear')(main_layer741)
        main_layer743 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer742_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer743',
                              activation='linear')(main_layer742)
        side_layer741 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer741',
                              activation='relu')(main_layer742)
        side_layer742 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer742',
                              activation='linear')(side_layer741)
        side_layer743 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer743',
                              activation='linear')(side_layer742)
        # 74

        # 75
        main_input75 = Input(shape=(1,), name='main_input75')
        main_layer751 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer751',
                              activation='softmax')(main_input75)
        main_layer752 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer752',
                              activation='linear')(main_layer751)
        main_layer753 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer752_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer753',
                              activation='linear')(main_layer752)
        side_layer751 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer751',
                              activation='relu')(main_layer752)
        side_layer752 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer752',
                              activation='linear')(side_layer751)
        side_layer753 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer753',
                              activation='linear')(side_layer752)
        # 75

        # 76
        main_input76 = Input(shape=(1,), name='main_input76')
        main_layer761 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer761',
                              activation='softmax')(main_input76)
        main_layer762 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer762',
                              activation='linear')(main_layer761)
        main_layer763 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer762_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer763',
                              activation='linear')(main_layer762)
        side_layer761 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer761',
                              activation='relu')(main_layer762)
        side_layer762 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer762',
                              activation='linear')(side_layer761)
        side_layer763 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer763',
                              activation='linear')(side_layer762)
        # 76

        # 77
        main_input77 = Input(shape=(1,), name='main_input77')
        main_layer771 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer771',
                              activation='softmax')(main_input77)
        main_layer772 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer772',
                              activation='linear')(main_layer771)
        main_layer773 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer772_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer773',
                              activation='linear')(main_layer772)
        side_layer771 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer771',
                              activation='relu')(main_layer772)
        side_layer772 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer772',
                              activation='linear')(side_layer771)
        side_layer773 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer773',
                              activation='linear')(side_layer772)
        # 77

        # 78
        main_input78 = Input(shape=(1,), name='main_input78')
        main_layer781 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer781',
                              activation='softmax')(main_input78)
        main_layer782 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer782',
                              activation='linear')(main_layer781)
        main_layer783 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer782_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer783',
                              activation='linear')(main_layer782)
        side_layer781 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer781',
                              activation='relu')(main_layer782)
        side_layer782 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer782',
                              activation='linear')(side_layer781)
        side_layer783 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer783',
                              activation='linear')(side_layer782)
        # 78

        # 81
        main_input81 = Input(shape=(1,), name='main_input81')
        main_layer811 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer811',
                              activation='softmax')(main_input81)
        main_layer812 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer812',
                              activation='linear')(main_layer811)
        main_layer813 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer812_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer813',
                              activation='linear')(main_layer812)
        side_layer811 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer811',
                              activation='relu')(main_layer812)
        side_layer812 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer812',
                              activation='linear')(side_layer811)
        side_layer813 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer813',
                              activation='linear')(side_layer812)
        # 81

        # 82
        main_input82 = Input(shape=(1,), name='main_input82')
        main_layer821 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer821',
                              activation='softmax')(main_input82)
        main_layer822 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer822',
                              activation='linear')(main_layer821)
        main_layer823 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer822_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer823',
                              activation='linear')(main_layer822)
        side_layer821 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer821',
                              activation='relu')(main_layer822)
        side_layer822 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer822',
                              activation='linear')(side_layer821)
        side_layer823 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer823',
                              activation='linear')(side_layer822)
        # 82

        # 83
        main_input83 = Input(shape=(1,), name='main_input83')
        main_layer831 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer831',
                              activation='softmax')(main_input83)
        main_layer832 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer832',
                              activation='linear')(main_layer831)
        main_layer833 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer832_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer833',
                              activation='linear')(main_layer832)
        side_layer831 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer831',
                              activation='relu')(main_layer832)
        side_layer832 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer832',
                              activation='linear')(side_layer831)
        side_layer833 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer833',
                              activation='linear')(side_layer832)
        # 83

        # 84
        main_input84 = Input(shape=(1,), name='main_input84')
        main_layer841 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer841',
                              activation='softmax')(main_input84)
        main_layer842 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer842',
                              activation='linear')(main_layer841)
        main_layer843 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer842_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer843',
                              activation='linear')(main_layer842)
        side_layer841 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer841',
                              activation='relu')(main_layer842)
        side_layer842 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer842',
                              activation='linear')(side_layer841)
        side_layer843 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer843',
                              activation='linear')(side_layer842)
        # 84

        # 85
        main_input85 = Input(shape=(1,), name='main_input85')
        main_layer851 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer851',
                              activation='softmax')(main_input85)
        main_layer852 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer852',
                              activation='linear')(main_layer851)
        main_layer853 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer852_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer853',
                              activation='linear')(main_layer852)
        side_layer851 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer851',
                              activation='relu')(main_layer852)
        side_layer852 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer852',
                              activation='linear')(side_layer851)
        side_layer853 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer853',
                              activation='linear')(side_layer852)
        # 85

        # 86
        main_input86 = Input(shape=(1,), name='main_input86')
        main_layer861 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer861',
                              activation='softmax')(main_input86)
        main_layer862 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer862',
                              activation='linear')(main_layer861)
        main_layer863 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer862_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer863',
                              activation='linear')(main_layer862)
        side_layer861 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer861',
                              activation='relu')(main_layer862)
        side_layer862 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer862',
                              activation='linear')(side_layer861)
        side_layer863 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer863',
                              activation='linear')(side_layer862)
        # 86

        # 87
        main_input87 = Input(shape=(1,), name='main_input87')
        main_layer871 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer871',
                              activation='softmax')(main_input87)
        main_layer872 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer872',
                              activation='linear')(main_layer871)
        main_layer873 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer872_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer873',
                              activation='linear')(main_layer872)
        side_layer871 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer871',
                              activation='relu')(main_layer872)
        side_layer872 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer872',
                              activation='linear')(side_layer871)
        side_layer873 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer873',
                              activation='linear')(side_layer872)
        # 87

        # 88
        main_input88 = Input(shape=(1,), name='main_input88')
        main_layer881 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=probability),
                              use_bias=False,
                              input_shape=(1,),
                              name='main_layer881',
                              activation='softmax')(main_input88)
        main_layer882 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=layer1_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer882',
                              activation='linear')(main_layer881)
        main_layer883 = Dense(3,
                              kernel_initializer=keras.initializers.constant(value=layer882_weights),
                              use_bias=False,
                              trainable=False,
                              name='main_layer883',
                              activation='linear')(main_layer882)
        side_layer881 = Dense(64,
                              kernel_initializer=keras.initializers.constant(value=sidelayer1_weights),
                              use_bias=True,
                              trainable=False,
                              bias_initializer=keras.initializers.constant(value=sidelayer1_bias),
                              name='side_layer881',
                              activation='relu')(main_layer882)
        side_layer882 = Dense(32,
                              kernel_initializer=keras.initializers.constant(value=sidelayer2_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer882',
                              activation='linear')(side_layer881)
        side_layer883 = Dense(1,
                              kernel_initializer=keras.initializers.constant(value=sidelayer3_weights),
                              use_bias=False,
                              trainable=False,
                              name='side_layer883',
                              activation='linear')(side_layer882)
        # 88

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
        shisasum15 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum15',
                           activation='linear')(main_layer152)
        shisasum16 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum16',
                           activation='linear')(main_layer162)
        shisasum17 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum17',
                           activation='linear')(main_layer172)
        shisasum18 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum18',
                           activation='linear')(main_layer182)

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
        shisasum25 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum25',
                           activation='linear')(main_layer252)
        shisasum26 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum26',
                           activation='linear')(main_layer262)
        shisasum27 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum27',
                           activation='linear')(main_layer272)
        shisasum28 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum28',
                           activation='linear')(main_layer282)

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
        shisasum35 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum35',
                           activation='linear')(main_layer352)
        shisasum36 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum36',
                           activation='linear')(main_layer362)
        shisasum37 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum37',
                           activation='linear')(main_layer372)
        shisasum38 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum38',
                           activation='linear')(main_layer382)

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
        shisasum45 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum45',
                           activation='linear')(main_layer452)
        shisasum46 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum46',
                           activation='linear')(main_layer462)
        shisasum47 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum47',
                           activation='linear')(main_layer472)
        shisasum48 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum48',
                           activation='linear')(main_layer482)

        shisasum51 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum51',
                           activation='linear')(main_layer512)
        shisasum52 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum52',
                           activation='linear')(main_layer522)
        shisasum53 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum53',
                           activation='linear')(main_layer532)
        shisasum54 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum54',
                           activation='linear')(main_layer542)
        shisasum55 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum55',
                           activation='linear')(main_layer552)
        shisasum56 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum56',
                           activation='linear')(main_layer562)
        shisasum57 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum57',
                           activation='linear')(main_layer572)
        shisasum58 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum58',
                           activation='linear')(main_layer582)

        shisasum61 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum61',
                           activation='linear')(main_layer612)
        shisasum62 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum62',
                           activation='linear')(main_layer622)
        shisasum63 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum63',
                           activation='linear')(main_layer632)
        shisasum64 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum64',
                           activation='linear')(main_layer642)
        shisasum65 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum65',
                           activation='linear')(main_layer652)
        shisasum66 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum66',
                           activation='linear')(main_layer662)
        shisasum67 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum67',
                           activation='linear')(main_layer672)
        shisasum68 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum68',
                           activation='linear')(main_layer682)

        shisasum71 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum71',
                           activation='linear')(main_layer712)
        shisasum72 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum72',
                           activation='linear')(main_layer722)
        shisasum73 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum73',
                           activation='linear')(main_layer732)
        shisasum74 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum74',
                           activation='linear')(main_layer742)
        shisasum75 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum75',
                           activation='linear')(main_layer752)
        shisasum76 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum76',
                           activation='linear')(main_layer762)
        shisasum77 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum77',
                           activation='linear')(main_layer772)
        shisasum78 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum78',
                           activation='linear')(main_layer782)

        shisasum81 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum81',
                           activation='linear')(main_layer812)
        shisasum82 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum82',
                           activation='linear')(main_layer822)
        shisasum83 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum83',
                           activation='linear')(main_layer832)
        shisasum84 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum84',
                           activation='linear')(main_layer842)
        shisasum85 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum85',
                           activation='linear')(main_layer852)
        shisasum86 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum86',
                           activation='linear')(main_layer862)
        shisasum87 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum87',
                           activation='linear')(main_layer872)
        shisasum88 = Dense(1,
                           kernel_initializer=keras.initializers.constant(value=shisasum_weights),
                           use_bias=False,
                           trainable=False,
                           name='shisasum88',
                           activation='linear')(main_layer882)

        concatenated1 = Concatenate(axis=1, name='concatenated1')(
            [shisasum11, shisasum12, shisasum13, shisasum14, shisasum15, shisasum16, shisasum17, shisasum18,
             shisasum21, shisasum22, shisasum23, shisasum24, shisasum25, shisasum26, shisasum27, shisasum28,
             shisasum31, shisasum32, shisasum33, shisasum31, shisasum35, shisasum36, shisasum37, shisasum38,
             shisasum41, shisasum42, shisasum43, shisasum44, shisasum45, shisasum46, shisasum47, shisasum48,
             shisasum51, shisasum52, shisasum53, shisasum54, shisasum55, shisasum56, shisasum57, shisasum58,
             shisasum61, shisasum62, shisasum63, shisasum64, shisasum65, shisasum66, shisasum67, shisasum68,
             shisasum71, shisasum72, shisasum73, shisasum74, shisasum75, shisasum76, shisasum77, shisasum78,
             shisasum81, shisasum82, shisasum83, shisasum84, shisasum85, shisasum86, shisasum87, shisasum88, ])

        reshaped = Reshape(input_shape=(64, -1), target_shape=(1, 8, 8), name='reshaped')(concatenated1)
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

        concatenated2 = Concatenate(axis=1, name='concatenated2')(
            [wavelet_LL, wavelet_HH, wavelet_HL, wavelet_LH, wavelet_NHH, wavelet_NHL, wavelet_NLH, wavelet_NLL])

        flaten1 = Flatten()(concatenated2)

        wavedense = Dense(1,
                          kernel_initializer=keras.initializers.constant(value=wave),
                          use_bias=False,
                          input_shape=(1,),
                          trainable=False,
                          name='wavedense')(flaten1)

        model1 = Model(
            [main_input11, main_input12, main_input13, main_input14, main_input15, main_input16, main_input17,
             main_input18,
             main_input21, main_input22, main_input23, main_input24, main_input25, main_input26, main_input27,
             main_input28,
             main_input31, main_input32, main_input33, main_input34, main_input35, main_input36, main_input37,
             main_input38,
             main_input41, main_input42, main_input43, main_input44, main_input45, main_input46, main_input47,
             main_input48,
             main_input51, main_input52, main_input53, main_input54, main_input55, main_input56, main_input57,
             main_input58,
             main_input61, main_input62, main_input63, main_input64, main_input65, main_input66, main_input67,
             main_input68,
             main_input71, main_input72, main_input73, main_input74, main_input75, main_input76, main_input77,
             main_input78,
             main_input81, main_input82, main_input83, main_input84, main_input85, main_input86, main_input87,
             main_input88],
            [main_layer113, side_layer113, main_layer123, side_layer123, main_layer133, side_layer133, main_layer143,
             side_layer143, main_layer153, side_layer153, main_layer163, side_layer163, main_layer173, side_layer173,
             main_layer183, side_layer183,
             main_layer213, side_layer213, main_layer223, side_layer223, main_layer233, side_layer233, main_layer243,
             side_layer243, main_layer253, side_layer253, main_layer263, side_layer263, main_layer273, side_layer273,
             main_layer283, side_layer283,
             main_layer313, side_layer313, main_layer323, side_layer323, main_layer333, side_layer333, main_layer343,
             side_layer343, main_layer353, side_layer353, main_layer363, side_layer363, main_layer373, side_layer373,
             main_layer383, side_layer383,
             main_layer413, side_layer413, main_layer423, side_layer423, main_layer433, side_layer433, main_layer443,
             side_layer443, main_layer453, side_layer453, main_layer463, side_layer463, main_layer473, side_layer473,
             main_layer483, side_layer483,
             main_layer513, side_layer513, main_layer523, side_layer523, main_layer533, side_layer533, main_layer543,
             side_layer543, main_layer553, side_layer553, main_layer563, side_layer563, main_layer573, side_layer573,
             main_layer583, side_layer583,
             main_layer613, side_layer613, main_layer623, side_layer623, main_layer633, side_layer633, main_layer643,
             side_layer643, main_layer653, side_layer653, main_layer663, side_layer663, main_layer673, side_layer673,
             main_layer683, side_layer683,
             main_layer713, side_layer713, main_layer723, side_layer723, main_layer733, side_layer733, main_layer743,
             side_layer743, main_layer753, side_layer753, main_layer763, side_layer763, main_layer773, side_layer773,
             main_layer783, side_layer783,
             main_layer813, side_layer813, main_layer823, side_layer823, main_layer833, side_layer833, main_layer843,
             side_layer843, main_layer853, side_layer853, main_layer863, side_layer863, main_layer873, side_layer873,
             main_layer883, side_layer883, wavedense])

        # print(model1.get_weights())

        rMSprop = RMSprop(lr=1e-1)
        adadelta = Adadelta(lr=1e-1)
        adam = Adam(lr=1, decay=0)
        # entropy权重0
        model1.compile(optimizer=adam,
                       loss='mse',
                       metrics=['accuracy'],
                       loss_weights={'main_layer113': alpha, 'side_layer113': beta, 'main_layer123': alpha,
                                     'side_layer123': beta, 'main_layer133': alpha, 'side_layer133': beta,
                                     'main_layer143': alpha, 'side_layer143': beta, 'main_layer153': alpha,
                                     'side_layer153': beta, 'main_layer163': alpha, 'side_layer163': beta,
                                     'main_layer173': alpha, 'side_layer173': beta, 'main_layer183': alpha,
                                     'side_layer183': beta,
                                     'main_layer213': alpha, 'side_layer213': beta, 'main_layer223': alpha,
                                     'side_layer223': beta, 'main_layer233': alpha, 'side_layer233': beta,
                                     'main_layer243': alpha, 'side_layer243': beta, 'main_layer253': alpha,
                                     'side_layer253': beta, 'main_layer263': alpha, 'side_layer263': beta,
                                     'main_layer273': alpha, 'side_layer273': beta, 'main_layer283': alpha,
                                     'side_layer283': beta,
                                     'main_layer313': alpha, 'side_layer313': beta, 'main_layer323': alpha,
                                     'side_layer323': beta, 'main_layer333': alpha, 'side_layer333': beta,
                                     'main_layer343': alpha, 'side_layer343': beta, 'main_layer353': alpha,
                                     'side_layer353': beta, 'main_layer363': alpha, 'side_layer363': beta,
                                     'main_layer373': alpha, 'side_layer373': beta, 'main_layer383': alpha,
                                     'side_layer383': beta,
                                     'main_layer413': alpha, 'side_layer413': beta, 'main_layer423': alpha,
                                     'side_layer423': beta, 'main_layer433': alpha, 'side_layer433': beta,
                                     'main_layer443': alpha, 'side_layer443': beta, 'main_layer453': alpha,
                                     'side_layer453': beta, 'main_layer463': alpha, 'side_layer463': beta,
                                     'main_layer473': alpha, 'side_layer473': beta, 'main_layer483': alpha,
                                     'side_layer483': beta,
                                     'main_layer513': alpha, 'side_layer513': beta, 'main_layer523': alpha,
                                     'side_layer523': beta, 'main_layer533': alpha, 'side_layer533': beta,
                                     'main_layer543': alpha, 'side_layer543': beta, 'main_layer553': alpha,
                                     'side_layer553': beta, 'main_layer563': alpha, 'side_layer563': beta,
                                     'main_layer573': alpha, 'side_layer573': beta, 'main_layer583': alpha,
                                     'side_layer583': beta,
                                     'main_layer613': alpha, 'side_layer613': beta, 'main_layer623': alpha,
                                     'side_layer623': beta, 'main_layer633': alpha, 'side_layer633': beta,
                                     'main_layer643': alpha, 'side_layer643': beta, 'main_layer653': alpha,
                                     'side_layer653': beta, 'main_layer663': alpha, 'side_layer663': beta,
                                     'main_layer673': alpha, 'side_layer673': beta, 'main_layer683': alpha,
                                     'side_layer683': beta,
                                     'main_layer713': alpha, 'side_layer713': beta, 'main_layer723': alpha,
                                     'side_layer723': beta, 'main_layer733': alpha, 'side_layer733': beta,
                                     'main_layer743': alpha, 'side_layer743': beta, 'main_layer753': alpha,
                                     'side_layer753': beta, 'main_layer763': alpha, 'side_layer763': beta,
                                     'main_layer773': alpha, 'side_layer773': beta, 'main_layer783': alpha,
                                     'side_layer783': beta,
                                     'main_layer813': alpha, 'side_layer813': beta, 'main_layer823': alpha,
                                     'side_layer823': beta, 'main_layer833': alpha, 'side_layer833': beta,
                                     'main_layer843': alpha, 'side_layer843': beta, 'main_layer853': alpha,
                                     'side_layer853': beta, 'main_layer863': alpha, 'side_layer863': beta,
                                     'main_layer873': alpha, 'side_layer873': beta, 'main_layer883': alpha,
                                     'side_layer883': beta, 'wavedense': gamma}
                       )

        model1.fit([input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey,
                    input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey,
                    input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey,
                    input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey,
                    input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey,
                    input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey,
                    input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey,
                    input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, input_grey, ],
                   [target_pixel11_100, target_side_100, target_pixel12_100, target_side_100, target_pixel13_100,
                    target_side_100, target_pixel14_100, target_side_100, target_pixel15_100, target_side_100,
                    target_pixel16_100, target_side_100, target_pixel17_100, target_side_100, target_pixel18_100,
                    target_side_100,
                    target_pixel21_100, target_side_100, target_pixel22_100, target_side_100, target_pixel23_100,
                    target_side_100, target_pixel24_100, target_side_100, target_pixel25_100, target_side_100,
                    target_pixel26_100, target_side_100, target_pixel27_100, target_side_100, target_pixel28_100,
                    target_side_100,
                    target_pixel31_100, target_side_100, target_pixel32_100, target_side_100, target_pixel33_100,
                    target_side_100, target_pixel34_100, target_side_100, target_pixel35_100, target_side_100,
                    target_pixel36_100, target_side_100, target_pixel37_100, target_side_100, target_pixel38_100,
                    target_side_100,
                    target_pixel41_100, target_side_100, target_pixel42_100, target_side_100, target_pixel43_100,
                    target_side_100, target_pixel44_100, target_side_100, target_pixel45_100, target_side_100,
                    target_pixel46_100, target_side_100, target_pixel47_100, target_side_100, target_pixel48_100,
                    target_side_100,
                    target_pixel51_100, target_side_100, target_pixel52_100, target_side_100, target_pixel53_100,
                    target_side_100, target_pixel54_100, target_side_100, target_pixel55_100, target_side_100,
                    target_pixel56_100, target_side_100, target_pixel57_100, target_side_100, target_pixel58_100,
                    target_side_100,
                    target_pixel61_100, target_side_100, target_pixel62_100, target_side_100, target_pixel63_100,
                    target_side_100, target_pixel64_100, target_side_100, target_pixel65_100, target_side_100,
                    target_pixel66_100, target_side_100, target_pixel67_100, target_side_100, target_pixel68_100,
                    target_side_100,
                    target_pixel71_100, target_side_100, target_pixel72_100, target_side_100, target_pixel73_100,
                    target_side_100, target_pixel74_100, target_side_100, target_pixel75_100, target_side_100,
                    target_pixel76_100, target_side_100, target_pixel77_100, target_side_100, target_pixel78_100,
                    target_side_100,
                    target_pixel81_100, target_side_100, target_pixel82_100, target_side_100, target_pixel83_100,
                    target_side_100, target_pixel84_100, target_side_100, target_pixel85_100, target_side_100,
                    target_pixel86_100, target_side_100, target_pixel87_100, target_side_100, target_pixel88_100,
                    target_side_100, test1],
                   epochs=4, batch_size=10)

        # print(model1.get_output_at(0))

        # print(model1.get_weights())

        # model1.summary()

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
        shisamap[m, n] = shisa*8

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
        shisamap[m, n+1] = shisa*8

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
        shisamap[m, n+2] = shisa*8

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
        shisamap[m, n+3] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer152').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m, n+4] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer162').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m, n+5] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer172').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m, n+6] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer182').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m, n+7] = shisa*8

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
        shisamap[m+1, n] = shisa*8

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
        shisamap[m+1, n+1] = shisa*8

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
        shisamap[m+1, n+2] = shisa*8

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
        shisamap[m+1, n+3] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer252').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+1, n+4] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer262').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+1, n+5] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer272').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+1, n+6] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer282').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+1, n+7] = shisa*8

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
        shisamap[m+2, n] = shisa*8

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
        shisamap[m+2, n+1] = shisa*8

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
        shisamap[m+2, n+2] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer342').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+2, n+3] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer352').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+2, n+4] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer362').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+2, n+5] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer372').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+2, n+6] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer382').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+2, n+7] = shisa*8

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
        shisamap[m+3, n] = shisa*8

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
        shisamap[m+3, n+1] = shisa*8

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
        shisamap[m+3, n+2] = shisa*8

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
        shisamap[m+3, n+3] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer452').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+3, n+4] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer462').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+3, n+5] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer472').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+3, n+6] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer482').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+3, n+7] = shisa*8

        # 5
        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer512').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+4, n] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer522').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+4, n+1] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer532').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+4, n+2] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer542').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+4, n+3] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer552').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+4, n+4] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer562').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+4, n+5] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer572').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+4, n+6] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer582').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+4, n+7] = shisa*8

        # 6
        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer612').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+5, n] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer622').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+5, n+1] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer632').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+5, n+2] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer642').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+5, n+3] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer652').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+5, n+4] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer662').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+5, n+5] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer672').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+5, n+6] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer682').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+5, n+7] = shisa*8

        # 7
        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer712').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+6, n] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer722').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+6, n+1] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer732').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+6, n+2] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer742').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+6, n+3] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer752').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+6, n+4] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer762').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+6, n+5] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer772').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+6, n+6] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer782').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+6, n+7] = shisa*8

        # 8
        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer812').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+7, n] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer822').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+7, n+1] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer832').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+7, n+2] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer842').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+7, n+3] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer852').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+7, n+4] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer862').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+7, n+5] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer872').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+7, n+6] = shisa*8

        dense1_layer_model = Model(inputs=model1.input,
                                   outputs=model1.get_layer('main_layer882').output)

        dense1_output = dense1_layer_model.predict(b)

        print(dense1_output.shape)
        print(dense1_output[0])
        shisa = 0
        print(shisa)
        for i in range(0, 32):
            shisa += i * dense1_output[0, i]
        print(shisa)
        shisamap[m+7, n+7] = shisa*8

        # ------------------------
        # 1
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer113').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer123').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer133').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer143').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer153').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer163').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer173').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer183').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # # 2
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer213').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer223').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer233').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer243').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer253').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer263').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer273').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer283').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # # 3
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer313').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer323').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer333').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer343').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer353').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer363').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer373').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer383').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # # 4
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer413').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer423').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer433').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer443').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer453').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer463').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer473').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer483').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # # 5
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer513').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer523').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer533').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer543').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer553').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer563').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer573').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer583').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # # 6
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer613').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer623').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer633').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer643').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer653').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer663').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer673').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer683').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # # 7
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer713').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer723').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer733').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer743').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer753').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer763').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer773').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer783').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # # 8
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer813').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer823').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer833').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer843').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer853').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer863').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer873').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])
        # dense1_layer_model = Model(inputs=model1.input,
        #                            outputs=model1.get_layer('main_layer883').output)
        # dense1_output = dense1_layer_model.predict(b)
        # print(dense1_output[0])

        print(iml[m, n,], iml[m, n + 1,], iml[m, n + 2,], iml[m, n + 3,], iml[m, n + 4,], iml[m, n + 5,],
              iml[m, n + 6,], iml[m, n + 7,])
        print(iml[m + 1, n,], iml[m + 1, n + 1,], iml[m + 1, n + 2,], iml[m + 1, n + 3,], iml[m + 1, n + 4,],
              iml[m + 1, n + 5,], iml[m + 1, n + 6,], iml[m + 1, n + 7,])
        print(iml[m + 2, n,], iml[m + 2, n + 1,], iml[m + 2, n + 2,], iml[m + 2, n + 3,], iml[m + 2, n + 4,],
              iml[m + 2, n + 5,], iml[m + 2, n + 6,], iml[m + 2, n + 7,])
        print(iml[m + 3, n,], iml[m + 3, n + 1,], iml[m + 3, n + 2,], iml[m + 3, n + 3,], iml[m + 3, n + 4,],
              iml[m + 3, n + 5,], iml[m + 3, n + 6,], iml[m + 3, n + 7,])
        print(iml[m + 4, n,], iml[m + 4, n + 1,], iml[m + 4, n + 2,], iml[m + 4, n + 3,], iml[m + 4, n + 4,],
              iml[m + 4, n + 5,], iml[m + 4, n + 6,], iml[m + 4, n + 7,])
        print(iml[m + 5, n,], iml[m + 5, n + 1,], iml[m + 5, n + 2,], iml[m + 5, n + 3,], iml[m + 5, n + 4,],
              iml[m + 5, n + 5,], iml[m + 5, n + 6,], iml[m + 5, n + 7,])
        print(iml[m + 6, n,], iml[m + 6, n + 1,], iml[m + 6, n + 2,], iml[m + 6, n + 3,], iml[m + 6, n + 4,],
              iml[m + 6, n + 5,], iml[m + 6, n + 6,], iml[m + 6, n + 7,])
        print(iml[m + 7, n,], iml[m + 7, n + 1,], iml[m + 7, n + 2,], iml[m + 7, n + 3,], iml[m + 7, n + 4,],
              iml[m + 7, n + 5,], iml[m + 7, n + 6,], iml[m + 7, n + 7,])

        print(shisamap)

print(shisamap)
shisaoutput = int(shisamap)
img = Image.fromarray(shisaoutput, 'L')
img.save('my.png')


