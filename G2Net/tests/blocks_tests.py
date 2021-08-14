from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from G2Net.layers.custom_blocks import ConvBlockG2Net, BigConvBlock, BigConvBlock4Input, Dropout4Input


input_0 = Input(shape=[4096, 1])
input_1 = Input(shape=[4096, 1])
input_2 = Input(shape=[4096, 1])
input_all = Input(shape=[4096, 1])


def ConvBlockG2NetSummary():
    conv_block_g2net = ConvBlockG2Net(filters=32)
    output = conv_block_g2net(input_0)
    model = Model(inputs=input_0, outputs=output)
    model.summary()


def BigConvBlockSummary():
    big_conv_block = BigConvBlock(32, 32)
    output = big_conv_block(input_0)
    model = Model(inputs=input_0, outputs=output)
    model.summary()


def BigConvBlock4InputSummary():
    big_conv_block4Input = BigConvBlock4Input(32, 32)
    x_0, x_1, x_2, x_all = big_conv_block4Input(input_0, input_1, input_2, input_all)
    model = Model(inputs=[input_0, input_1, input_2, input_all], outputs=[x_0, x_1, x_2, x_all])
    model.summary()


def Dropout4InputSummary():
    dropout_4Input = Dropout4Input(0.2)
    x_0, x_1, x_2, x_all = dropout_4Input(input_0, input_1, input_2, input_all)
    model = Model(inputs=[input_0, input_1, input_2, input_all], outputs=[x_0, x_1, x_2, x_all])
    model.summary()


ConvBlockG2NetSummary()
BigConvBlockSummary()
BigConvBlock4InputSummary()
Dropout4InputSummary()