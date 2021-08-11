import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Conv1D, MaxPool1D,\
    Add


class ConvBlockG2Net(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, pooling_factor=2, conv_nums=2, normalization_type='batch',
                 activation='relu'
                 ):
        super(ConvBlockG2Net, self).__init__(name='ConvBlockG2Net')
        if normalization_type == "layer":
            self.norm = LayerNormalization()
        else:
            self.norm = BatchNormalization()
        self.conv_blocks = [Conv1D(filters=filters, kernel_size=kernel_size,
                                   padding='same', activation=activation)
                            for i in range(conv_nums)]
        self.max_pool = MaxPool1D(pool_size=pooling_factor, strides=pooling_factor, padding='same')
        self.residual_conv_block = Conv1D(filters=filters, kernel_size=kernel_size,
                                          padding='same', activation=activation)
        self.residual_pooling = MaxPool1D(pool_size=pooling_factor, strides=pooling_factor, padding='same')
        self.add_block = Add()

    def call(self, input_tensor):
        normalized = self.norm(input_tensor)
        x = normalized
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.max_pool(x)
        y = self.residual_conv_block(normalized)
        y = self.residual_pooling(y)
        return self.add_block([x, y])

