import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Conv1D, MaxPool1D, \
    Add, Dropout, UpSampling1D


class ConvBlockG2Net(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, pooling_factor=2, conv_nums=2, normalization_type='batch',
                 activation='relu', name=None
                 ):
        super(ConvBlockG2Net, self).__init__(name=name)
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


class BigConvBlock(tf.keras.Model):
    def __init__(self,
                 filters_1, filters_2,
                 kernel_size_1=3, kernel_size_2=3,
                 pooling_factor=2, activation='relu', normalization_type='batch',
                 conv_num_1=2, conv_num_2=2, name=None):
        super(BigConvBlock, self).__init__(name=name)
        if normalization_type == "layer":
            self.norm = LayerNormalization()
        else:
            self.norm = BatchNormalization()
        self.conv_block_1 = ConvBlockG2Net(filters_1, kernel_size=kernel_size_1, pooling_factor=pooling_factor,
                                           conv_nums=conv_num_1, normalization_type=normalization_type,
                                           activation=activation
                                           )
        self.conv_block_2 = ConvBlockG2Net(filters_2, kernel_size=kernel_size_2, pooling_factor=pooling_factor,
                                           conv_nums=conv_num_2, normalization_type=normalization_type,
                                           activation=activation
                                           )
        self.residual_conv_block = Conv1D(filters=filters_2, kernel_size=kernel_size_2,
                                          padding='same', activation=activation)
        self.residual_pooling = MaxPool1D(pool_size=pooling_factor**2, strides=pooling_factor**2, padding='same')
        self.add_block = Add()

    def call(self, input_tensor):
        '''
        :param input_tensor: shape [batch, seq, channel]
        :return: shape [batch, seq/4, filters_2]
        '''
        x = self.conv_block_1(input_tensor)
        x = self.conv_block_2(x)
        y = self.norm(input_tensor)
        y = self.residual_conv_block(y)
        y = self.residual_pooling(y)
        return self.add_block([x, y])


class BigConvBlock4Input(tf.keras.Model):
    def __init__(self, filters_1, filters_2, kernel_size_1=3, kernel_size_2=3,
                 pooling_factor=2, activation='relu', normalization_type='batch',
                 conv_num_1=2, conv_num_2=2, name=None):
        super(BigConvBlock4Input, self).__init__(name=name)
        self.big_conv_0 = BigConvBlock(filters_1, filters_2, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2,
                                       pooling_factor=pooling_factor, activation=activation,
                                       normalization_type=normalization_type,
                                       conv_num_1=conv_num_1, conv_num_2=conv_num_2
                                       )
        self.big_conv_1 = BigConvBlock(filters_1, filters_2, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2,
                                       pooling_factor=pooling_factor, activation=activation,
                                       normalization_type=normalization_type,
                                       conv_num_1=conv_num_1, conv_num_2=conv_num_2
                                       )
        self.big_conv_2 = BigConvBlock(filters_1, filters_2, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2,
                                       pooling_factor=pooling_factor, activation=activation,
                                       normalization_type=normalization_type,
                                       conv_num_1=conv_num_1, conv_num_2=conv_num_2
                                       )
        self.big_conv_all = BigConvBlock(filters_1, filters_2, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2,
                                         pooling_factor=pooling_factor, activation=activation,
                                         normalization_type=normalization_type,
                                         conv_num_1=conv_num_1, conv_num_2=conv_num_2
                                         )
        self.add_block = Add()

    def call(self, input_tensor_0, input_tensor_1, input_tensor_2, input_tensor_all):
        x_0 = self.big_conv_0(input_tensor_0)
        x_1 = self.big_conv_1(input_tensor_1)
        x_2 = self.big_conv_2(input_tensor_2)
        x_all = self.big_conv_all(input_tensor_all)
        x_all = self.add_block([x_0, x_1, x_2, x_all])
        return x_0, x_1, x_2, x_all


class Dropout4Input(tf.keras.Model):
    def __init__(self, dropout=0.2, name=None):
        super(Dropout4Input, self).__init__(name=name)
        self.dropout_0 = Dropout(dropout)
        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)
        self.dropout_all = Dropout(dropout)

    def call(self, input_tensor_0, input_tensor_1, input_tensor_2, input_tensor_all):
        x_0 = self.dropout_0(input_tensor_0)
        x_1 = self.dropout_1(input_tensor_1)
        x_2 = self.dropout_2(input_tensor_2)
        x_all = self.dropout_all(input_tensor_all)
        return x_0, x_1, x_2, x_all


class ConvBlockG2NetDecode(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, pooling_factor=2, conv_nums=2, normalization_type='batch',
                 activation='relu', name=None
                 ):
        super(ConvBlockG2NetDecode, self).__init__(name=name)
        if normalization_type == "layer":
            self.norm = LayerNormalization()
        else:
            self.norm = BatchNormalization()
        self.conv_blocks = [Conv1D(filters=filters, kernel_size=kernel_size,
                                   padding='same', activation=activation)
                            for i in range(conv_nums)]
        self.upsampling = UpSampling1D(size=pooling_factor)
        self.residual_conv_block = Conv1D(filters=filters, kernel_size=kernel_size,
                                          padding='same', activation=activation)
        self.residual_upsampling = UpSampling1D(size=pooling_factor)
        self.add_block = Add()

    def call(self, input_tensor):
        normalized = self.norm(input_tensor)
        x = normalized
        x = self.upsampling(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        y = self.residual_upsampling(normalized)
        y = self.residual_conv_block(y)
        return self.add_block([x, y])


class BigConvBlockDecode(tf.keras.Model):
    def __init__(self,
                 filters_1, filters_2,
                 kernel_size_1=3, kernel_size_2=3,
                 pooling_factor=2, activation='relu', normalization_type='batch',
                 conv_num_1=2, conv_num_2=2, name=None):
        super(BigConvBlockDecode, self).__init__(name=name)
        if normalization_type == "layer":
            self.norm = LayerNormalization()
        else:
            self.norm = BatchNormalization()
        self.conv_block_1 = ConvBlockG2NetDecode(filters_1, kernel_size=kernel_size_1, pooling_factor=pooling_factor,
                                           conv_nums=conv_num_1, normalization_type=normalization_type,
                                           activation=activation
                                           )
        self.conv_block_2 = ConvBlockG2NetDecode(filters_2, kernel_size=kernel_size_2, pooling_factor=pooling_factor,
                                           conv_nums=conv_num_2, normalization_type=normalization_type,
                                           activation=activation
                                           )
        self.residual_conv_block = Conv1D(filters=filters_2, kernel_size=kernel_size_2,
                                          padding='same', activation=activation)
        self.residual_upsampling = UpSampling1D(size=pooling_factor**2)
        self.add_block = Add()

    def call(self, input_tensor):
        '''
        :param input_tensor: shape [batch, seq, channel]
        :return: shape [batch, seq/4, filters_2]
        '''
        x = self.conv_block_1(input_tensor)
        x = self.conv_block_2(x)
        y = self.norm(input_tensor)
        y = self.residual_upsampling(y)
        y = self.residual_conv_block(y)
        return self.add_block([x, y])


class BigConvBlock4InputDecode(tf.keras.Model):
    def __init__(self, filters_1, filters_2, kernel_size_1=3, kernel_size_2=3,
                 pooling_factor=2, activation='relu', normalization_type='batch',
                 conv_num_1=2, conv_num_2=2, name=None):
        super(BigConvBlock4InputDecode, self).__init__(name=name)
        self.big_conv_0 = BigConvBlockDecode(filters_1, filters_2, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2,
                                       pooling_factor=pooling_factor, activation=activation,
                                       normalization_type=normalization_type,
                                       conv_num_1=conv_num_1, conv_num_2=conv_num_2
                                       )
        self.big_conv_1 = BigConvBlockDecode(filters_1, filters_2, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2,
                                       pooling_factor=pooling_factor, activation=activation,
                                       normalization_type=normalization_type,
                                       conv_num_1=conv_num_1, conv_num_2=conv_num_2
                                       )
        self.big_conv_2 = BigConvBlockDecode(filters_1, filters_2, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2,
                                       pooling_factor=pooling_factor, activation=activation,
                                       normalization_type=normalization_type,
                                       conv_num_1=conv_num_1, conv_num_2=conv_num_2
                                       )
        self.big_conv_all = BigConvBlockDecode(filters_1, filters_2, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2,
                                         pooling_factor=pooling_factor, activation=activation,
                                         normalization_type=normalization_type,
                                         conv_num_1=conv_num_1, conv_num_2=conv_num_2
                                         )
        self.add_block = Add()

    def call(self, input_tensor_0, input_tensor_1, input_tensor_2, input_tensor_all):
        x_0 = self.big_conv_0(input_tensor_0)
        x_1 = self.big_conv_1(input_tensor_1)
        x_2 = self.big_conv_2(input_tensor_2)
        x_all = self.big_conv_all(input_tensor_all)
        x_all = self.add_block([x_0, x_1, x_2, x_all])
        return x_0, x_1, x_2, x_all