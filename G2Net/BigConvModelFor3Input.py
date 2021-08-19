from G2Net.G2Net.custom_blocks import BigConvBlock4Input, Dropout4Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Concatenate, Dropout


def ConvModelFor3Input(big_conv_num=5,
                       filters_sequence=[[4,8],[8,16],[16,32],[32,64],[64,128]],
                       kernel_size_sequence=[[3,3],[3,3],[3,3],[3,3],[3,3]],
                       pooling_factors=[2,2,2,2,2],
                       activations=['relu','relu','relu','relu','relu'],
                       normalization_types=['batch']*5,
                       conv_nums=[[2,2]]*5,
                       dropouts=[0.2]*5,
                       dropout_head=0.2):
    assert len(filters_sequence)==big_conv_num, f"filters for {len(filters_sequence)} blocks, but big_conv_num is {big_conv_num}"
    assert len(kernel_size_sequence)==big_conv_num, f"kernel_size_sequence for {len(kernel_size_sequence)} blocks, but big_conv_num is {big_conv_num}"
    assert len(pooling_factors)==big_conv_num, f"pooling_factors for {len(pooling_factors)} blocks, but big_conv_num is {big_conv_num}"
    assert len(activations)==big_conv_num, f"activations for {len(activations)} blocks, but big_conv_num is {big_conv_num}"
    assert len(normalization_types)==big_conv_num, f"normalization_types for {len(normalization_types)} blocks, but big_conv_num is {big_conv_num}"
    assert len(conv_nums)==big_conv_num, f"conv_nums for {len(conv_nums)} blocks, but big_conv_num is {big_conv_num}"
    assert len(dropouts)==big_conv_num, f"dropouts for {len(dropouts)} blocks, but big_conv_num is {big_conv_num}"
    input_0 = Input(shape=[4096, 1])
    input_1 = Input(shape=[4096, 1])
    input_2 = Input(shape=[4096, 1])
    x_0 = input_0
    x_1 = input_1
    x_2 = input_2
    x_all = Concatenate()([x_0, x_1, x_2])
    for block_num in range(big_conv_num):
        x_0, x_1, x_2, x_all = BigConvBlock4Input(filters_1=filters_sequence[block_num][0],
                                                  filters_2=filters_sequence[block_num][1],
                                                  kernel_size_1=kernel_size_sequence[block_num][0],
                                                  kernel_size_2=kernel_size_sequence[block_num][1],
                                                  pooling_factor=pooling_factors[block_num],
                                                  activation=activations[block_num],
                                                  normalization_type=normalization_types[block_num],
                                                  conv_num_1=conv_nums[block_num][0],
                                                  conv_num_2=conv_nums[block_num][1]
                                                  )(x_0, x_1, x_2, x_all)
        x_0, x_1, x_2, x_all = Dropout4Input(dropout=dropouts[block_num])(x_0, x_1, x_2, x_all)
    x_0 = Flatten()(x_0)
    x_1 = Flatten()(x_1)
    x_2 = Flatten()(x_2)
    x_all = Flatten()(x_all)
    flat_all = Concatenate()([x_0, x_1, x_2, x_all])
    flat_all = Dropout(dropout_head)(flat_all)
    output = Dense(1, activation='sigmoid')(flat_all)
    return Model(inputs=[input_0, input_1, input_2], outputs=output)

