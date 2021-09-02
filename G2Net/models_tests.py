from G2Net.G2Net.BigConvModelFor3Input import ConvModelFor3Input
from G2Net.G2Net.autoencoder_models import Encoder,Decoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

encoder = Encoder(big_conv_num=3,
                  filters_sequence=[[8,16],[16,32],[32,64]],
                  kernel_size_sequence=[[10,10]]*3,
                  pooling_factors=[4,4,4],
                  activations=['selu']*3,
                  normalization_types=['batch']*3,
                  conv_nums=[[3,2]]*3,
                  dropouts=[0.03]*3,
                  dropout_head=0.1)
# encoder.build(input_shape=((4096,1),(4096,1),(4096,1)))
# encoder.summary()

decoder = Decoder(
                input_length=256,
                big_conv_num=3,
                filters_sequence=[[64,32],[32,16],[16,8]],
                kernel_size_sequence=[[10,10]]*3,
                pooling_factors=[4,4,4],
                activations=['selu']*3,
                normalization_types=['batch']*5,
                conv_nums=[[3,2]]*3,
                  )
# decoder.build(input_shape=(256,))
# decoder.build(input_shape=(256,))
# decoder.summary()

input_0 = Input((4096,1))
input_1 = Input((4096,1))
input_2 = Input((4096,1))
latent = encoder((input_0,input_1,input_2))
outputs = decoder(latent)
model = Model(inputs=(input_0,input_1,input_2), outputs=outputs)
model.build(input_shape=((4096,1),(4096,1),(4096,1)))
model.summary()