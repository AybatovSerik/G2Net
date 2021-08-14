import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from models.BigConvModelFor3Input import ConvModelFor3Input

model = ConvModelFor3Input()
model.build(input_shape=((4096,1),(4096,1),(4096,1)))
model.summary()