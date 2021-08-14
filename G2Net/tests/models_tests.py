from G2Net.models.BigConvModelFor3Input import ConvModelFor3Input

model = ConvModelFor3Input()
model.build(input_shape=((4096,1),(4096,1),(4096,1)))
model.summary()