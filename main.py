

from neural_net import AutoEncoder

d_x = 784     #28*28=784
d1 = 128
d2 = 64
d_y = 32
layers = [d_x, d1, d2, d_y, d2, d1, d_x]     #layers[i] = ith-layer's neurons. layer-0 is input layer.

auto_encoder = AutoEncoder(layers)
auto_encoder.train(total_epochs=10000, batch_size=250)
auto_encoder.test()
auto_encoder.delete()