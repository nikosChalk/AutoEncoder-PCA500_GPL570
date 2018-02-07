

from neural_net import AutoEncoder
from input import DataSet

dataset = DataSet('data', 'PCA500_GPL570')
dataset.normalize() # normalize dataset in range [0, 1]
d_x = 500     #28*28=784
d1 = 700
d2 = 500
d_y = 200
layers = [d_x, d1, d2, d_y, d2, d1, d_x]     #layers[i] = ith-layer's neurons. layer-0 is input layer.


auto_encoder = AutoEncoder(layers, dataset)
auto_encoder.train(total_epochs=3000, batch_size=200)
auto_encoder.test()
auto_encoder.delete()
