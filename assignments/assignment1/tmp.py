import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from models.model_numpy import SoftmaxNetwork, FullyConnectedLayer
from utils import Sigmoid, Softmax, Tanh, init_weights, plot_decision_boundary

# dataset generating options
N = 1000
noise = 0.3
input_dim = 2

# generating dataset
np.random.seed(0)
x, y_tar = make_moons(n_samples=N, noise=noise)
plt.scatter(x[:, 0], x[:, 1], s=20, c=y_tar, cmap=plt.cm.Spectral)
#plt.show()

# creating layers for the network
seed = np.random.randint(1e6)
def build_simple_network():
    np.random.seed(seed)
    hid_layers = [FullyConnectedLayer(input_dim=2, hidden_dim=10, act_func=Tanh(), w_initializer=init_weights),
                 FullyConnectedLayer(input_dim=10, hidden_dim=2, act_func=Softmax(), w_initializer=init_weights)]
    return SoftmaxNetwork(hid_layers)

# decision boundary for non-trained network
network_numpy = build_simple_network()
y_pred = network_numpy.predict(x)
print 'accuracy: {:.3f}'.format(accuracy_score(y_true=y_tar, y_pred=y_pred))
plot_decision_boundary(x, y_tar, network_numpy.predict)

from tqdm import tqdm
from models.opt_derivative_free import mult_random_search, mult_cem

###RANDOM SEARCH###

def f(params):
    network_numpy.set_params_from_flatten_array(params)
    return network_numpy.calc_accuracy(x, y_tar)

# training
network_numpy = build_simple_network()
params_mean = np.zeros(network_numpy.n_params)
av_acc, max_acc = [], []
n_iter = 50
res_rs = mult_random_search(f, params_mean, params_std=1., n_workers=1, batch_size=200, n_iter=n_iter)
for i in tqdm(xrange(n_iter)):
    res = next(res_rs)
    av_acc.append(res['results'].mean())
    max_acc.append(np.max(res['results']))
plt.plot(av_acc)
plt.plot(max_acc)
plt.show()
print 'Last iteration. Average accuracy in batch: {:.2f}%; Accuracy with best weights: {:.2f}%'\
                                                                      .format(av_acc[-1]*100, max_acc[-1]*100)

# decision boundary
best_weights = res['best_params']
network_numpy.set_params_from_flatten_array(best_weights)
plot_decision_boundary(x, y_tar, network_numpy.predict)
plt.show()

###CEM###

# training
network_numpy = build_simple_network()
params_mean = np.zeros(network_numpy.n_params)
av_acc, max_acc = [], []
n_iter = 50
res_rs = mult_cem(f, params_mean, params_std=1., n_workers=4, batch_size=200, n_iter=n_iter)
for i in tqdm(xrange(n_iter)):
    res = next(res_rs)
    av_acc.append(res['results'].mean())
    max_acc.append(np.max(res['results']))
plt.plot(av_acc)
plt.plot(max_acc)
plt.show()
print 'Last iteration. Average accuracy in batch: {:.2f}%; Accuracy with best weights: {:.2f}%'\
                                                                      .format(av_acc[-1]*100, max_acc[-1]*100)

# decision boundary
best_weights = res['best_params']
network_numpy.set_params_from_flatten_array(best_weights)
plot_decision_boundary(x, y_tar, network_numpy.predict)
plt.show()

###BATCH GRADIENT###

# training
network_numpy = build_simple_network()
acc = [network_numpy.calc_accuracy(x, y_tar)]
n_iter = 400
for i in tqdm(xrange(n_iter)):
    network_numpy.update_weights(x=x, y_tar=y_tar, eps=1.)
    acc.append(network_numpy.calc_accuracy(x, y_tar))
plt.plot(acc)
plt.show()
print 'Accuracy with last weights: {:.2f}%'.format(acc[-1]*100)

# decision boundary
plot_decision_boundary(x, y_tar, network_numpy.predict)
plt.show()

