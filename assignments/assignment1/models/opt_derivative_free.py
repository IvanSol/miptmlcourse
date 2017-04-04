
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

from multiprocessing import Pool

def mult_random_search(f, params_mean, params_std=1., n_workers=2, batch_size=100, n_iter=50):
    """ Multiprocessing version of Random Search algorithm."""
    params_std = np.ones_like(params_mean) * params_std
    best_params = params_mean
    best_accuracy = 0
    #pool = Pool(processes=n_workers)
    for _ in range(n_iter):
        params = np.array([params_std * np.random.randn(params_mean.shape[0]) for _ in xrange(batch_size)]).T + np.matlib.repmat(best_params, batch_size, 1).T
        params_arr = map(lambda x: x[:, 0], np.split(params, batch_size, axis=-1))
        ys = np.array(map(f, params_arr)) #pool
        if np.max(ys) > best_accuracy:
            best_params = params_arr[np.argmax(ys)]
            best_accuracy = np.max(ys)
        yield {'results': ys, 'best_params': best_params, 'best_accuracy': best_accuracy}
    #pool.close()

def mult_cem(f, params_mean, params_std=1., n_workers=2, batch_size=100, n_iter=50, elite_frac=0.2):
    """ Multiprocessing version of CEM algorithm."""
    # https://en.wikipedia.org/wiki/Cross-entropy_method
    n_elite = int(np.round(batch_size * elite_frac))
    params_std = np.ones_like(params_mean) * params_std
    #pool = Pool(processes=n_workers)
    for _ in range(n_iter):
        params = np.multiply(np.matlib.repmat(params_std, batch_size, 1).T,
                             np.array([np.random.randn(params_mean.shape[0]) for _ in xrange(batch_size)]).T
                             ) + np.matlib.repmat(params_mean, batch_size, 1).T

        params_arr = map(lambda x: x[:, 0], np.split(params, batch_size, axis=-1))
        ys = np.array(map(f, params_arr)) #pool
        elite_idx = np.argsort(ys)[-n_elite:]
        params_elite = params[:, elite_idx];
        params_std = np.std(params_elite, axis=1)
        params_mean = np.mean(params_elite, axis=1);
        best_params = params_arr[elite_idx[-1]]
        yield {'results': ys, 'best_params': best_params}
    #pool.close()
