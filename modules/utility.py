# A helper module for various sub-tasks
from time import time
import numpy as np
import jax, copy
import jax.numpy as jnp
import pickle
from jax.tree_util import tree_map
from flax.training import checkpoints as ck
import bayesnf as bnf


def timer(func):
	"""
	Timing wrapper for a generic function.
	Prints the time spent inside the function to the output.
	"""
	def new_func(*args, **kwargs):
		start = time()
		val = func(*args,**kwargs)
		end = time()
		print(f'Time taken by {func.__name__} is {end-start:.4f} seconds')
		return val
	return new_func





def smooth(y, box_pts=10):
    """
    Smooth a time series using a boxcar average.

    Parameters
    ----------
    y : array_like
        The time series to be smoothed.
    box_pts : int, optional
        The window size for the boxcar average. Defaults to 10.

    Returns
    -------
    y_smooth : array_like
        The smoothed time series.
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth




def save(model, file_path, step=None):
    data = copy.deepcopy(model.__dict__)
    ck.save_checkpoint(file_path, model.params_, step=step)
    with open(f'{file_path}/attrs.pkl', 'wb') as f:
        data.pop('params_')
        pickle.dump(data, f)
    

@timer
def load(file_path, df_train, **kwargs):
    model = bnf.BayesianNeuralFieldMAP(**kwargs)
    model.fit(df_train.iloc[:1, :], seed=jax.random.PRNGKey(0), num_epochs=1)
    params_restored = ck.restore_checkpoint(file_path, target=None)
    model.params_ = model.params_._replace(**params_restored)

    with open(f'{file_path}/attrs.pkl', "rb") as f:
        rest = pickle.load(f)

    for k, v in rest.items():
        model.__dict__[k] = v
    
    return model

def count_params(model):
    s = 0
    for p in model.params_:
        s += np.prod(p.shape)
    return s