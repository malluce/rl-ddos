import numpy as np


def maybe_cast_to_arr(arg):
    """
    Casts the argument to a numpy array of shape (1,), if it is of shape () or no array at all.
    Required for continuous actions/observations with shape ().
    :param arg: the argument to cast
    :return: the parsed argument
    """
    return arg if type(arg) == np.ndarray and len(arg.shape) > 0 else np.array(arg).reshape((1,))
