import numpy as np


def check_linex_args(function):
    def assert_args(*args, **kwargs):
        y_true = kwargs['y_true'] if 'y_true' in kwargs else args[0]
        y_predict = kwargs['y_predict'] if 'y_predict' in kwargs else args[1]
        try:
            assert_shape_msg = (
                "In linex_error function, argument y_true and y_predict must have the same shape. {} != {}"
                .format(y_true.shape, y_predict.shape)
            )
            assert (y_true.shape == y_predict.shape), assert_shape_msg
        except AttributeError:
            # That's fine, y_true or y_predict are not mandatory arrays
            pass
        assert_len_msg = (
            "In linex_error function, argument y_true and y_predict must have the same length. {} != {}"
            .format(len(y_true), len(y_predict))
        )
        assert (len(y_true) == len(y_predict)), assert_len_msg
        return function(*args, **kwargs)
    return assert_args


@check_linex_args
def linex_error(y_true, y_predict, alpha=-0.1):
    """
    Compute LinEx loss from true labels and predicted labels

    Parameters
    ----------
    y_true : true values of y. Could be a list, a flat array or an array of shape (1, N). Must be the same length as y_predict.
    y_predict : predicted values of y. Could be a list, a flat array or an array of shape (1, N). Must be the same length as y_true.
    alpha : exponential coefficient in linex loss. Should be a negative float as stated in challenge description.
    >>> linex_error([1, 1], [1, 1])
    0.0
    >>> linex_error(np.array([1, 2]), np.array([1, 12])) == np.exp(1) - 2
    True
    """
    error = np.array(y_true) - np.array(y_predict)
    max_error = max(error)
    log_sum_exp = max_error + np.log(np.sum(np.exp(alpha * (error - max_error))))
    return np.exp(log_sum_exp) - np.sum(alpha * error) - len(y_true)
