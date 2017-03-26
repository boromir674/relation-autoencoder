import numpy as np
import theano
import theano.tensor as T


class AdaGrad(object):
    def __init__(self, params):
        """
        # Create a list of SharedVariables holding arrays of the same shape as all the weight matrices (parameters) of both the encoder and decoder_type models\n
        :param params: list of encoder and decoder_type weight matrices (decoder_type parameters)
        """
        self.accumulator = []
        for para_i in params:
            eps_p = np.zeros_like(para_i.get_value(borrow=True), dtype=theano.config.floatX)
            self.accumulator.append(theano.shared(eps_p, borrow=True))
        # TODO replace above with: self.accumulator = [theano.shared(np.zeros_like(p.get_value()), borrow=True) for p in params]

    def update(self, learning_rate, params, cost):
        """AdaGrad computations\n
        :param learning_rate: the rate for the weights updating
        :param params: list of SharedVariables: ecoder's and decoder_type's weight matrices
        :param cost: the objective function with respect to which to compute the gradients
        :type cost: scalar (0-dimensional) tensor variable or None
        :return:
        :rtype: a list of tuples
        """
        grads = T.grad(cost, params)
        updates = []
        for param_i, grad_i, acc_i in zip(params, grads, self.accumulator):
            acc = acc_i + T.sqr(grad_i)
            updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc) + 1e-6)))
            updates.append((acc_i, acc))
        return updates


class SGD(object):
    @staticmethod
    def update(learning_rate, params, cost):
        """Stochastic Gradient Descend computations\n
        :param learning_rate: the rate for the weights updating
        :param params: list of SharedVariables: ecoder's and decoder_type's weight matrices
        :type params: list of SharedVariables
        :param cost: the objective function with respect to which to compute the gradients
        :type cost: scalar (0-dimensional) tensor variable or None
        :return: pairs of the input parameters and their respective updated values as symbolic expressions
        :rtype: list of tuples of 2 variables
        """
        grads = T.grad(cost, params)
        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))
        return updates
