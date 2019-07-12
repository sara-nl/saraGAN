from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras import backend
import numpy as np


# Main loss function.
class WassersteinLoss(Loss):
    def call(self, y_true, y_pred):
        return backend.mean(y_pred)


# Gradient Penalty Loss form Gadriani 2017
class GradientPenaltyLoss(Loss):

    def call(self, *args, **kwargs):
        return self.__call__(args, kwargs)

    def __call__(self, y_true, y_pred, averaged_samples, **kwargs):
        gradients = backend.gradients(y_pred, averaged_samples)[0]
        #   ... compute norm by squaring
        gradients_sqr = backend.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = backend.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = backend.sqrt(gradients_sqr_sum)
        gradient_penalty = backend.square(gradient_l2_norm - 1)

        return backend.mean(gradient_penalty)


# Page 14 first equation.
class DriftLoss(Loss):
    def call(self, y_true, y_pred):
        return backend.mean(backend.square(y_pred))
