from tensorflow.python.keras.losses import Loss
import tensorflow as tf


# Main loss function.
class WassersteinLoss(Loss):
    def __call__(self, y_pred):
        return tf.reduce_mean(y_pred)


class GradientPenaltyLoss(Loss):
    def __call__(self, discriminator, real, fake, labels):
        def _interpolate(a, b=None):
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)

            if labels is not None:
                pred = discriminator([x, labels])
            else:
                pred = discriminator(x)

        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.) ** 2)
        return gp
