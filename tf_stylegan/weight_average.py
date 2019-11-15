
import tensorflow as tf

a = tf.Variable(0.5)
b = tf.Variable(0.6)
c = a + b

def update_moving_average(sess, moving_average, generator_vars, rate=.999):

    assert len(moving_average) == len(generator_vars)

    for i, var in enumerate(moving_average):
        update = var.assign(rate * var + (1 - rate) * generator_vars[i])
        sess.run(update)


with tf.Session() as sess:

    generator_moving_average = None
    sess.run(tf.global_variables_initializer())

    if generator_moving_average is None:
        generator_moving_average = tf.trainable_variables()

    c_np = sess.run(c)

    for var in tf.trainable_variables():
        sess.run(var.assign_add(0.01))

        print(var.eval())

