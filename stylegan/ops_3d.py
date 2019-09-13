import tensorflow as tf
import numpy as np
from collections import OrderedDict
from tensorflow.python.keras import layers
from tensorflow import keras
from functools import partial

##################################################################################
# Layer
##################################################################################

# pad = ceil[ (kernel - stride) / 2 ]

def get_weight(weight_shape, gain, lrmul):
    fan_in = np.prod(weight_shape[:-1])  # [kernel, kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in)  # He init

    # equalized learning rate
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul
    
    initializer = partial(tf.keras.initializers.RandomNormal(), shape=weight_shape)
    weight = tf.Variable(initial_value=initializer, name='weight', shape=weight_shape, dtype=tf.float32) * runtime_coef
    return weight


class Conv3D(layers.Layer):
    def __init__(self, weight_shape, stride=1, gain=np.sqrt(2), lrmul=1.0):
        super(Conv3D, self).__init__()
        self.weight = get_weight(weight_shape, gain, lrmul)
        self.stride = stride
    
    def call(self, inputs):
        x = tf.nn.conv3d(input=inputs, filters=self.weight, 
                         strides=[1, self.stride, self.stride, self.stride, 1], padding='SAME')
        return x

def conv(x, channels, kernel=3, stride=1, gain=np.sqrt(2), lrmul=1.0):
    weight_shape = [kernel, kernel, kernel, x.get_shape().as_list()[-1], channels]
    return Conv3D(weight_shape, stride=stride, gain=gain, lrmul=lrmul)(x)


class FullyConnected(layers.Layer):
    def __init__(self, weight_shape, gain=np.sqrt(2), lrmul=1.0):
        super(FullyConnected, self).__init__()
        self.weight = get_weight(weight_shape, gain, lrmul)
        
    def call(self, inputs):
        x = tf.matmul(inputs, self.weight)
        return x

def fully_connected(x, units, gain=np.sqrt(2), lrmul=1.0):
    x = flatten(x)
    weight_shape = [x.get_shape().as_list()[-1], units]
    return FullyConnected(weight_shape, gain, lrmul)(x)


def flatten(x) :
    return layers.Flatten()(x)


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return layers.LeakyReLU(alpha=alpha)(x)

##################################################################################
# Normalization function
##################################################################################


class PixelNorm(layers.Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()
        
    def call(self, inputs):
        norm = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        x = inputs * tf.math.rsqrt(norm + 1e-8)
        return x


def pixel_norm(x):
    return PixelNorm()(x)

def adaptive_instance_norm(x, w):
    x = instance_norm(x)
    x = style_mod(x, w)
    return x

def instance_norm(x, epsilon=1e-8):
    x = x - tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=[1, 2, 3], keepdims=True) + epsilon)
    return x

##################################################################################
# StyleGAN trick function
##################################################################################

def compute_loss(real_images, real_logit, fake_logit, tape):
    r1_gamma, r2_gamma = 10.0, 0.0

    # discriminator loss: gradient penalty
    d_loss_gan = tf.nn.softplus(fake_logit) + tf.nn.softplus(-real_logit)
    real_loss = tf.reduce_sum(real_logit)
    # print(tape.gradient(real_loss, real_images))
    real_grads = tape.gradient(real_loss, real_images)[0]
    r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
    # r1_penalty = tf.reduce_mean(r1_penalty)
    d_loss = d_loss_gan + r1_penalty * (r1_gamma * 0.5)
    d_loss = tf.reduce_mean(d_loss)

    # generator loss: logistic nonsaturating
    g_loss = tf.nn.softplus(-fake_logit)
    g_loss = tf.reduce_mean(g_loss)

    return d_loss, g_loss

def lerp(a, b, t):
    # t == 1.0: use b
    # t == 0.0: use a
    out = a + (b - a) * t
    return out

def lerp_clip(a, b, t):
    # t >= 1.0: use b
    # t <= 0.0: use a
    out = a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
    return out

def smooth_transition(prv, cur, res, transition_res, alpha):
    # alpha == 1.0: use only previous resolution output
    # alpha == 0.0: use only current resolution output

    # use alpha for current resolution transition
    if transition_res == res:
        out = lerp_clip(cur, prv, alpha)

    # ex) transition_res=32, current_res=16
    # use res=16 block output
    else:   # transition_res > res
        out = lerp_clip(cur, prv, 0.0)
    return out

def smooth_transition_state(batch_size, global_step, train_trans_images_per_res_tensor, zero_constant):
    # alpha == 1.0: use only previous resolution output
    # alpha == 0.0: use only current resolution output
    n_cur_img = batch_size * global_step
    n_cur_img = tf.cast(n_cur_img, dtype=tf.float32)

    is_transition_state = tf.less_equal(n_cur_img, train_trans_images_per_res_tensor)
    alpha = tf.cond(is_transition_state,
                    true_fn=lambda: (train_trans_images_per_res_tensor - n_cur_img) / train_trans_images_per_res_tensor,
                    false_fn=lambda: zero_constant)
    return alpha

def get_alpha_const(iterations, batch_size, global_step) :
    # additional variables (reuse zero constants)
    zero_constant = tf.constant(0.0, dtype=tf.float32, shape=[])

    # additional variables (for training only)
    train_trans_images_per_res_tensor = tf.constant(iterations, dtype=tf.float32, shape=[], name='train_trans_images_per_res')

    # determine smooth transition state and compute alpha value
    alpha_const = smooth_transition_state(batch_size, global_step, train_trans_images_per_res_tensor, zero_constant)

    return alpha_const, zero_constant

##################################################################################
# StyleGAN discriminator
##################################################################################

class DiscriminatorBlock(keras.Sequential):
    def __init__(self, res, n_f0, n_f1):
        super(DiscriminatorBlock, self).__init__()
        self.res = res
        self.n_f0 = n_f0
        self.n_f1 = n_f1
    
    def call(self, inputs):
        x = conv(inputs, channels=self.n_f0, kernel=3, stride=1, gain=np.sqrt(2), lrmul=1.0)
        x = apply_bias(x, lrmul=1.0)
        x = lrelu(x, 0.2)

        x = blur3d(x, [1, 2, 1])
        x = downscale_conv(x, self.n_f1, kernel=3, gain=np.sqrt(2), lrmul=1.0)
        x = apply_bias(x, lrmul=1.0)
        x = lrelu(x, 0.2)
    
        return x

def discriminator_block(x, res, n_f0, n_f1):
    return DiscriminatorBlock(res, n_f0, n_f1)(x)


class DiscriminatorLastBlock(keras.Sequential):
    def __init__(self, res, n_f0, n_f1):
        super(DiscriminatorLastBlock, self).__init__()
        self.res = res
        self.n_f0 = n_f0
        self.n_f1 = n_f1
    
    def call(self, inputs):
        x = minibatch_stddev_layer(inputs, group_size=4, num_new_features=1)

        x = conv(x, channels=self.n_f0, kernel=3, stride=1, gain=np.sqrt(2), lrmul=1.0)
        x = apply_bias(x, lrmul=1.0)
        x = lrelu(x, 0.2)

        x = fully_connected(x, units=self.n_f1, gain=np.sqrt(2), lrmul=1.0)
        x = apply_bias(x, lrmul=1.0)
        x = lrelu(x, 0.2)

        x = fully_connected(x, units=1, gain=1.0, lrmul=1.0)
        x = apply_bias(x, lrmul=1.0)
        
        return x

def discriminator_last_block(x, res, n_f0, n_f1):
    return DiscriminatorLastBlock(res, n_f0, n_f1)(x)

    
##################################################################################
# StyleGAN generator
##################################################################################

def get_style_class(resolutions, featuremaps) :

    coarse_styles = OrderedDict()
    middle_styles = OrderedDict()
    fine_styles = OrderedDict()

    for res, n_f in zip(resolutions, featuremaps) :
        if res >= 4 and res <= 8 :
            coarse_styles[res] = n_f
        elif res >= 16 and res <= 32 :
            middle_styles[res] = n_f
        else :
            fine_styles[res] = n_f

    return coarse_styles, middle_styles, fine_styles


def synthesis_const_block(res, w_broadcasted, n_f):
    w0 = w_broadcasted[:, 0]
    w1 = w_broadcasted[:, 1]


    batch_size = tf.shape(w0)[0]
    
    x = tf.Variable(name='const', shape=[1, 4, 4, 4, n_f], dtype=tf.float32, 
                   initial_value=tf.ones_initializer()(shape=[1, 4, 4, 4, n_f]))

    x = tf.tile(x, [batch_size, 1, 1, 1, 1])

    x = apply_noise(x) # B module
    x = apply_bias(x, lrmul=1.0)

    x = lrelu(x, 0.2)
    x = adaptive_instance_norm(x, w0) # A module

    x = conv(x, channels=n_f, kernel=3, stride=1, gain=np.sqrt(2), lrmul=1.0)

    x = apply_noise(x) # B module
    x = apply_bias(x, lrmul=1.0)

    x = lrelu(x, 0.2)
    x = adaptive_instance_norm(x, w1) # A module

    return x


def synthesis_block(x, res, w_broadcasted, layer_index, n_f):
    w0 = w_broadcasted[:, layer_index]
    w1 = w_broadcasted[:, layer_index + 1]

    x = upscale_conv(x, n_f, kernel=3, gain=np.sqrt(2), lrmul=1.0)
    x = blur3d(x, [1, 2, 1])

    x = apply_noise(x) # B module
    x = apply_bias(x, lrmul=1.0)

    x = lrelu(x, 0.2)
    x = adaptive_instance_norm(x, w0) # A module

    x = conv(x, n_f, kernel=3, stride=1, gain=np.sqrt(2), lrmul=1.0)

    x = apply_noise(x) # B module
    x = apply_bias(x, lrmul=1.0)

    x = lrelu(x, 0.2)
    x = adaptive_instance_norm(x, w1) # A module

    return x

##################################################################################
# StyleGAN Etc
##################################################################################

def downscale_conv(x, channels, kernel, gain, lrmul):
    height, width = x.shape[1], x.shape[2]
    fused_scale = (min(height, width) * 2) >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        x = conv(x, channels=channels, kernel=kernel, stride=1, gain=gain, lrmul=lrmul)
        x = downscale3d(x)
        return x

    # Fused => perform both ops simultaneously using tf.nn.conv3d().
    weight = get_weight([kernel, kernel, kernel, x.get_shape().as_list()[-1], channels], gain, lrmul)
    weight = tf.pad(weight, [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    weight = tf.add_n([weight[1:, 1:, 1:], 
                       weight[:-1, 1:, 1:], 
                       weight[1:, :-1, 1:], 
                       weight[1:, 1:, :-1],
                       weight[1:, :-1, :-1],
                       weight[:-1, 1:, :-1],
                       weight[:-1, :-1, 1:],
                       weight[:-1, :-1, :-1]]) * 0.25
    
    x = tf.nn.conv3d(input=x, filters=weight, strides=[1, 2, 2, 2, 1], padding='SAME')
    return x


def upscale_conv(x, channels, kernel, gain=np.sqrt(2), lrmul=1.0):
    batch_size = tf.shape(x)[0]
    height, width, depth = x.shape[1], x.shape[2], x.shape[3]
    fused_scale = (min(height, width) * 2) >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        x = upscale3d(x)
        x = conv(x, channels=channels, kernel=kernel, stride=1, gain=gain, lrmul=lrmul)
        return x

    # Fused => perform both ops simultaneously using tf.nn.conv3d_transpose().
    weight_shape = [kernel, kernel, kernel, channels, x.get_shape().as_list()[-1]]
    output_shape = [batch_size, height * 2, width * 2, depth * 2, channels]

    weight = get_weight(weight_shape, gain, lrmul)
    weight = tf.pad(weight, [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    weight = tf.add_n([weight[1:, 1:, 1:], 
                       weight[:-1, 1:, 1:], 
                       weight[1:, :-1, 1:], 
                       weight[1:, 1:, :-1],
                       weight[:-1, :-1, 1:],
                       weight[:-1, 1:, :-1],
                       weight[1:, :-1, :-1],
                       weight[:-1, :-1, :-1]])

    x = tf.nn.conv3d_transpose(x, filters=weight, output_shape=output_shape, strides=[1, 2, 2, 2, 1], padding='SAME')

    return x


# CvL: Should no longer be used for our images, which are 
def torgb(x, res):
    x = conv(x, channels=1, kernel=1, stride=1, gain=1.0, lrmul=1.0)
    x = apply_bias(x, lrmul=1.0)
    return x

def fromrgb(x, res, n_f):
    x = conv(x, channels=n_f, kernel=1, stride=1, gain=np.sqrt(2), lrmul=1.0)
    x = apply_bias(x, lrmul=1.0)
    x = lrelu(x, 0.2)
    return x


def style_mod(x, w):
    units = x.shape[-1] * 2
    style = fully_connected(w, units=units, gain=1.0, lrmul=1.0)
    style = apply_bias(style, lrmul=1.0)

    style = tf.reshape(style, [-1, 2, 1, 1, 1, x.shape[-1]])
    
    x = x * (style[:, 0] + 1) + style[:, 1]

    return x


class AddNoise(layers.Layer):
    def __init__(self, shape):
        super(AddNoise, self).__init__()
        weight = tf.Variable(name='weight', shape=shape, initial_value=tf.zeros_initializer()(shape=shape))
        self.weight = tf.reshape(weight, [1, 1, 1, 1, -1])
        
    def call(self, inputs):
        noise = tf.random.normal([tf.shape(inputs)[0], inputs.shape[1], inputs.shape[2], inputs.shape[3], 1])
        return inputs + self.weight * noise
        
def apply_noise(x):
    shape = [x.get_shape().as_list()[-1]]
    return AddNoise(shape)(x)


class AddBias(layers.Layer):
    def __init__(self, lrmul, shape):
        super(AddBias, self).__init__()
        self.bias = tf.Variable(name='bias', shape=shape, initial_value=tf.zeros_initializer()(shape=shape)) * lrmul
    
    
    def call(self, inputs):
        
        if len(inputs.shape) == 2:
            x = inputs + self.bias
        else:
            x = inputs + tf.reshape(self.bias, [1, 1, 1, 1, -1])

        return x

def apply_bias(x, lrmul):
    shape = [x.shape[-1]]
    return AddBias(lrmul, shape)(x)


##################################################################################
# StyleGAN Official operation
##################################################################################

# ----------------------------------------------------------------------------
# CvL: tf.nn.conv3d in newer versions of tensorflow should automatically fall back to depthwise if the shape if the filter is smaller than input in shape.
# https://github.com/tensorflow/tensorflow/pull/31492
# However, it has been merged into tensorflow master, but not yet in a tagged version (and definetely not in 2.0rc0, which I have installed)
# Thus, we implement a manual solution, which according to the PR is about 30% slower or so.
# We slice manually, and then invoke the normal conv3d on the sliced inputs, before concatenating again.
def conv3d_depthwise(x, f, strides, padding):
    x = tf.split(x, x.shape[-1], axis=-1)
    filters = tf.split(f, f.shape[-2], axis=-2) # f.shape[-2], axis=-2 because the input channels are the one-but-last argument
    x = tf.concat([tf.nn.conv3d(i, f, strides=strides, padding=padding) for i, f in zip(x, filters)], axis=-1)
    return x

# Primitive ops for manipulating 4D activation tensors.
# The gradients of these are not necessary efficient or even meaningful.
def _blur3d(x, f, normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 5 and all(dim is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis, np.newaxis] * f[np.newaxis, np.newaxis, :] * f[np.newaxis, :, np.newaxis]
        
    assert f.ndim == 3    
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1, ::-1]
    f = f[:, :, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, 1, int(x.shape[-1]), 1])
    
    # No-op => early exit.
    if f.shape == (1, 1, 1) and f[0, 0, 0] == 1:
        return x
    
    # Convolve using depthwise_conv3d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv3d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, stride, stride, stride, 1]
    x = conv3d_depthwise(x, f, strides=strides, padding='SAME') # conv3d automatically falls back to depthwise if weight in shape smaller than input in shape.
    x = tf.cast(x, orig_dtype)
    return x


def _upscale3d(x, factor=2, gain=1):
    assert x.shape.ndims == 5 and all(dim is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = x.shape # [bs, h, w, d, c]
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3], 1, s[-1]])
    x = tf.tile(x, [1, 1, factor, 1, factor, 1, factor, 1])
    x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3] * factor, s[-1]])
    return x


def _downscale3d(x, factor=2, gain=1):
    assert x.shape.ndims == 5 and all(dim is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur3d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur3d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, factor, factor, factor, 1]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID')


# ----------------------------------------------------------------------------
# High-level ops for manipulating 4D activation tensors.
# The gradients of these are meant to be as efficient as possible.

def blur3d(x, f, normalize=True):
    @tf.custom_gradient
    def func(x):
        y = _blur3d(x, f, normalize)

        @tf.custom_gradient
        def grad(dy):
            dx = _blur3d(dy, f, normalize, flip=True)
            return dx, lambda ddx: _blur3d(ddx, f, normalize)

        return y, grad

    return func(x)


class Upscale3D(layers.Layer):
    def __init__(self, factor):
        super(Upscale3D, self).__init__()
        self.factor = factor
        
    def call(self, inputs):
        
        x = inputs
        
        @tf.custom_gradient
        def func(x):
            y = _upscale3d(x, self.factor)

            @tf.custom_gradient
            def grad(dy):
                dx = _downscale3d(dy, self.factor, gain=self.factor ** 2)
                return dx, lambda ddx: _upscale3d(ddx, self.factor)

            return y, grad
        
        return func(x)    

def upscale3d(x, factor=2):
    return Upscale3D(factor)(x)


class Downscale3D(layers.Layer):
    def __init__(self, factor):
        super(Downscale3D, self).__init__()
        self.factor = factor
        
    def call(self, inputs):
        x = inputs
        @tf.custom_gradient
        def func(x):
            y = _downscale3d(x, self.factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _upscale3d(dy, self.factor, gain=1 / self.factor ** 2)
                return dx, lambda ddx: _downscale3d(ddx, self.factor)

            return y, grad
        
        return func(x)

def downscale3d(x, factor=2):
    return Downscale3D(factor)(x)

class MinibatchStdLayer(layers.Layer):
    def __init__(self, group_size=4, num_new_features=1):
        super(MinibatchStdLayer, self).__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features
        
    def call(self, inputs):
        group_size = tf.minimum(self.group_size, tf.shape(inputs)[0])
        s = inputs.shape
        y = tf.reshape(inputs, [group_size, -1, self.num_new_features, s[1] // self.num_new_features, s[2], s[3], s[4]])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[2, 3, 4, 5], keepdims=True)
        y = tf.reduce_mean(y, axis=2)
        y = tf.cast(y, inputs.dtype)
        y = tf.tile(y, [group_size, s[1], s[2], s[3], 1])
        return tf.concat([inputs, y], axis=-1)


def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    return MinibatchStdLayer(group_size, num_new_features)(x)


##################################################################################
# Etc
##################################################################################

def filter_trainable_variables(res):
    res_in_focus = [2 ** r for r in range(int(np.log2(res)), 1, -1)]
    res_in_focus = res_in_focus[::-1]

    t_vars = tf.trainable_variables()
    d_vars = list()
    g_vars = list()
    for var in t_vars:
        if var.name.startswith('generator') :
            if 'g_mapping' in var.name:
                g_vars.append(var)
            elif 'g_synthesis' in var.name:
                for r in res_in_focus:
                    if '{:d}x{:d}'.format(r, r) in var.name:
                        g_vars.append(var)
        elif var.name.startswith('discriminator'):
            for r in res_in_focus:
                if '{:d}x{:d}'.format(r, r) in var.name:
                    d_vars.append(var)

    return d_vars, g_vars

def resolution_list(img_size) :

    res = 4
    x = []

    while True :
        if res > img_size :
            break
        else :
            x.append(res)
            res = res * 2

    return x

def featuremap_list(img_size) :

    start_feature_map = 512
    feature_map = start_feature_map
    x = []

    fix_num = 0

    while True :
        if img_size < 4 :
            break
        else :
            x.append(feature_map)
            img_size = img_size // 2

            if fix_num > 2 :
                feature_map = feature_map // 2

            fix_num += 1

    return x

def get_batch_sizes(gpu_num) :

    # batch size for each gpu

    if gpu_num == 1 :
        x = OrderedDict([(4, 128), (8, 128), (16, 128), (32, 64), (64, 32), (128, 16), (256, 8), (512, 4), (1024, 4)])

    elif gpu_num == 2 or gpu_num == 3 :
        x = OrderedDict([(4, 128), (8, 128), (16, 64), (32, 32), (64, 16), (128, 8), (256, 4), (512, 4), (1024, 4)])

    elif gpu_num == 4 or gpu_num == 5 or gpu_num == 6 :
        x = OrderedDict([(4, 128), (8, 64), (16, 32), (32, 16), (64, 8), (128, 4), (256, 4), (512, 4), (1024, 4)])

    elif gpu_num == 7 or gpu_num == 8 or gpu_num == 9 :
        x = OrderedDict([(4, 64), (8, 32), (16, 16), (32, 8), (64, 4), (128, 4), (256, 4), (512, 4), (1024, 4)])

    else : # >= 10
        x = OrderedDict([(4, 32), (8, 16), (16, 8), (32, 4), (64, 2), (128, 2), (256, 2), (512, 2), (1024, 2)])

    return x

def get_end_iteration(iter, max_iter, do_trans, res_list, start_res) :

    end_iter = max_iter

    for res in res_list[res_list.index(start_res):-1] :
        if do_trans[res] :
            end_iter -= iter
        else :
            end_iter -= iter // 2

    return end_iter
