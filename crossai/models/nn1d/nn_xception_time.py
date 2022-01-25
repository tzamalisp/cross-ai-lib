# import necessary libraries

from tensorflow.keras.layers import Input, Dense, Conv1D, Add
from tensorflow.keras.layers import SeparableConv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras import Model  # creating the Conv-Batch Norm block
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import tensorflow_addons as tfa
# crossai imports
from crossai.ai import dropout_layer
from crossai.ai import compile_model


def xception_time(number_of_classes=None, train_data_shape=None,
                  xception_adaptive_size=50,
                  xception_adapt_ws_divide=4, n_filters=16,
                  kernel_initialize="he_uniform", kernel_regularize=1e-05,
                  kernel_constraint=3,
                  drp_input=0, drp_mid=0, drp_high=0, spatial=False,
                  optimizer="adam", lr_rate=3e-04, adam_epsilon=1e-07,
                  loss="sparse_categorical_crossentropy",
                  activation="softmax"):
    """

    Args:
        number_of_classes:
        train_data_shape:
        xception_adaptive_size:
        xception_adapt_ws_divide:
        n_filters:
        kernel_initialize: (str) The kernel initialization function
        kernel_regularize: Regularize function applied to the kernel weights matrix.
        kernel_constraint: Constraint function applied to the kernel matrix.
        drp_input:
        drp_mid:
        drp_high:
        spatial:
        optimizer:
        lr_rate:
        adam_epsilon:
        loss:
        activation:

    Returns:

    """

    # divide the number of WS with the adaptive size in case of non-divisible
    # number
    if train_data_shape[0] % xception_adaptive_size != 0:
        xception_adaptive_size = int(train_data_shape[0] /
                                     xception_adapt_ws_divide)
    else:
        xception_adaptive_size = xception_adaptive_size

    # from pprint import pprint
    # pprint(args_dict)

    if kernel_regularize is not None:
        kernel_regularize = l2(kernel_regularize)
    if kernel_constraint is not None:
        kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

    # Initiating Model Topology
    input_layer = Input(train_data_shape)

    x = dropout_layer(input_tensor=input_layer,
                      drp_on=True,
                      drp_rate=drp_input,
                      spatial=False)

    # COMPONENT 1 - Xception Block
    x = xception_block(input_tensor=x, n_filters=n_filters,
                       kernel_initialize=kernel_initialize,
                       kernel_regularize=kernel_regularize,
                       kernel_constraint=kernel_constraint)

    # COMPONENT 2
    # Head of the sequential component
    head_nf = n_filters * 32
    # transform the input with window size W to a fixed
    # length of adaptive size (default 50)
    x = tfa.layers.AdaptiveAveragePooling1D(xception_adaptive_size)(x)

    # Dropout
    x = dropout_layer(input_tensor=x, drp_on=True, drp_rate=drp_mid,
                      spatial=spatial)

    # stack 3 Conv1x1 Convolutions to reduce the time-series to
    # the number of the classes
    x_post = conv1d_block(input_tensor=x, nf=head_nf/2, drp_on=False,
                          drp_rate=0.5, spatial=True,
                          kernel_initialize=kernel_initialize,
                          kernel_regularize=kernel_regularize,
                          kernel_constraint=kernel_constraint)
    x_post = conv1d_block(input_tensor=x_post, nf=head_nf/4, drp_on=False,
                          drp_rate=0.5, spatial=True,
                          kernel_initialize=kernel_initialize,
                          kernel_regularize=kernel_regularize,
                          kernel_constraint=kernel_constraint)
    x_post = conv1d_block(input_tensor=x_post, nf=number_of_classes,
                          drp_on=False, drp_rate=0.5, spatial=True,
                          kernel_initialize=kernel_initialize,
                          kernel_regularize=kernel_regularize,
                          kernel_constraint=kernel_constraint)

    # convert the length of the input signal to 1 with the
    aap = tfa.layers.AdaptiveAveragePooling1D(1)(x_post)

    # Dropout
    aap = dropout_layer(input_tensor=aap, drp_on=True, drp_rate=drp_high,
                        spatial=spatial)

    # flatten
    flatten = Flatten()(aap)

    # output
    output_layer = Dense(number_of_classes, activation=activation)(flatten)

    # build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # compile the model
    model = compile_model(model=model,
                          optimizer=optimizer,
                          lr_rate=lr_rate,
                          adam_epsilon=adam_epsilon,
                          loss=loss)

    return model


def conv1d_block(input_tensor, nf, ks=1, strd=1, pad="same", bias=False,
                 bn=True, act=True, act_func="relu", drp_on=False,
                 drp_rate=0.5, spatial=True, kernel_initialize=None,
                 kernel_regularize=None,
                 kernel_constraint=None):
    """

    Args:
        input_tensor:
        nf:
        ks:
        strd:
        pad:
        bias:
        bn:
        act:
        act_func:
        drp_on:
        drp_rate:
        spatial:
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:

    Returns:

    """
    # Convolution
    x = Conv1D(filters=int(nf), kernel_size=ks, strides=strd,
               padding=pad, use_bias=bias,
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint
               )(input_tensor)

    # Batch Normalization
    if bn:
        x = BatchNormalization()(x)
    # Activation function
    if act:
        x = Activation(act_func)(x)

    # Dropout
    x = dropout_layer(input_tensor=x, drp_rate=drp_rate,
                      drp_on=drp_on,
                      spatial=spatial)

    return x


def xception_block(input_tensor, n_filters, depth=4, use_residual=True,
                   kernel_initialize=None, kernel_regularize=None,
                   kernel_constraint=None):
    """

    Args:
        input_tensor:
        n_filters:
        depth:
        use_residual:
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:

    Returns:

    """
    x = input_tensor
    input_res = input_tensor

    counter_drp_rate = 0

    for d in range(depth):
        if counter_drp_rate < 2:
            drp_rate = 0.1
        else:
            drp_rate = 0.25

        xception_filters = n_filters * 2 ** d
        # print("Xception Module Filters: {}".format(xception_filters))
        x = _xception_module(input_tensor=x, n_filters=xception_filters,
                             kernel_initialize=kernel_initialize,
                             kernel_regularize=kernel_regularize,
                             kernel_constraint=kernel_constraint)

        if use_residual and d % 2 == 1:
            residual_conv_filters = n_filters * 4 * (2 ** d)
            # print("Residual Filters: {}".format(residual_conv_filters))
            x = _shortcut_layer(input_res, x, residual_conv_filters,
                                kernel_initialize=kernel_initialize,
                                kernel_regularize=kernel_regularize,
                                kernel_constraint=kernel_constraint)
            input_res = x

        counter_drp_rate += 1

    return x


def _xception_module(input_tensor, n_filters, use_bottleneck=True,
                     kernel_size=41, stride=1,
                     kernel_initialize=None, kernel_regularize=None,
                     kernel_constraint=None):
    """

    Args:
        input_tensor:
        n_filters:
        use_bottleneck:
        kernel_size:
        stride:
        kernel_initialize:
        kernel_regularize
        kernel_constraint

    Returns:

    """
    # based on paper --> padding 0 : "valid", padding 1: "same"
    # FIRST PATH
    # bottleneck
    if use_bottleneck and n_filters > 1:
        print("Bottleneck filter: {}".format(n_filters))
        input_inception = Conv1D(filters=n_filters, kernel_size=1,
                                 padding="valid",
                                 use_bias=False,
                                 kernel_initializer=kernel_initialize,
                                 kernel_regularizer=kernel_regularize,
                                 kernel_constraint=kernel_constraint
                                 )(input_tensor)

    else:
        input_inception = input_tensor

    # Depth-wise Separable convolutions
    conv_list = []
    # kernels: [11, 21, 41]
    # paddings: [5, 10, 20]
    kernel_sizes, padding_sizes = kernel_padding_size_lists(kernel_size)

    for kernel, padding in zip(kernel_sizes, padding_sizes):
        print("Xception filter: {} - kernel: {}".format(n_filters, kernel))
        conv_list.append(separable_conv_1d(input_inception=input_inception,
                                           n_filters=n_filters, kernel=kernel,
                                           stride=stride,
                                           kernel_initialize=kernel_initialize,
                                           kernel_regularize=kernel_regularize,
                                           kernel_constraint=kernel_constraint)
                         )

    # SECOND PATH
    x = MaxPooling1D(pool_size=3, strides=stride,
                     padding="same")(input_tensor)

    x = Conv1D(filters=n_filters, kernel_size=1, padding="valid",
               use_bias=False,
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint
               )(x)

    conv_list.append(x)

    x_post = Concatenate(axis=2)(conv_list)

    return x_post


def separable_conv_1d(input_inception, n_filters, kernel, stride,
                      kernel_initialize, kernel_regularize, kernel_constraint):
    """

    Args:
        input_inception:
        n_filters:
        kernel:
        stride:
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:

    Returns:

    """
    x = SeparableConv1D(filters=n_filters, kernel_size=kernel,
                        strides=stride, padding="same",
                        use_bias=False,
                        kernel_initializer=kernel_initialize,
                        kernel_regularizer=kernel_regularize,
                        kernel_constraint=kernel_constraint
                        )(input_inception)

    return x


def _shortcut_layer(input_tensor, out_tensor, n_filters, kernel_initialize,
                    kernel_regularize, kernel_constraint):
    """

    Args:
        input_tensor:
        out_tensor:
        n_filters:
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:

    Returns:

    """
    if n_filters > 1:
        print("Residual Filters: {}".format(n_filters))
        x = Conv1D(filters=n_filters, kernel_size=1,
                   padding="same", use_bias=False,
                   kernel_initializer=kernel_initialize,
                   kernel_regularizer=kernel_regularize,
                   kernel_constraint=kernel_constraint
                   )(input_tensor)
    else:
        raise print("Define a number of filters.")

    shortcut_y = BatchNormalization()(x)

    x = Add()([shortcut_y, out_tensor])
    x = Activation("relu")(x)

    return x


def kernel_padding_size_lists(max_kernel_size):
    """

    Args:
        max_kernel_size:

    Returns:

    """
    i = 0
    kernel_size_list = []
    padding_list = []
    while i < 3:
        size = max_kernel_size // (2 ** i)
        if size == max_kernel_size:
            kernel_size_list.append(int(size))
            padding_list.append(int((size - 1) / 2))
        else:
            kernel_size_list.append(int(size + 1))
            padding_list.append(int(size / 2))
        i += 1

    return kernel_size_list, padding_list
