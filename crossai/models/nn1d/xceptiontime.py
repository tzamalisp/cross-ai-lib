import logging
from tensorflow.keras.layers import Input, Dense, Conv1D, Add
from tensorflow.keras.layers import SeparableConv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras import Model  # creating the Conv-Batch Norm block
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import tensorflow_addons as tfa
from crossai.models.nn1d.layers import DropoutLayer


class XceptionTime(Model):
    def __init__(self, train_data_shape, number_of_classes, xception_adaptive_size=50, xception_adapt_ws_divide=4,
                 n_filters=16, kernel_initialize="he_uniform", kernel_regularize=1e-5,
                 kernel_constraint=3, drp_input=0, drp_mid=0, drp_high=0,
                 spatial=False, activation="softmax"):
        """

        Args:
            number_of_classes (int): The number of classes. Default: 1.
            train_data_shape (tuple (int, int)): The shape the train data.
            xception_adaptive_size (int): The adaptive size. Default: 50.
            xcpetion_adapt_ws_divide (int): The number that will divide the
            adaptive size. Default 4.
            n_filters (int): The number of filters. Default: 16
            kernel_initialize (str): The variance scaling initializer.
            Default: "he_uniform".
            kernel_regularize (float or str): Can be float or str in 1e-5
            format.
            Regularizer to apply a penalty on the layer's kernel.
            kernel_constraint (int): The constraint of the value of the
            incoming weights. Default 3.
            drp_input (float): Fraction of the input units to drop in the input
            dropout layer. Default: 0.
            drp_mid (float): Fraction of the input units to drop in the mid
            dropout layer. Default: 0.
            drp_high (float): Fraction of the input units to drop in the last
            dropout layer. Default: 0.
            activation (str): The activation function.

        Returns:

        """
        super(XceptionTime, self).__init__()
        self.train_data_shape = train_data_shape
        self.number_of_classes = number_of_classes
        self.xception_adaptive_size = xception_adaptive_size
        self.xception_adapt_ws_divide = xception_adapt_ws_divide
        self.n_filters = n_filters
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint
        self.drp_input = drp_input
        self.drp_mid = drp_mid
        self.drp_high = drp_high
        self.spatial = spatial
        self.activation = activation

        logging.info(self.train_data_shape[0])
        logging.info(self.xception_adapt_ws_divide)
        if self.train_data_shape[0] % self.xception_adapt_ws_divide == 0:
            xception_adaptive_size = int(self.train_data_shape[0] / self.xception_adapt_ws_divide)
        else:
            xception_adaptive_size = self.xception_adaptive_size
            print("Provide a dividable number for the window size.")
            raise Exception("Provide a dividable number for the window size.")
        print("Input size W of window transformed into a fixed length of {} sample "
              "for AAP mid layer:".format(xception_adaptive_size))

        kernel_regularize = None
        if self.kernel_regularize is not None:
            kernel_regularize = l2(self.kernel_regularize)
        kernel_constraint = None
        if self.kernel_constraint is not None:
            kernel_constraint = MaxNorm(max_value=self.kernel_constraint, axis=[0, 1])

        # Initiating Model Topology

        self.drp0 = DropoutLayer(drp_rate=self.drp_input,
                                 spatial=False)

        # COMPONENT 1 - Xception Block
        self.xception_block0 = XceptionBlock(n_filters=self.n_filters,
                                             kernel_initialize=self.kernel_initialize,
                                             kernel_regularize=kernel_regularize,
                                             kernel_constraint=kernel_constraint)

        # COMPONENT 2
        # Head of the sequential component
        head_nf = self.n_filters * 32
        # transform the input with window size W to a fixed length of adaptive size (default 50)
        self.aap0 = tfa.layers.AdaptiveAveragePooling1D(xception_adaptive_size)

        # Dropout
        self.drp1 = DropoutLayer(drp_rate=self.drp_mid, spatial=self.spatial)

        # stack 3 Conv1x1 Convolutions to reduce the time-series to the number of the classes
        self.x_post0 = Conv1DBlock(nf=head_nf / 2, drp_on=False, drp_rate=0.5, spatial=True,
                                   kernel_initialize=self.kernel_initialize,
                                   kernel_regularize=kernel_regularize,
                                   kernel_constraint=kernel_constraint)
        self.x_post1 = Conv1DBlock(nf=head_nf / 4, drp_on=False, drp_rate=0.5, spatial=True,
                                   kernel_initialize=self.kernel_initialize,
                                   kernel_regularize=kernel_regularize,
                                   kernel_constraint=kernel_constraint)
        self.x_post2 = Conv1DBlock(nf=self.number_of_classes, drp_on=False, drp_rate=0.5, spatial=True,
                                   kernel_initialize=self.kernel_initialize,
                                   kernel_regularize=kernel_regularize,
                                   kernel_constraint=kernel_constraint)

        # convert the length of the input signal to 1 with the
        self.aap1 = tfa.layers.AdaptiveAveragePooling1D(1)

        # Dropout
        self.drp2 = DropoutLayer(drp_rate=self.drp_high, spatial=self.spatial)

        # flatten
        self.flat0 = Flatten()

        # output
        self.out = Dense(self.number_of_classes, activation=self.activation)

    def call(self, inputs):
        x = self.drp0(inputs)
        x = self.xception_block0(x)
        x = self.drp1(x)
        x = self.aap0(x)
        x = self.x_post0(x)
        x = self.x_post1(x)
        x = self.x_post2(x)
        x = self.drp2(x)
        x = self.aap1(x)
        x = self.flat0(x)
        x = self.out(x)

        return x

    def get_config(self):
        return {"train_data_shape": self.train_data_shape,
                "number_of_classes": self.number_of_classes,
                "xception_adaptive_size": self.xception_adaptive_size,
                "xception_adapt_ws_divide": self.xception_adapt_ws_divide,
                "n_filters": self.n_filters,
                "kernel_initialize": self.kernel_initialize,
                "kernel_regularize": self.kernel_regularize,
                "kernel_constraint": self.kernel_constraint,
                "drp_input": self.drp_input,
                "drp_mid": self.drp_mid,
                "drp_high": self.drp_high,
                "spatial": self.spatial,
                "activation": self.activation
                }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Conv1DBlock(Layer):
    def __init__(self, nf, ks=1, strd=1, pad="same", bias=False, bn=True, act=True, act_func="relu",
                 drp_on=False, drp_rate=0.5, spatial=True, kernel_initialize=None, kernel_regularize=None,
                 kernel_constraint=None):
        super(Conv1DBlock, self).__init__()
        self.nf = nf
        self.ks = ks
        self.strd = strd
        self.pad = pad
        self.bias = bias
        self.bn = bn
        self.act = act
        self.act_func = act_func
        self.drp_on = drp_on
        self.drp_rate = drp_rate
        self.spatial = spatial
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint

        self.conv1d = Conv1D(filters=int(nf), kernel_size=ks, strides=strd,
                             padding=pad, use_bias=bias,
                             kernel_initializer=kernel_initialize,
                             kernel_regularizer=kernel_regularize,
                             kernel_constraint=kernel_constraint)

        self.batch_norm_layer = None
        if self.bn:
            self.batch_norm_layer = BatchNormalization()

        self.activation_layer = None
        if self.act:
            self.activation_layer = Activation(act_func)

        self.drp_layer = None
        if self.drp_on:
            self.drp_layer = DropoutLayer(drp_rate, spatial)

    def call(self, inputs):
        x = self.conv1d(inputs)
        if self.bn:
            x = self.batch_norm_layer(x)
        if self.act:
            x = self.activation_layer(x)
        if self.drp_on:
            x = self.drp_layer(x)
        return x

    def get_config(self):
        return {"nf": self.nf,
                "ks": self.ks,
                "pad": self.pad,
                "bias": self.bias,
                "bn": self.bn,
                "act": self.act,
                "act_func": self.act_func,
                "drp_on": self.drp_on,
                "drp_rate": self.drp_rate,
                "spatial": self.spatial,
                "kernel_initialize": self.kernel_initialize,
                "kernel_regularize": self.kernel_regularize,
                "kernel_constraint": self.kernel_constraint}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class XceptionBlock(Layer):
    def __init__(self, n_filters, depth=4, use_residual=True, kernel_initialize=None, kernel_regularize=None,
                 kernel_constraint=None):
        super(XceptionBlock, self).__init__()
        self.n_filters = n_filters
        self.use_residual = use_residual
        self.depth = depth
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint

        self.xception_modules_list = list()
        self.residual_conv_list = list()
        self.bn_list = list()
        self.act_list = list()
        self.add_list = list()


        if self.n_filters <= 1:
            msg = "Invalid Number of filters {}. Define a number of filters.".format(n_filters)
            logging.error(msg)
            raise Exception(msg)

        for d in range(self.depth):
            xception_filters = n_filters * 2 ** d
            xception_module = XceptionModule(n_filters=xception_filters,
                                             kernel_initialize=kernel_initialize,
                                             kernel_regularize=kernel_regularize,
                                             kernel_constraint=kernel_constraint)
            self.xception_modules_list.append(xception_module)

            if self.use_residual and d % 2 == 1:
                residual_conv_filters = n_filters * 4 * (2 ** d)
                print("Residual Filters: {}".format(residual_conv_filters))
                residual_conv = Conv1D(filters=residual_conv_filters,
                                       kernel_size=1,
                                       padding="same", use_bias=False,
                                       kernel_initializer=kernel_initialize,
                                       kernel_regularizer=kernel_regularize,
                                       kernel_constraint=kernel_constraint)
                self.residual_conv_list.append(residual_conv)
                self.bn_list.append(BatchNormalization())
                self.add_list.append(Add())
                self.act_list.append(Activation("relu"))

    def call(self, inputs):
        x = inputs
        input_res = inputs

        rc = 0
        # Residual counter for iterating shortcut conv layers list
        # (residual_conv_list)
        for d in range(self.depth):
            x = self.xception_modules_list[d](x)

            if self.use_residual and d % 2 == 1:
                # Residual blocks
                res_out = self.residual_conv_list[rc](input_res)
                shortcut_y = self.bn_list[rc](res_out)
                res_out = self.add_list[rc]([shortcut_y, x])
                x = self.act_list[rc](res_out)
                rc += 1
                input_res = x

        return x

    def get_config(self):
        return {"n_filters": self.n_filters,
                "use_residual": self.use_residual,
                "depth": self.depth,
                "kernel_initialize": self.kernel_initialize,
                "kernel_regularize": self.kernel_regularize,
                "kernel_constraint": self.kernel_constraint}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class XceptionModule(Layer):
    def __init__(self, n_filters, use_bottleneck=True, kernel_size=41, stride=1,
                 kernel_initialize=None, kernel_regularize=None, kernel_constraint=None):
        super(XceptionModule, self).__init__()
        self.n_filters = n_filters
        self.use_bottleneck = use_bottleneck
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint
        print("Bottleneck filter: {}".format(n_filters))
        self.inception_conv = Conv1D(filters=n_filters, kernel_size=1, padding="valid",
                                     use_bias=False,
                                     kernel_initializer=kernel_initialize,
                                     kernel_regularizer=kernel_regularize,
                                     kernel_constraint=kernel_constraint)
        # kernels: [11, 21, 41]
        # paddings: [5, 10, 20]
        kernel_sizes, padding_sizes = kernel_padding_size_lists(kernel_size)

        self.conv_list = list()
        for kernel, padding in zip(kernel_sizes, padding_sizes):
            print("Xception filter: {} - kernel: {}".format(n_filters, kernel))
            separable_conv_1d = SeparableConv1D(filters=n_filters, kernel_size=kernel,
                                                strides=self.stride, padding="same",
                                                use_bias=False,
                                                kernel_initializer=kernel_initialize,
                                                kernel_regularizer=kernel_regularize,
                                                kernel_constraint=kernel_constraint)
            self.conv_list.append(separable_conv_1d)

        self.max_pool = MaxPooling1D(pool_size=3, strides=self.stride, padding="same")
        self.conv1d_2 = Conv1D(filters=n_filters, kernel_size=1, padding="valid", use_bias=False,
                               kernel_initializer=kernel_initialize,
                               kernel_regularizer=kernel_regularize,
                               kernel_constraint=kernel_constraint)
        self.concat = Concatenate(axis=2)

    def call(self, inputs):
        if self.use_bottleneck and self.n_filters > 1:
            x = self.inception_conv(inputs)
        else:
            x = inputs
        separable_conv_list = list()
        for separable_conv_1d in self.conv_list:
            separable_conv_list.append(separable_conv_1d(x))

        # SECOND PATH
        x2 = self.max_pool(inputs)
        x2 = self.conv1d_2(x2)
        separable_conv_list.append(x2)

        # Concatenate
        x_post = self.concat(separable_conv_list)

        return x_post

    def get_config(self):
        return {"n_filters": self.n_filters,
                "use_bottleneck": self.use_bottleneck,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "kernel_initialize": self.kernel_initialize,
                "kernel_regularize": self.kernel_regularize,
                "kernel_constraint": self.kernel_constraint}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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
