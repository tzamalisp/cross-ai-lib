from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Add
from crossai.models.nn1d.layers import dropout_layer


class InceptionTimeModel(Model):
    def __init__(self, number_of_classes=1,
                 train_data_shape=None,
                 nb_filters=32, use_residual=True, use_bottleneck=True,
                 depth=6, kernel_size=41, bottleneck_size=32, drp_input=0,
                 drp_high=0, kernel_initialize="he_uniform",
                 kernel_regularize=4e-5,
                 kernel_constraint=3):
        """

        Args:
            number_of_classes (int): The number of classes (>1). Default set to 1.
            train_data_shape (tuple (int,int)): The train data shape.
            nb_filters (int): The number of nb filters.
            use_residual (bool): Whether to use a residual block or not.
            use_bottleneck (bool): Whether to use a bottleneck layer or not.
            depth (int): The depth of the network.
            kernel_size (int): The kernel size of the network.
            bottleneck_size (int): The number of output filters in the convolution.
            drp_input (float): Range 0-1.
            drp_high (float): Range 0-1.
            kernel_initialize (str): The variance scaling initializer. Default: "he_uniform".
            kernel_regularize (float or str): Can be float or str in 1e-5 format.
            Regularizer to apply a penalty on the layer's kernel.
            kernel_constraint (int): The constraint of the value of the incoming weights. Default 3.

        Returns:

        """
        number_of_classes = number_of_classes
        train_data_shape = train_data_shape
        nb_filters = nb_filters
        use_residual = use_residual
        use_bottleneck = use_bottleneck
        depth = depth
        kernel_size = kernel_size
        bottleneck_size = bottleneck_size
        drp_input = drp_input
        drp_high = drp_high
        spatial = False,
        kernel_initialize = kernel_initialize
        kernel_regularize = kernel_regularize
        if isinstance(kernel_regularize, str):
            kernel_regularize = float(kernel_regularize.replace("−", "-"))
        kernel_constraint = kernel_constraint
        activation = "softmax"

        kernel_size = kernel_size - 1

        if kernel_regularize is not None:
            kernel_regularize = l2(kernel_regularize)
        if kernel_constraint is not None:
            kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

        input_layer = Input(train_data_shape)

        x = input_layer
        input_res = input_layer

        x = dropout_layer(input_tensor=x,
                          drp_on=True,
                          drp_rate=drp_input,
                          spatial=False)

        for d in range(depth):

            x = _inception_module(input_tensor=x, use_bottleneck=use_bottleneck, bottleneck_size=bottleneck_size,
                                  activation=activation, nb_filters=nb_filters, kernel_size=kernel_size,
                                  kernel_initialize=kernel_initialize, kernel_regularize=kernel_regularize,
                                  kernel_constraint=kernel_constraint
                                  )

            if use_residual and d % 3 == 2:
                x = _shortcut_layer(input_tensor=input_res, out_tensor=x, kernel_initialize=kernel_initialize,
                                    kernel_regularize=kernel_regularize, kernel_constraint=kernel_constraint)
                input_res = x

        gap_layer = GlobalAveragePooling1D()(x)

        # Dropout
        x = dropout_layer(input_tensor=gap_layer, drp_on=True,
                          drp_rate=drp_high, spatial=spatial)

        output_layer = Dense(number_of_classes,
                             activation=activation
                             )(x)

        model = Model(inputs=input_layer, outputs=output_layer)

        self.model = model


def _shortcut_layer(input_tensor, out_tensor, kernel_initialize, kernel_regularize, kernel_constraint):
    """

    Args:
        input_tensor:
        out_tensor:
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:

    Returns:

    """
    print("shortcut filters:", out_tensor.shape[-1])
    shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                        padding="same",
                        use_bias=False,
                        kernel_initializer=kernel_initialize,
                        kernel_regularizer=kernel_regularize,
                        kernel_constraint=kernel_constraint
                        )(input_tensor)
    shortcut_y = BatchNormalization()(shortcut_y)

    x = Add()([shortcut_y, out_tensor])
    x = Activation("relu")(x)
    return x


def _inception_module(input_tensor, use_bottleneck, bottleneck_size, activation, nb_filters, kernel_size,
                      kernel_initialize, kernel_regularize, kernel_constraint, stride=1):
    """

    Args:
        input_tensor:
        use_bottleneck:
        bottleneck_size:
        activation:
        nb_filters:
        kernel_size:
        kernel_initialize:
        kernel_regularize:
        kernel_constraint:
        stride:

    Returns:

    """
    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = Conv1D(filters=bottleneck_size, kernel_size=1,
                                 padding="same", activation="linear",
                                 use_bias=False,
                                 kernel_initializer=kernel_initialize,
                                 kernel_regularizer=kernel_regularize,
                                 kernel_constraint=kernel_constraint
                                 )(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
    print("kernel size list: ", kernel_size_s)

    conv_list = []

    for i in range(len(kernel_size_s)):
        print("Inception filters: {} - kernel: {}".format(nb_filters, kernel_size_s[i]))
        conv_list.append(Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                strides=stride, padding="same", activation=activation,
                                use_bias=False,
                                kernel_initializer=kernel_initialize,
                                kernel_regularizer=kernel_regularize,
                                kernel_constraint=kernel_constraint
                                )(input_inception))

    max_pool_1 = MaxPooling1D(pool_size=3, strides=stride, padding="same")(input_tensor)

    conv_6 = Conv1D(filters=nb_filters, kernel_size=1,
                    padding="same", activation=activation,
                    use_bias=False,
                    kernel_initializer=kernel_initialize,
                    kernel_regularizer=kernel_regularize,
                    kernel_constraint=kernel_constraint
                    )(max_pool_1)

    conv_list.append(conv_6)

    x = Concatenate(axis=2)(conv_list)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)

    return x
