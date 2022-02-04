import numpy as np
import logging
from crossai.models.nn1d.base_model import BaseModel
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
from crossai.models.nn1d.base_model import dropout_layer


class InceptionTimeModel(BaseModel):
    def define_model(self, config=None):

        number_of_classes = config.get("number_of_classes", self.number_of_classes)
        train_data_shape = config.get("dataset_shape", self.dataset_shape)
        nb_filters = config.get("nb_filters", 32)
        use_residual = config.get("use_residual", True)
        use_bottleneck = config.get("use_bottleneck", True)
        depth = config.get("depth", 6)
        kernel_size = config.get("kernel_size", 41)
        bottleneck_size = config.get("bottleneck_size", 32)
        drp_input = config.get("drp_input", 0)
        drp_high = config.get("drp_high", 0)
        spatial = False,
        kernel_initialize = config.get("kernel_initialize", "he_uniform")
        kernel_regularize = config.get("kernel_regularize", 4e-5)
        if isinstance(kernel_regularize, str):
            kernel_regularize = float(kernel_regularize.replace("−", "-"))
            config.update({"kernel_regularize": kernel_regularize})
        kernel_constraint = config.get("kernel_constraint", 3)
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
