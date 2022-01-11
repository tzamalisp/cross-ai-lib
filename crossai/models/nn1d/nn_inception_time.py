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
# crossai imports
from crossai.ai.ai_layers import dropout_layer
from crossai.ai.model_compile import compile_model
from crossai.ai.argument_checker import model_arguments_check_transform


class ClassifierInceptionTime:

    def __init__(self, number_of_classes=None, train_data_shape=None,
                 nb_filters=32, use_residual=True, use_bottleneck=True,
                 depth=6, kernel_size=41, drp_input=0, drp_high=0,
                 spatial=False, kernel_initialize="he_uniform",
                 kernel_regularize=None, kernel_constraint=None,
                 optimizer="adam", lr_rate=3e-04, adam_epsilon=1e-07,
                 loss="sparse_categorical_crossentropy",
                 activation="softmax"):

        self.number_of_classes = number_of_classes
        self.train_data_shape = train_data_shape
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.bottleneck_size = 32
        self.drp_input = drp_input
        self.drp_high = drp_high
        self.spatial = spatial
        self.kernel_initialize = kernel_initialize
        self.kernel_regularize = kernel_regularize
        self.kernel_constraint = kernel_constraint
        self.optimizer = optimizer
        self.lr_rate = lr_rate
        self.adam_epsilon = adam_epsilon
        self.loss = loss
        self.activation = activation

        if kernel_regularize is not None:
            self.kernel_regularize = l2(kernel_regularize)
        if kernel_constraint is not None:
            self.kernel_constraint = MaxNorm(max_value=kernel_constraint,
                                             axis=[0, 1])

    def _inception_module(self, input_tensor, stride=1, activation="linear"):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=self.bottleneck_size,
                                     kernel_size=1,
                                     padding="same", activation=activation,
                                     use_bias=False,
                                     kernel_initializer=self.kernel_initialize,
                                     kernel_regularizer=self.kernel_regularize,
                                     kernel_constraint=self.kernel_constraint
                                     )(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(Conv1D(filters=self.nb_filters,
                                    kernel_size=kernel_size_s[i],
                                    strides=stride, padding="same",
                                    activation=activation,
                                    use_bias=False,
                                    kernel_initializer=self.kernel_initialize,
                                    kernel_regularizer=self.kernel_regularize,
                                    kernel_constraint=self.kernel_constraint
                                    )(input_inception))

        max_pool_1 = MaxPooling1D(pool_size=3,
                                  strides=stride, padding="same")(input_tensor)

        conv_6 = Conv1D(filters=self.nb_filters, kernel_size=1,
                        padding="same", activation=activation,
                        use_bias=False,
                        kernel_initializer=self.kernel_initialize,
                        kernel_regularizer=self.kernel_regularize,
                        kernel_constraint=self.kernel_constraint
                        )(max_pool_1)

        conv_list.append(conv_6)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation="relu")(x)

        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                            padding="same",
                            use_bias=False,
                            kernel_initializer=self.kernel_initialize,
                            kernel_regularizer=self.kernel_regularize,
                            kernel_constraint=self.kernel_constraint
                            )(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_tensor])
        x = Activation("relu")(x)
        return x

    def build_model(self):
        input_layer = Input(self.train_data_shape)

        x = input_layer
        input_res = input_layer

        x = dropout_layer(input_tensor=x,
                          drp_on=True,
                          drp_rate=self.drp_input,
                          spatial=False)

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = GlobalAveragePooling1D()(x)

        # Dropout
        x = dropout_layer(input_tensor=gap_layer, drp_on=True,
                          drp_rate=self.drp_high, spatial=self.spatial)

        output_layer = Dense(self.number_of_classes,
                             activation=self.activation
                             )(x)

        model = Model(inputs=input_layer, outputs=output_layer)

        model = compile_model(model=model,
                              optimizer=self.optimizer,
                              lr_rate=self.lr_rate,
                              adam_epsilon=self.adam_epsilon,
                              loss=self.loss)

        return model
