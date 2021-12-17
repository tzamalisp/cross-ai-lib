from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import regularizers
# cross-ai imports
from tools.ai.ai_layers import dropout_layer
from tools.ai.model_compile import compile_model


def bi_lstm(number_of_classes, train_data_shape, drp_input=0, drp_mid=0, drp_high=0,
            kernel_initialize="he_uniform", kernel_regularize=1e-05, kernel_constraint=3,
            optimizer="adam", lr_rate=3e-04, adam_epsilon=1e-07, loss="sparse_categorical_crossentropy",
            activation="softmax"):

    if kernel_regularize is not None:
        kernel_regularize = l2(kernel_regularize)
    if kernel_constraint is not None:
        kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

    model = Sequential()

    model.add(InputLayer(input_shape=train_data_shape))

    model.add(Dropout(drp_input))

    # 1st hidden layer
    model.add(Bidirectional(LSTM(units=train_data_shape[0],
                                 return_sequences=True,
                                 kernel_initializer=kernel_initialize,
                                 kernel_regularizer=kernel_regularize,
                                 kernel_constraint=kernel_constraint)))
    model.add(Activation("tanh"))
    # 2nd hidden layer
    model.add(Bidirectional(LSTM(units=train_data_shape[0],
                                 return_sequences=True,
                                 kernel_initializer=kernel_initialize,
                                 kernel_regularizer=kernel_regularize,
                                 kernel_constraint=kernel_constraint)))
    model.add(Activation("tanh"))
    # 3rd hidden layer
    model.add(Bidirectional(LSTM(units=train_data_shape[0],
                                 return_sequences=True,
                                 kernel_initializer=kernel_initialize,
                                 kernel_regularizer=kernel_regularize,
                                 kernel_constraint=kernel_constraint)))
    model.add(Activation("tanh"))
    # 4th hidden layer
    model.add(Bidirectional(LSTM(units=train_data_shape[0],
                                 return_sequences=False,
                                 kernel_initializer=kernel_initialize,
                                 kernel_regularizer=kernel_regularize,
                                 kernel_constraint=kernel_constraint)))
    model.add(Activation("tanh"))

    model.add(Dropout(drp_mid))

    # Dense layer
    model.add(Dense(units=train_data_shape[0],
                    kernel_initializer=kernel_initialize,
                    kernel_regularizer=kernel_regularize,
                    kernel_constraint=kernel_constraint))
    model.add(Activation("relu"))

    model.add(Dropout(drp_high))

    # Output layer
    model.add(Dense(units=number_of_classes,
                    activation=activation))

    # compile the model
    model = compile_model(model=model,
                          optimizer=optimizer,
                          lr_rate=lr_rate,
                          adam_epsilon=adam_epsilon,
                          loss=loss)

    return model
