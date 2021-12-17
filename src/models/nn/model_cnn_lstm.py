import logging
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, ConvLSTM2D, Dense, Dropout, \
     Flatten,LSTM, MaxPooling1D, TimeDistributed
from src.models.nn.base_model import BaseModel


class CNN1DLSTM(BaseModel):
    def read_configuration(self, config):
        self.arch.num_of_classes = config.get("number_of_classes")
        self.arch.split_step = config.get("split_step")
        self.arch.kernel_size = config.get("kernel")
        self.arch.conv_layers = config.get("conv_layers")
        self.arch.filters = config.get("filters")
        self.arch.lstm_units = config.get("lstm_units")
        self.arch.lstm_layers = config.get("lstm_layers")
        self.arch.dense_layers = config.get("dense_layers")
        self.arch.dense_units = config.get("dense_units")
        self.arch.input_shape = config.get("dataset_shape")
        self.arch.dropout_percent = config.get("dropout")

        debug_msg_str = "model name : {}\n".format(self.model_name)
        debug_msg_str = debug_msg_str + "shaping : {}\n".format(self.arch.input_shape)
        debug_msg_str = debug_msg_str + "kernel : {}\n".format(self.arch.kernel_size)
        debug_msg_str = debug_msg_str + "filters : {}\n".format(self.arch.filters)
        debug_msg_str = debug_msg_str + "conv layers : {}\n".format(self.arch.conv_layers)
        debug_msg_str = debug_msg_str + "lstm layers : {}\n".format(self.arch.lstm_layers)
        debug_msg_str = debug_msg_str + "lstm units : {}\n".format(self.arch.lstm_units)
        debug_msg_str = debug_msg_str + "Dense units : {}\n".format(self.arch.dense_units)
        logging.debug(debug_msg_str)
        print(debug_msg_str)

    def define_model(self, config=None):
        self.read_configuration(config)

        # Reshape input into steps * length
        self.arch.input_instance_length = self.arch.input_shape[0]
        self.arch.step_length = self.arch.input_instance_length // self.arch.split_step
        print("Reshaping {} length instances into {} subsequences of length {}".
                      format(self.arch.input_instance_length,
                             self.arch.split_step,
                             self.arch.step_length))

        self.model = Sequential(name=self.model_name)
        self.model.add(
            TimeDistributed(Conv1D(filters=self.arch.filters[0],
                                   kernel_size=self.arch.kernel_size,
                                   activation='relu'),
                            input_shape=(None,
                                         self.arch.step_length,
                                         self.arch.input_shape[-1])
                            ))
        for i in range(1, self.arch.conv_layers):
            # Convolutional layers
            self.model.add(
                TimeDistributed(Conv1D(filters=self.arch.filters[i],
                                       kernel_size=self.arch.kernel_size,
                                       activation="relu")))
        self.model.add(TimeDistributed(Dropout(0.5)))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

        self.model.add(TimeDistributed(Flatten()))
        for i in range(0, self.arch.lstm_layers - 1):
            self.model.add(
                LSTM(self.arch.lstm_units[i], return_sequences=True))
        self.model.add(LSTM(units=int(self.arch.lstm_units[-1]), return_sequences=False))
        self.model.add(Dropout(0.5))
        for d in range(0, self.arch.dense_layers):
            self.model.add(Dense(units=self.arch.dense_units[d],
                                 activation="relu"))
        self.model.add(Dense(self.arch.num_of_classes, activation="softmax"))

    def fit(self, train_dataset, validation_dataset, verbose=None):
        train_dataset_reshaped = self.reshape_instance_to_steps(train_dataset[0])
        validation_dataset_reshaped = self.reshape_instance_to_steps(validation_dataset[0])
        print("Train dataset dimensions : {}".format(train_dataset_reshaped.shape))
        print("Validation dataset dimensions : {}".format(validation_dataset_reshaped.shape))
        train_dataset = (train_dataset_reshaped, train_dataset[1])
        validation_dataset = (validation_dataset_reshaped, validation_dataset[1])
        super().fit(train_dataset, validation_dataset, verbose)

    def evaluate(self, test_X, test_y):
        test_X_reshaped = self.reshape_instance_to_steps(test_X)
        return super().evaluate(test_X_reshaped, test_y)

    def predict(self, data, labels=None):
        if isinstance(data, pd.DataFrame):
            data = data.values
        data_reshaped = self.reshape_instance_to_steps(data)
        return super().predict(data_reshaped, labels)

    def reshape_instance_to_steps(self, data):
        """

        Args:
            data:

        Returns:

        """
        if self.arch.input_instance_length % self.arch.split_step > 0:
            msg = "Input instance length is not exactly divisable with the defined step."
            logging.warning(msg)

        data = data.reshape(data.shape[0], self.arch.split_step, self.arch.step_length, self.arch.input_shape[-1])
        print("New data shape : {}".format(data.shape))
        return data


class Conv1DLSTM(CNN1DLSTM):
    def define_model(self, config=None):
        self.read_configuration(config)

        # Reshape input into steps * length
        self.arch.input_instance_length = self.arch.input_shape[0]
        self.arch.step_length = self.arch.input_instance_length // self.arch.split_step
        print("Reshaping {} length instances into {} subsequences of length {}".
              format(self.arch.input_instance_length,
                     self.arch.split_step,
                     self.arch.step_length))

        self.model = Sequential(name=self.model_name)
        self.model.add(
            ConvLSTM2D(filters=self.arch.filters[0],
                       kernel_size=(1, self.arch.kernel_size),
                       activation="relu",
                       input_shape=(self.arch.split_step,
                                    1,
                                    self.arch.step_length,
                                    self.arch.input_shape[-1])
                       )
            )
        for i in range(1, self.arch.conv_layers):
            # Convolutional layers
            self.model.add(
                ConvLSTM2D(filters=self.arch.filters[i],
                           kernel_size=(1, self.arch.kernel_size),
                           activation="relu"
                           )
            )
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())

        for d in range(0, self.arch.dense_layers):
            self.model.add(Dense(units=self.arch.dense_units[d],
                                 activation="relu"))
        self.model.add(Dense(self.arch.num_of_classes, activation="softmax"))

    def reshape_instance_to_steps(self, data):
        """

        Args:
            data:

        Returns:

        """
        if self.arch.input_instance_length % self.arch.split_step > 0:
            msg = "Input instance length is not exactly divisable with the defined step."
            logging.warning(msg)

        data = data.reshape(data.shape[0], self.arch.split_step, 1, self.arch.step_length, self.arch.input_shape[-1])
        print("New data shape : {}".format(data.shape))
        return data

