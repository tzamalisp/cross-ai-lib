import logging
from tensorflow import keras
from tensorflow.keras.layers import Layer


class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


class MCSpatialDropout1D(keras.layers.SpatialDropout1D):
    def call(self, inputs):
        return super().call(inputs, training=True)


class DropoutLayer(Layer):
    """
    This class creates a Droupout layer for a model.
    """
    def __init__(self, drp_rate=0.1, spatial=True):
        super(DropoutLayer, self).__init__()
        self.drp_rate = drp_rate
        self.spatial = spatial
        if spatial is True:
            self.drp = MCSpatialDropout1D(drp_rate)
        else:
            self.drp = MCDropout(drp_rate)

    def call(self, inputs):
        return self.drp(inputs)

    def get_config(self):
        return {"drp_rate": self.drp_rate,
                "spatial": self.spatial}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def dropout_layer(input_tensor, drp_on=True, drp_rate=0.1, spatial=True):
    """

    Args:
        input_tensor:
        drp_on:
        drp_rate:
        spatial:

    Returns:

    """
    if drp_on is True:
        if spatial is True:
            x = MCSpatialDropout1D(drp_rate)(input_tensor)
            # print("MC Spatial Dropout Rate: {}".format(drp_rate))
        else:
            x = MCDropout(drp_rate)(input_tensor)
            # print("MC Dropout Rate: {}".format(drp_rate))
    else:
        x = input_tensor

    return x