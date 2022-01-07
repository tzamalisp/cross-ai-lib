from src.models.nn import MCDropout, MCSpatialDropout1D


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
