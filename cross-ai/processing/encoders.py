def encode_categorical_to_numerical(labels):
    """

    Args:
        labels: (dataframe) The dataframe with the target labels in str format.

    Returns: (np.array) Returns an array with the labels in int format.

    """
    unique_labels_list = labels.unique()
    integer_ids = []
    i = 0
    while i < len(list(unique_labels_list)):
        integer_ids.append(i)
        i += 1
    labels_int_ids = labels.replace(to_replace=unique_labels_list,
                                    value=integer_ids)
    return labels_int_ids