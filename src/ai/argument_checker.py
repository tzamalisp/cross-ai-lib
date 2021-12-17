kernel_initializers_valid_list = ["he_uniform"]
optimizers_valid_list = ["adam"]


def model_arguments_check_transform(config):
    """

    Args:
        config:

    Returns:
        The configuration data with the checked and transformed values
    """

    # DATA RELATED
    # raise exception in case the number of classes or the train data shape is not defined to the classifier params
    if config["clf_params"]["number_of_classes"] is None:
        print("Please fill the number of classes parameter to the config as config['clf_params']['number_of_classes']")
        raise Exception
    if config["clf_params"]["train_data_shape"] is None:
        print("Please fill the training data shape to the config as config['clf_params']['train_data_shape']")
        raise Exception

    # KERNEL
    # ensure the kernel initializer is of valid form and inside the list of accepted kernel initializers
    for initializer in config["param_grid_train"]["kernel_initialize"]:
        if initializer not in kernel_initializers_valid_list:
            print("Please provide a valid kernel initializer.")
            raise Exception
    # ensure the kernel regularize is a float value if not None, and create the L2 Regularization
    for idx, regularize in enumerate(config["param_grid_train"]["kernel_regularize"]):
        if regularize is not None:
            config["param_grid_train"]["kernel_regularize"][idx] = float(regularize)
    # ensure the kernel constraint is an integer value if not None, and create the MaxNorm constraint
    for idx, constraint in enumerate(config["param_grid_train"]["kernel_constraint"]):
        if constraint is not None:
            config["param_grid_train"]["kernel_constraint"][idx] = int(constraint)

    # OPTIMIZER
    # ensure the optimizer is valid and inside the list of accepted optimizers
    for optimizer in config["param_grid_train"]["optimizer"]:
        if optimizer not in optimizers_valid_list:
            print("Provide a valid optimizer.")
            raise Exception
    # ensure the learning rate is a valid float value and not None
    for idx, lr in enumerate(config["param_grid_train"]["lr_rate"]):
        if lr is not None:
            config["param_grid_train"]["lr_rate"][idx] = float(lr)
    # ensure that if optimizer is the Adam to check the relevant epsilon value
    for idx, epsilon in enumerate(config["param_grid_train"]["adam_epsilon"]):
        if "adam" in config["param_grid_train"]["optimizer"] and epsilon is not None:
            config["param_grid_train"]["adam_epsilon"][idx] = float(epsilon)

    return config
