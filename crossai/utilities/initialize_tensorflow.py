"""
Functions related to tensorflow initialization.
"""
import os
import tensorflow as tf
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def initialize_tensorflow_configuration(tf_memory_limit=None, gpuid=None,
                                        device="logical",
                                        use_cpu=False, cpu_threads=None):
    """
    Snippet, which restricts the complete GPU memory allocation to one
    program/notebook especially in cases of running
    in a shared resources server. Memory allocation value is defined in
    tf_memory_limit argument.
    Args:
        device:
        tf_memory_limit (int, optional): When defined, tf allocated memory is
        restricted.
        gpuid (int): If more than one GPU, select on which GPU to run. Default
        is None (run to all GPUs).
        use_cpu (boolean): If set to True, the GPU configuration is ignored and
        only CPU is used.
        cpu_threads(int, optional): If set, te CPU threads used are restricted.

    Returns:

    """

    # Suppress Tensorflow printouts
    tf.get_logger().setLevel("INFO")

    gpus = tf.config.experimental.list_physical_devices("GPU")
    logging.info(
        "Tensorflow initialization."
        " Memory limit: {}".format(-1 if tf_memory_limit is None
                                   else tf_memory_limit))
    if use_cpu:
        print("Using CPU for execution.")
        # Hide GPU from visible devices
        tf.config.set_visible_devices([], "GPU")
        if cpu_threads:
            tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
            tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)
    else:
        if gpus:

            print("Physical GPUs {}".format(len(gpus)))
            # Restrict TensorFlow to only allocate 1GB of memory on the first
            # GPU
            try:
                if gpuid:
                    tf.config.set_visible_devices(gpus[gpuid], 'GPU')
                if tf_memory_limit is not None:
                    print("TF memory limit : {}".format(tf_memory_limit))
                    if device == "logical":
                        tf.config.set_logical_device_configuration(
                            gpus[gpuid if gpuid else 0],
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=tf_memory_limit)])
                    else:
                        tf.config.experimental. \
                            set_virtual_device_configuration(
                             gpus[gpuid if gpuid else 0],
                             [tf.config.experimental.
                                 VirtualDeviceConfiguration(
                                  memory_limit=tf_memory_limit)])
                logical_gpus = tf.config.experimental.list_logical_devices(
                    "GPU")
                print("Logical GPUs {}".format(len(logical_gpus)))

            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
