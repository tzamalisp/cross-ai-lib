"""
Functions for performing exploratory analysis on a motion dataset. Statistics about average length,
instances of each class.
"""
import os

from utilities.visualizations import number_of_instances_barplot, number_of_instances_pie, \
    plot_distribution_of_data
from configuration_functions.project_configuration_variables import project_configuration
from pathlib import Path
import numpy as np
import logging


def pprint_dataset_statistics_dict(stats_dict):
    """
    Creates a structured text from the dictionary of the calculated statistics.
    Args:
        stats_dict (dict): A dictionary, containing the calculated statistics of the gestures
            motion dataset according to the calculation from  `generate_statistics_from_arrays` function.

    Returns:

    """
    pretty_str = ""
    pretty_str += "\nUnique gestures:\n"
    pretty_str += "\tGestureID | Mean Length (samples) | STD (+-) (samples) |" \
                  " Min Length (samples)| Max Length (samples)|\n"
    sorted_keys_list = list(stats_dict["instances_stats"].keys())
    sorted_keys_list.sort()
    for key in sorted_keys_list:
        pretty_str += "\t    {0}   |   {1:.3f}   | {2:07.3f}  |" \
                      "    {3:3d}     |    {4:4d}    | \n".format(key,
                                                                  stats_dict["instances_stats"][
                                                                      key][
                                                                      "avg_len"],
                                                                  stats_dict["instances_stats"][
                                                                      key][
                                                                      "std_len"],
                                                                  stats_dict["instances_stats"][
                                                                      key][
                                                                      "min_len"],
                                                                  stats_dict["instances_stats"][
                                                                      key][
                                                                      "max_len"]
                                                                  )
    pretty_str += "Overall dataset's gestures : \n"
    pretty_str += "\t    ALL   |   {0:.3f}   | {1:07.3f}  |" \
                  "    {2:3d}     |    {3:4d}    | \n".format(stats_dict["overall_stats"]["avg_len"],
                                                              stats_dict["overall_stats"]["std_len"],
                                                              stats_dict["overall_stats"]["min_len"],
                                                              stats_dict["overall_stats"]["max_len"]
                                                              )

    return pretty_str


def generate_dataset_statistics_plot_reports(stats_dict, report_name=None, path_to_save=None, verbose=True):
    """

    Args:
        stats_dict:
        report_name:
        path_to_save:

    Returns:

    """
    if verbose:
        logging.info("----------------")

        if report_name:
            logging.info("Report : %s", report_name)
        logging.info(pprint_dataset_statistics_dict(stats_dict))
        logging.info("----------------")
    if path_to_save:
        path_to_save = Path(path_to_save)

    if report_name:
        path_to_save = path_to_save.joinpath(report_name)
        path_to_save.mkdir(parents=True, exist_ok=True)
        # Write stats to file
        path_to_save.joinpath("statistics_report.txt").write_text(pprint_dataset_statistics_dict(stats_dict))
    # In the barplot, pass a dictionary, only with the counts field
    number_of_instances_barplot({key: value["count"] for key, value in stats_dict["instances_stats"].items()
                                 if (isinstance(value, dict) and "count" in value.keys())},
                                path_to_save=None if not path_to_save else
                                path_to_save.joinpath("number_of_categories.png"))

    number_of_instances_pie({key: value["count"] for key, value in stats_dict["instances_stats"].items()
                             if (isinstance(value, dict) and "count" in value.keys())},
                            title="All gestures' instances distribution",
                            path_to_save=None if not path_to_save else
                            path_to_save.joinpath("gestures_instances_all_distribution_piechart.png"))
    plot_distribution_of_data({"All gestures' instances length distribution": stats_dict["overall_stats"]["lengths"]},
                              description="Length", ylabel="Number of instances", bins=15,
                              path_to_save=None if not path_to_save else
                              path_to_save.joinpath("overall_gestures_instances_length_distribution.png"))
    plot_distribution_of_data({key: value["lengths"] for key, value in stats_dict["instances_stats"].items()
                               if (isinstance(value, dict) and "lengths" in value.keys())},
                              description="Instances lengths",
                              title="All gestures' instances length distribution for each category", xlabel="Length",
                              ylabel="# of instances",
                              path_to_save=None if not path_to_save else
                              path_to_save.joinpath("gestures_length_distribution.png")
                              )
    # TODO add option for plotting calculated sampling frequencies


def generate_statistics_from_documents(all_documents):
    """

    Args:
        all_documents:

    Returns:

    """
    overall_stats = dict()
    overall_stats["count"] = len(all_documents)
    overall_stats["lengths"] = list()

    instances_stats = dict()
    for doc in all_documents:
        if not doc["movementID"] in instances_stats.keys():
            instances_stats[doc["movementID"]] = dict()
            instances_stats[doc["movementID"]]["count"] = 0
            instances_stats[doc["movementID"]]["lengths"] = list()
        instances_stats[doc["movementID"]]["count"] += 1
        instances_stats[doc["movementID"]]["lengths"].append(doc.get("datalen")
                                                             if doc.get("datalen") is not None else
                                                             len(doc.get("data").get("epoch (ms)")))
        overall_stats["lengths"].append(doc.get("datalen")
                                        if doc.get("datalen") is not None else
                                        len(doc.get("data").get("epoch (ms)")))
    overall_stats.update({"avg_len": np.mean(overall_stats["lengths"])})
    overall_stats.update({"std_len": np.std(overall_stats["lengths"])})
    overall_stats.update({"max_len": np.max(overall_stats["lengths"])})
    overall_stats.update({"min_len": np.min(overall_stats["lengths"])})
    for key in instances_stats.keys():
        instances_stats[key].update({"avg_len": np.mean(instances_stats[key]["lengths"])})
        instances_stats[key].update({"std_len": np.std(instances_stats[key]["lengths"])})
        instances_stats[key].update({"max_len": np.max(instances_stats[key]["lengths"])})
        instances_stats[key].update({"min_len": np.min(instances_stats[key]["lengths"])})
    return {
        "overall_stats": overall_stats,
        "instances_stats": instances_stats
    }


def generate_statistics_from_arrays(data_array, labels_vector):
    """

    Args:
        data_array:
        labels_vector:

    Returns:

    """
    overall_stats = dict()
    overall_stats["count"] = len(data_array)
    overall_stats["lengths"] = list()
    labels = project_configuration["processing"]["dataset_labels"]
    instances_stats = dict()
    if isinstance(data_array, list):
        data_count = len(data_array)
    else:
        data_count = data_array.shape[0]
    for ind in range(0, data_count):
        doc = data_array[ind]
        label_index = int(labels_vector[ind])
        label = labels[label_index]
        if label not in instances_stats.keys():
            instances_stats[label] = dict()
            instances_stats[label]["count"] = 0
            instances_stats[label]["lengths"] = list()
        instances_stats[label]["count"] += 1
        instances_stats[label]["lengths"].append(doc.shape[0])
        overall_stats["lengths"].append(doc.shape[0])
    overall_stats.update({"avg_len": np.mean(overall_stats["lengths"])})
    overall_stats.update({"std_len": np.std(overall_stats["lengths"])})
    overall_stats.update({"max_len": np.max(overall_stats["lengths"])})
    overall_stats.update({"min_len": np.min(overall_stats["lengths"])})
    for key in instances_stats.keys():
        instances_stats[key].update({"avg_len": np.mean(instances_stats[key]["lengths"])})
        instances_stats[key].update({"std_len": np.std(instances_stats[key]["lengths"])})
        instances_stats[key].update({"max_len": np.max(instances_stats[key]["lengths"])})
        instances_stats[key].update({"min_len": np.min(instances_stats[key]["lengths"])})
    return {
        "overall_stats": overall_stats,
        "instances_stats": instances_stats
    }
