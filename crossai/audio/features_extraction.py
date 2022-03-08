import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction, \
    beat_extraction
from pathlib import PurePath
import librosa as lb
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features_from_waveform
import json
import pandas as pd
import os
from tqdm import tqdm


def long_feature_wav(wav_file, mid_window, mid_step,
                     short_window, short_step,
                     accept_small_wavs=False,
                     compute_beat=True, features_to_compute=["ann", "libr", "surf"]):
    """
    This function computes the long-term features per WAV file.
    Very useful to create a collection of json files (1 song -> 1 json).
    Genre as a feature should be added (very simple).
    Args:
        wav_file (str): The path to the WAV file.
        mid_window (int): The mid-term window (in seconds).
        mid_step (int): The mid-term step (in seconds).
        short_window (int): The short-term window (in seconds).
        short_step (int): The short-term step (in seconds).
        accept_small_wavs (boolean): Whether to accept small WAVs or not.
        compute_beat (boolean): Whether to compute beat related features or not. These features are only computed when
        "ann" is selected in the features_to_compute argument.
        features_to_compute (list of str): A list of the features that will be computed. The only options are "ann", "libr"
        and "surf" or a combination of them. For more details check out the library's README.md. All options are enabled
        by default.
    Returns:
        mid_term_feaures (np.array): The feature vector of a singular wav file.
        mid_feature_names (list): The feature names, useful for formatting.
    """

    mid_term_features = np.array([])
    mid_feature_names = []
    for feature in features_to_compute:
        if feature not in ["ann", "libr", "surf"]:
            raise ValueError("The option {} does not exist or is misspelled".format(feature))
    sampling_rate, signal = audioBasicIO.read_audio_file(wav_file)
    if sampling_rate == 0:
        return -1

    signal = audioBasicIO.stereo_to_mono(signal)

    size_tolerance = 5
    if accept_small_wavs:
        size_tolerance = 100
    if signal.shape[0] < float(sampling_rate) / size_tolerance:
        print("  (AUDIO FILE TOO SMALL - SKIPPING)")
        return -1
    if "ann" in features_to_compute:
        if compute_beat:
            mid_features, short_features, mid_feature_names = \
                mid_feature_extraction(signal, sampling_rate,
                                       round(mid_window * sampling_rate),
                                       round(mid_step * sampling_rate),
                                       round(sampling_rate * short_window),
                                       round(sampling_rate * short_step))
            beat, beat_conf = beat_extraction(short_features, short_step)
        else:
            mid_features, _, mid_feature_names = \
                mid_feature_extraction(signal, sampling_rate,
                                       round(mid_window * sampling_rate),
                                       round(mid_step * sampling_rate),
                                       round(sampling_rate * short_window),
                                       round(sampling_rate * short_step))

        mid_features = np.transpose(mid_features)
        mid_features = mid_features.mean(axis=0)
        # long term averaging of mid-term statistics
        if (not np.isnan(mid_features).any()) and \
                (not np.isinf(mid_features).any()):
            if compute_beat:
                mid_features = np.append(mid_features, beat)
                mid_features = np.append(mid_features, beat_conf)
                mid_feature_names.append("beat")
                mid_feature_names.append("beat_conf")

        # Block of code responsible for extra features

    if "libr" in features_to_compute:
        librosa_feat, librosa_feat_names = compute_libr_features(
            wav_file, sampling_rate=sampling_rate)
        mid_features = np.append(mid_features, librosa_feat)
        for element in librosa_feat_names:
            mid_feature_names.append(element)

    if "surf" in features_to_compute:
        surfboard_feat, surfboard_feat_names = compute_surf_features(
            wav_file, sampling_rate=sampling_rate)
        mid_features = np.append(mid_features, surfboard_feat)
        for element in surfboard_feat_names:
            mid_feature_names.append(element)

    if len(mid_term_features) == 0:
        # append feature vector
        mid_term_features = mid_features
    else:
        mid_term_features = np.vstack((mid_term_features, mid_features))

    return mid_term_features, mid_feature_names


def features_to_json(root_path, file_name, save_location, m_win, m_step, s_win, s_step,
                     accept_small_wavs, compute_beat, features_to_compute=["ann", "libr", "surf"]):
    """
    Function that saves the features returned from long_feature_wav
    to a json file. This functions operates on a singular wav file.
    The function will automatically add the root folder name as the 'label' field in the json.
    Args:
        root_path (str): absolute path of the dataset, useful for audio loading.
        file_name (str): Self explanatory.
        save_location (str): Self explanatory.
        m_win (int): The mid-term window (in seconds).
        m_step (int): The mid-term step (in seconds).
        s_win (int): The short-term window (in seconds).
        s_step (int): The short-term step (in seconds).
        accept_small_wavs (boolean): Whether to accept small WAVs or not.
        compute_beat (boolean): Whether to compute beat related features or not. These features are only computed when
        "ann" is selected in the features_to_compute argument.
        features_to_compute (list of str): A list of the features that will be computed. The only options are "ann", "libr"
        and "surf" or a combination of them. For more details check out the library's README.md. All options are enabled
        by default.
    Returns:
        json_file_name (str): The path of the json file that contains the features.
    """

    long_feature_return = long_feature_wav(root_path + '/' + file_name, m_win,
                                           m_step,
                                           s_win, s_step, accept_small_wavs,
                                           compute_beat, features_to_compute)

    if long_feature_return == -1:
        return -1

    feature_values, feature_names = long_feature_return
    json_data = dict(zip(feature_names, feature_values))

    # Adding the genre tag to the json dictionary, using pathlib for simplicity
    p = PurePath(root_path)
    label = p.name
    json_data['label'] = label

    json_file_name = save_location + '/' + file_name + '.json'
    with open(json_file_name, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    del json_data
    return json_file_name


def compute_libr_features(filename, sampling_rate=22050):
    """
    Function that extracts the additional Librosa features
    Args:
      filename (str): THe name of the WAV file.
      sampling_rate (int): Used because pyAudioAnalysis uses different sampling rate
      for each wav file.

    Returns:
      features (np.array): the calculated features, returned as numpy array for consistency (1 x 12).
      feature_names (list): The feature names for consistency and pandas formatting (1 x 12).
    """

    y, sr = lb.load(filename, sr=sampling_rate)

    feature_names = ["spectral_bandwidth_mean", "spectral_flatness_mean",
                     "spectral_rms_mean",
                     "spectral_bandwidth_std", "spectral_flatness_std",
                     "spectral_rms_std",
                     "spectral_bandwidth_delta_mean",
                     "spectral_bandwidth_delta_std",
                     "spectral_flatness_delta_mean",
                     "spectral_flatness_delta_std",
                     "spectral_rms_delta_mean", "spectral_rms_delta_std"]

    features = []
    calculations = []
    calculations.append(lb.feature.spectral_bandwidth(y=y, sr=sr))
    calculations.append(lb.feature.spectral_flatness(y=y))
    calculations.append(lb.feature.rms(y=y))

    for c in calculations:
        features.append(np.mean(c))
        features.append(np.std(c))
        features.append(np.mean(lb.feature.delta(c)))
        features.append(np.std(lb.feature.delta(c)))

    return np.array(features), feature_names


def compute_surf_features(filename, sampling_rate=44100):
    """
    Function that extracts the additional Surfboard features.
    Args:
       filename (str): The name of the wav file
       sampling_rate (int): Used because pyAudioAnalysis uses different sampling rate
       for each wav file

    Returns:
       feature_values (np.array): the calculated features, returned as numpy array for consistency (1 x 13)
       feature_names (list): the feature names for consistency and pandas formating (1 x 13)
    """

    sound = Waveform(path=filename, sample_rate=sampling_rate)

    features_list = ['spectral_kurtosis', 'spectral_skewness',
                     'spectral_slope',
                     'loudness']  # features can also be specified in a yaml file

    # extract features with mean, std, dmean, dstd stats. Stats are computed on the spectral features. Loudness is just a scalar
    feature_dict = extract_features_from_waveform(features_list,
                                                  ['mean', 'std',
                                                   'first_derivative_mean',
                                                   'first_derivative_std'],
                                                  sound)
    # convert to df first for consistency
    feature_dataframe = pd.DataFrame([feature_dict])
    # Surfboard exports features into dataframes. We convert the dataframe columns into a list and the row into a numpy array, for consistency.

    feature_values = feature_dataframe.to_numpy()
    feature_names = list(feature_dataframe.columns)

    return feature_values, feature_names


def create_features_directory(dir_path, m_win, m_step, s_win, s_step,
                              accept_small_wavs, compute_beat, features_to_compute=["ann", "libr", "surf"]):
    """
    Function that generates an identical directory structure to the one of the input WAV files
    that contains the feature json files of each audio file. The name of the new directory will be the same as
    dir_path argument with the suffix "_features" added to it.

    Args:
        dir_path (str): The path of the directory that contains the audio files.
        m_win (int): The mid-term window (in seconds).
        m_step (int): The mid-term step (in seconds).
        s_win (int): The short-term window (in seconds).
        s_step (int): The short-term step (in seconds).
        accept_small_wavs (boolean): Whether to accept small WAVs or not.
        compute_beat (boolean): Whether to compute beat related features or not.
        compute_beat (boolean): Whether to compute beat related features or not. These features are only computed when
        "ann" is selected in the features_to_compute argument.
        features_to_compute (list of str): A list of the features that will be computed. The only options are "ann", "libr"
        and "surf" or a combination of them. For more details check out the library's README.md. All options are enabled
        by default.

    Returns:
        None
    """
    file_suffix = '.wav'
    directory_path = dir_path
    music_files = []
    path_obj = PurePath(directory_path)
    folder_name = path_obj.name
    for root, _, files in os.walk(directory_path):
        new_folder = root.replace(folder_name, folder_name+'_features')
        if not os.path.isdir(new_folder):
            os.mkdir(new_folder)
        for file_element in files:
            if file_element.endswith(file_suffix):
                music_files.append((root, file_element))
    music_files = tqdm(music_files, desc='Extracting features...')
    for element in music_files:
        root, file_name = element
        save_folder = root.replace(folder_name, folder_name+'_features')
        features_to_json(root, file_name, save_folder, m_win, m_step, s_win, s_step,
                         accept_small_wavs, compute_beat, features_to_compute)


def features_dataframe_builder(json_directory):
    """
    Given a directory, this function will
    create a list of dicts from the json files
    in the directory. Based on this list a dataframe
    that describes the whole dataset is constructed.
    Additionally, the path of each file is stored.

    Args:
        json_directory (str): The path to the json directory that through the create_features_directory function.
    Returns:
        (pd.Dataframe): A dataframe with the features extracted from the whole dataset. (num_of_instances x 165)
    """

    dict_list = []

    if not os.path.isdir(json_directory):
        print('Invalid path! Check your config file.')
        return

    for root, _, files in os.walk(json_directory):
        for file_element in files:
            if not file_element.endswith('.json'):  # ignore other files
                continue

            json_file_path = PurePath(root, file_element)
            with open(json_file_path) as file:
                json_dict = json.load(file)
                json_dict['file_path'] = json_file_path
                dict_list.append(json_dict)

    return pd.DataFrame(dict_list)
