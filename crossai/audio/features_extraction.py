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


def long_feature_wav(wav_file, mid_window, mid_step,
                     short_window, short_step,
                     accept_small_wavs=False,
                     compute_beat=True,
                     librosa_features=False,
                     surfboard_features=False):
    """
    This function computes the long-term feature per WAV file.
    It is identical to directory_feature_extraction, with simple
    modifications in order to be applied to singular files.
    Very useful to create a collection of json files (1 song -> 1 json).
    Genre as a feature should be added (very simple).
    ARGUMENTS:
        - wav_file:        the path of the WAVE directory
        - mid_window, mid_step:    mid-term window and step (in seconds)
        - short_window, short_step:    short-term window and step (in seconds)
    RETURNS:
        - mid_term_feaures: The feature vector of a singular wav file
        - mid_feature_names: The feature names, useful for formating
    """

    mid_term_features = np.array([])

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

        if librosa_features:
            librosa_feat, librosa_feat_names = _audio_to_librosa_features(
                wav_file, sampling_rate=sampling_rate)
            mid_features = np.append(mid_features, librosa_feat)
            for element in librosa_feat_names:
                mid_feature_names.append(element)

        if surfboard_features:
            surfboard_feat, surfboard_feat_names = _audio_to_surfboard_features(
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


def features_to_json(root_path, file_name, save_location, yaml_object):
    """
    Function that saves the features returned from long_feature_wav
    to json files. This functions operates on a singular wav file.
    Appends the genre to the json file also.
    ARGUMENTS:
     - root_path: absolute path of the dataset, useful for audio loading
     - file_name: self explanatory
     - save_location: self explanatory
     - yaml_object: obj of the yaml object, contains parameters for the feature extraction
    """
    m_win, m_step, s_win, s_step, compute_beat, accept_small_wavs = \
    yaml_object['parameters'].values()

    long_feature_return = long_feature_wav(root_path + '/' + file_name, m_win,
                                           m_step,
                                           s_win, s_step, accept_small_wavs,
                                           compute_beat,
                                           librosa_features=yaml_object[
                                               'librosa_features'],
                                           surfboard_features=yaml_object[
                                               'surfboard_features'])

    if long_feature_return == -1:
        return -1

    feature_values, feature_names = long_feature_return
    json_data = dict(zip(feature_names, feature_values))

    # Adding the genre tag to the json dictionary, using pathlib for simplicity
    p = PurePath(root_path)
    genre = p.name
    json_data['genre'] = genre

    json_file_name = save_location + '/' + file_name + '.json'
    with open(json_file_name, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    del json_data
    return json_file_name


def _audio_to_librosa_features(filename, sampling_rate=22050):
    """
    Function that extracts the additional Librosa features
    ARGUMENTS:
     - filename: name of the wav file
     - sampling_rate: used because pyAudioAnalysis uses different sampling rate
     for each wav file

     RETURNS:
     - features: the calculated features, returned as numpy array for consistency (1 x 12)
     - feature_names: the feature names for consistency and pandas formating (1 x 12)
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


def _audio_to_surfboard_features(filename, sampling_rate=44100):
    """
    Function that extracts the additional Surfboard features
    ARGUMENTS:
     - filename: name of the wav file
     - sampling_rate: used because pyAudioAnalysis uses different sampling rate
     for each wav file

     RETURNS:
     - feature_values: the calculated features, returned as numpy array for consistency (1 x 13)
     - feature_names: the feature names for consistency and pandas formating (1 x 13)
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