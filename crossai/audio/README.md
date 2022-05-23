#Audio features extraction

The audio features extraction process is based on the premises that the user will provide an audio dataset with the
following format:
```
root_dataset_directory
│
├── label_1
│   │
│   ├── audio_file_1
│   ├── audio_file_2
├── label_2
│   ├── audio_file_3
│   ├── audio_file_4
└── label_3
```

where each of the subdirectories contain only audio files with a certain label (e.g genre).

Nevertheless, a user can provide either a single audio file to the `long_feature_wav` function to extract features 
in a json format and save it to a target location.

```python
# The following stores a json containing the features extracted by an audio file and stores it 
# to the target store path. The function will automatically add the root folder name as the 'label' field in the json.
features_json = features_to_json(root_path="/label_folder_path", filename="filename", 
                                 save_location="/target_store_path", m_win=1,
                                 m_step=1, s_win=0.1, s_step=0.05,accept_small_wavs=True,
                                 compute_beat=True, features_to_compute=["ann", "libr", "surf"])
```

For ease of use two additional functions have been provided: 

* `create_features_directory`: Creates a directory that contains the jsons with the features extracted for each audio 
   file and has the same structure as the input dataset_directory (as presented in the example).
```python
# Function that generates an identical directory structure to the one of the input WAV files
# that contains the feature json files of each audio file. The name of the new directory will be the same as 
# dir_path argument with the suffix "_features" added to it.
create_features_directory(dir_path="/path_where_the_audio_files_are stored", m_win=1, m_step=1,
                          s_win=0.1, s_step=0.05,accept_small_wavs=True,
                          compute_beat=True, features_to_compute=["ann", "libr", "surf"])
```
* `features_dataframe_builder`: Returns a `pd.Dataframe` that contains all the features extracted for all the audio files in the dataset. The `json_directory` argument should be the root of the directory created through the The path to the json directory that through the `create_features_directory` function.

# Full features' list 

For each of the features the `mean` is calculated along with the `std`. That makes the total number of features for each feature equal to 2. For most of the features their respective `deltas` (mean and std) are calculated as well. That brings up the total number of features per feature equal to 4. The only exceptions are `Spectral kurtosis`, `Spectral skewness` and `Spectral Slope`, where instead of their delta their first derivatives (mean and std) are calculated.
Features `Loudness`, `Beat` and `Beat confidence` are contributing with only one feature as no additional features can be computed from them. 
**If all the feature extraction options are enabled the total amount of features that are being computed is 163. Including the label the features are 164.**

The first column indicates the feature extraction options that someone can utilize inside the relative functions in the argument `features_to_compute`.

| Feature extraction option                                 | Extra Features<br/> (delta, derivative) | Total features | Feature Name       | Feature Description                                                                                                                                                                                                            |
|----------------------------------------------------------|-----------------------------------------|----------------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| "aan"                                                    | &check;                                 | 4              | Zero Crossing Rate | The rate of sign-changes of the signal <br/>during the duration of a particular frame.                                                                                                                                         |
|                                                          | &check;                                 | 4              | Energy             | The sum of squares of the signal values,<br/> normalized by the respective frame length.                                                                                                                                       |
|                                                          | &check;                                 | 4              | Entropy of Energy	 | The entropy of sub-frames' normalized energies.<br/> It can be interpreted as a measure of abrupt changes.                                                                                                                     |
|                                                          | &check;                                 | 4              | Spectral Centroid	 | The center of gravity of the spectrum.                                                                                                                                                                                         |
|                                                          | &check;                                 | 4              | Spectral Spread	   | The second central moment of the spectrum.                                                                                                                                                                                     |
|                                                          | &check;                                 | 4              | Spectral Entropy	  | Entropy of the normalized spectral energies for a set of sub-frames.                                                                                                                                                           |
|                                                          | &check;                                 | 4              | Spectral Flux      | The squared difference between the normalized magnitudes of the spectra of the two successive frames.                                                                                                                          |
|                                                          | &check;                                 | 4              | Spectral Rolloff		 | The frequency below which 90% of the magnitude distribution of the spectrum is concentrated.                                                                                                                                   |
|                                                          | &check;                                 | 52             | MFCCs              | Mel Frequency Cepstral Coefficients form a cepstral representation where the frequency bands are not linear but distributed according to the mel-scale.                                                                        |
|                                                          | &check;                                 | 48             | Chroma Vector	     | A 12-element representation of the spectral energy where the bins represent the 12 equal-tempered pitch classes of western-type music (semitone spacing).                                                                      |
|                                                          | &check;                                 | 4              | Chroma Deviation	  | The standard deviation of the 12 chroma coefficients.                                                                                                                                                                          |
| "libr"                                                   | &check;                                 | 4              | Spectral Bandwidth | P’th-order spectral bandwidth                                                                                                                                                                                                  |
|                                                          | &check;                                 | 4              | Spectral Flatness  | Spectral flatness (or tonality coefficient) is a measure to quantify how much noise-like a sound is, as opposed to being tone-like. A high spectral flatness (closer to 1.0) indicates the spectrum is similar to white noise. |
|                                                          | &check;                                 | 4              | Spectral rms       | Root-mean-square (RMS)                                                                                                                                                                                                         |
| "surf"                                                   | - , &check;                             | 4              | Spectral kurtosis  | Spectral kurtosis of the waveform.                                                                                                                                                                                             |
|                                                          | - , &check;                             | 4              | Spectral skewness  | Spectral skewness of the waveform.                                                                                                                                                                                             |
|                                                          | - , &check;                             | 4              | Spectral slope     | The spectral slope of the waveform.                                                                                                                                                                                            |
|                                                          |                                         | 1              | Loudness           | The loudness of the waveform.                                                                                                                                                                                                  |
| Only if argument `compute_beat` is `True`                |                                         | 1              | Beat               | Estimate of the beats per minute.                                                                                                                                                                                              |
| and "aan" is in `features_to_compute` <br/>list argument |                                         | 1              | Beat confidence    | A confidence measure of the bpm estimated.                                                                                                                                                                                     |

## All currently exported features

| Feature name | 
| ------------ |
| zcr_mean<br> energy_mean<br> energy_entropy_mean<br> spectral_centroid_mean<br> spectral_spread_mean<br> spectral_entropy_mean<br> spectral_flux_mean<br> spectral_rolloff_mean<br> mfcc_1_mean<br> mfcc_2_mean<br> mfcc_3_mean<br> mfcc_4_mean<br> mfcc_5_mean<br> mfcc_6_mean<br> mfcc_7_mean<br> mfcc_8_mean<br> mfcc_9_mean<br> mfcc_10_mean<br> mfcc_11_mean<br> mfcc_12_mean<br> mfcc_13_mean<br> chroma_1_mean<br> chroma_2_mean<br> chroma_3_mean<br> chroma_4_mean<br> chroma_5_mean<br> chroma_6_mean<br> chroma_7_mean<br> chroma_8_mean<br> chroma_9_mean<br> chroma_10_mean<br> chroma_11_mean<br> chroma_12_mean<br> chroma_std_mean<br> delta zcr_mean<br> delta energy_mean:,delta energy_entropy_mean<br> delta spectral_centroid_mean<br> delta spectral_spread_mean<br> delta spectral_entropy_mean<br> delta spectral_flux_mean<br> delta spectral_rolloff_mean<br> delta mfcc_1_mean<br> delta mfcc_2_mean<br> delta mfcc_3_mean<br> delta mfcc_4_mean<br> delta mfcc_5_mean<br> delta mfcc_6_mean<br> delta mfcc_7_mean<br> delta mfcc_8_mean<br> delta mfcc_9_mean<br> delta mfcc_10_mean<br> delta mfcc_11_mean<br> delta mfcc_12_mean<br> delta mfcc_13_mean<br> delta chroma_1_mean<br> delta chroma_2_mean<br> delta chroma_3_mean<br> delta chroma_4_mean<br> delta chroma_5_mean<br> delta chroma_6_mean<br> delta chroma_7_mean<br> delta chroma_8_mean<br> delta chroma_9_mean<br> delta chroma_10_mean<br> delta chroma_11_mean<br> delta chroma_12_mean<br> delta chroma_std_mean<br> zcr_std<br> energy_std<br> energy_entropy_std<br> spectral_centroid_std<br> spectral_spread_std<br> spectral_entropy_std<br> spectral_flux_std<br> spectral_rolloff_std<br> mfcc_1_std<br> mfcc_2_std<br> mfcc_3_std<br> mfcc_4_std<br> mfcc_5_std<br> mfcc_6_std<br> mfcc_7_std<br> mfcc_8_std<br> mfcc_9_std<br> mfcc_10_std<br> mfcc_11_std<br> mfcc_12_std<br> mfcc_13_std<br> chroma_1_std<br> chroma_2_std<br> chroma_3_std<br> chroma_4_std<br> chroma_5_std<br> chroma_6_std<br> chroma_7_std<br> chroma_8_std<br> chroma_9_std<br> chroma_10_std<br> chroma_11_std<br> chroma_12_std<br> chroma_std_std<br> delta zcr_std<br> delta energy_std<br> delta energy_entropy_std<br> delta spectral_centroid_std<br> delta spectral_spread_std<br> delta spectral_entropy_std<br> delta spectral_flux_std<br> delta spectral_rolloff_std<br> delta mfcc_1_std<br> delta mfcc_2_std<br> delta mfcc_3_std<br> delta mfcc_4_std<br> delta mfcc_5_std<br> delta mfcc_6_std<br> delta mfcc_7_std<br> delta mfcc_8_std<br> delta mfcc_9_std<br> delta mfcc_10_std<br> delta mfcc_11_std<br> delta mfcc_12_std<br> delta mfcc_13_std<br> delta chroma_1_std<br> delta chroma_2_std<br> delta chroma_3_std<br> delta chroma_4_std<br> delta chroma_5_std<br> delta chroma_6_std<br> delta chroma_7_std<br> delta chroma_8_std<br> delta chroma_9_std<br> delta chroma_10_std<br> delta chroma_11_std<br> delta chroma_12_std<br> delta chroma_std_std<br> beat<br> beat_conf<br> spectral_bandwidth_mean<br> spectral_flatness_mean<br> spectral_rms_mean<br> spectral_bandwidth_std<br> spectral_flatness_std<br> spectral_rms_std<br> spectral_bandwidth_delta_mean<br> spectral_bandwidth_delta_std<br> spectral_flatness_delta_mean<br> spectral_flatness_delta_std<br> spectral_rms_delta_mean<br> spectral_rms_delta_std<br> spectral_kurtosis_mean<br> spectral_kurtosis_std<br> spectral_kurtosis_first_derivative_mean<br> spectral_kurtosis_first_derivative_std<br> spectral_skewness_mean<br> spectral_skewness_std<br> spectral_skewness_first_derivative_mean<br> spectral_skewness_first_derivative_std<br> spectral_slope_mean<br> spectral_slope_std<br> spectral_slope_first_derivative_mean<br> spectral_slope_first_derivative_std<br>              |
