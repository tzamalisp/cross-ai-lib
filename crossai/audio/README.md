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
in a json format.

For ease of use two additional functions have been provided: 

* `create_features_directory`: Creates a directory that contains the jsons with the features extracted for each audio 
   file and has the same structure as the input dataset_directory (as presented in the example).
* `features_dataframe_builder`: Returns a `pd.Dataframe` that contains all the features extracted for all the audio files in the dataset. The `json_directory` argument should be the root of the directory created through the The path to the json directory that through the `create_features_directory` function. 