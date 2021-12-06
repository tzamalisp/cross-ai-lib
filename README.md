CrossAI Library

This repository contains a library of high-level functionalities capable of building 
Artificial Intelligence processing pipelines for Time-Series and Natural Language Processing.

Contributors:
* [Pantelis Tzamalis](https://www.linkedin.com/in/pantelis-tzamalis/)
* [Andreas Bardoutsos](https://www.linkedin.com/in/andreasbardoutsos/)
* [Dimitris Markantonatos](https://www.linkedin.com/in/dimitris-markantonatos-4a7b25196/)

## Time Series for Motion Mining

### Processing Motion

Below are the processing functionalities that the library contains:

* IMU-motion signals features extraction
* IMU-motion signals data augmentation 
* Processing and labeling time-series, with focus on motion
* Visualization of IMU-motion signals
* Visualization of time-series classification
* Dataset exploratory analysis and Visualization. 
* Dataset post-processing on different data fields (image, signals, audio, text):
  * Train/Test Split
  * Scaling: Normalization, Standardization, Gaussianization
  * Dimensionality Reduction: PCA

### AI Models

Here are presented the Neural Network models that are implemented for the 1D dimension, 
i.e. for the time-series analysis, including the corresponding citation for each one:

XceptionTime
```
Rahimian, Elahe, Soheil Zabihi, Seyed Farokh Atashzar, Amir Asif, and Arash Mohammadi. 
"Xceptiontime: A novel deep architecture based on depthwise separable convolutions for hand gesture classification." 
arXiv preprint arXiv:1911.03803 (2019).
```

InceptionTime
```
Fawaz, Hassan Ismail, Benjamin Lucas, Germain Forestier, Charlotte Pelletier, Daniel F. Schmidt, Jonathan Weber, 
Geoffrey I. Webb, Lhassane Idoumghar, Pierre-Alain Muller, and François Petitjean. 
"Inceptiontime: Finding alexnet for time series classification." 
Data Mining and Knowledge Discovery 34, no. 6 (2020): 1936-1962.
```

BiLSTM-Time
```
Hou, Jiahui, Xiang-Yang Li, Peide Zhu, Zefan Wang, Yu Wang, Jianwei Qian, and Panlong Yang. 
"Signspeaker: A real-time, high-precision smartwatch-based sign language translator." 
In The 25th Annual International Conference on Mobile Computing and Networking, pp. 1-15. 2019.
```

## Text Mining

Regarding text mining, a series of methods, algorithms and visualization techniques have been
implemented.

### Processing Text

Processing text:

* 

Features Extraction:

*

## Citing Cross-AI

If you use CrossAI library in your research, please use the following BibTeX entry:

```
@misc{
    CrossAI_Library, 
    author = {Tzamalis, Pantelis and Bardoutsos, Andreas and Markantonatos, Dimitris}, 
    url = {https://github.com/tzamalisp/cross-ai-lib}, 
    month = {8}, 
    title = {{Cross-AI Library}}, 
    year = {2021}
}
```