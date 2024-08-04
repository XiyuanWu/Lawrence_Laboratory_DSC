# Lawrence Livermore National Laboratory Data Science Challenge 2024 (LLNL DSC 2024)

![Static Badge](https://img.shields.io/badge/NumPy-path?style=for-the-badge&logo=Numpy&color=%green)
![Static Badge](https://img.shields.io/badge/Pandas-path?style=for-the-badge&logo=Pandas&color=%23150458)
![Static Badge](https://img.shields.io/badge/Scikit%20Learn-path?style=for-the-badge&logo=scikit-learn&color=orange)
![Static Badge](https://img.shields.io/badge/Pytorch-path?style=for-the-badge&logo=pytorch&color=purple)


## Overview

Arrhythmia is a medical condition characterized by irregular heartbeats, which, if left untreated, can have life-threatening effects. This project leverages machine learning to detect heart abnormalities using inexpensive electrocardiogram (ECG) data. Our model can classify ECG data as usual or one of four abnormal heartbeats with a false negative rate of only 0.87%. Additionally, our model’s predictions for myocardial activation times are, on average, only 2 ms off. These results suggest that machine learning is a promising, cost-effective solution for arrhythmia detection.

## Background

Heart disease is the leading cause of death in the United States. Arrhythmias are a result of underlying heart problems and classifying them can provide life-saving care. A common method of getting heartbeat data is a standard 12 lead ECG, where electric signals from the heart are measured. We aim to use these signals to predict when specific parts of the heart activate.  

## Poster

[DSC Poster](DSC_poster_#7.pdf)


## Installation/Usage

1. Clone this repo recursively
```
git clone --recurse-submodules <link for this repo>
```
2. Set up Dataset

    For Task1&2, Dataset can be found on [here](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

    For Task 3&4, Download the dataset by running the following code
    ```
    source download_intracardiac_dataset.sh
    ```
    
3. After set up, run each notebook in order. 
4. To run the deep learning model(task 3 and beyond), 32GB of RAM or a GPU is required in order to run.
