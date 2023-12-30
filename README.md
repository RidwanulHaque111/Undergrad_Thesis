# Unveiling Crop Diversity through Bat Algorithm-Driven Framework for Multispectral Satellite Image Analysis

This repository contains my research work for classifying crop diversity from multispectral satellite images based on different neural network models and Bat-Algorithm optimization.

Crops are primary source of food, contributing to global nutrition and supporting various industries.
As different Crop Variety provides different nutrition, Understanding and classifying crops are essential for optimizing resources.
Neural network models can be used to classify crops from satellite images. 

## Why Use Satellite Images for Crop Classification?

* Large-Scale Crop Coverage: Satellite images provide comprehensive coverage of agricultural areas, allowing for the analysis of large expanses of farmland.
* Temporal Monitoring: Satellites capture images over time, enabling the monitoring of crop growth stages, changes, and seasonal variations.
* Remote Sensing: Satellite-based remote sensing offers a non-invasive way to collect data, reducing the need for physical field surveys.
* Timely Information: Satellite imagery provides real-time or near-real-time information, crucial for making timely decisions in agriculture.

## Research Objectives

* Exploring the capacity of satellite images
* Identifying problematic areas within agricultural regions
* Comparing the performance of the Bat Algorithm-based approach with traditional classification methods


## Research Significance

* Precision Agriculture
* Scientific Advancement
* Food Security
* Resource Conservation

## Dataset Collection
I have selected a small ROI from the state "Alabama", United States of America. 
* Labelled crop images were downloaded from USDA National Agricultural Statistics Service CropScape database 2022. [](https://nassgeodata.gmu.edu/CropScape/)
* Sample satellite images were downloaded from United States Geological Survey (USGS) Earth Explorer platform. [](https://earthexplorer.usgs.gov/)

![Labelled Image](https://github.com/RidwanulHaque111/Undergrad_Thesis/blob/d84d72c5d1d2ad9944239546fe350c0817510d9a/additional%20images/Labelled%20Image%20Sample.png)

The labelled image and satellite image bands have the dimension of
(8200x8200) pixels. (8200 x 8200) = 67,240,000 pixels or 67.24 million labelled crop dataset. 
(67.24 x 5) = 336.2 million sample data are used in the research for the five satellite image bands.

![Satellite Image Bands](https://github.com/RidwanulHaque111/Undergrad_Thesis/blob/d84d72c5d1d2ad9944239546fe350c0817510d9a/additional%20images/Satellite%20Image%20Bands.png)


## Data Preprocessing
* Geometric Correction: Aligning the satellite imagery to a known map projection.
* Synchronization: Matching the labelled crop data from CropScape with the corresponding satellite images.
* Calibration: Converting the raw digital values of the satellite imagery to physical units of radiance.
* Atmospheric Correction: Removing the effects of the atmosphere from the satellite imagery.

## Data Labels Reading

I have used the GDAL package to read the labelled crop data-set. GDAL stands for “Geospatial Data Abstraction Library” which is an open-source library and designed for translating and processing raster and vector geospatial data formats.

GDAL supports GeoTIFF. GDAL can read and work with various geospatial data formats:
- Open Image Dataset
- Metadata Access
- Read Specific Labels
- Close Image Dataset


## Model Building

1.	Reading the USDA crop labels using GDAL package
2.	Checking the percentage of each crop.
3.	The dataset I have been using currently has 48 unique classes (crop variety)
4.	I have selected the top 5 crops for now and built the model.
5.	Reading the images and stack them in the columns in a PANDAS data frame (python)
6.	Export these data into CSV file
7.	We have our required Data to train and test.
8.	Split the train and test sets, label encode the crop variety and normalize it before feeding the data into the model.
9.	Training the model using -
    - Simple Neural Network (NN),
    - Convolutional Neural Network (CNN),
    - Feedforward Neural Network (FNN),
    - Recurrent Neural Network (RNN)
    - Bat Algorithm Optimization (BA).



## Train, Test and Validation Set Splitting
* 90%, 5%, 5% split for Train, Test and Validation Set were performed.


## Methods

* We have used the Bat Algorithm Optimization as the primary method.
* Additionally, we have used these methods :
    - Simple Neural Network (SNN),
    - Convolutional Neural Network (CNN),
    - Feedforward Neural Network (FNN) 
    - Recurrent Neural Network (RNN)

## Why Use Bat Algorithm?

* Effectiveness is primarily determined by its ability to accurately classify crops in satellite images
* Selection capability, robustness to variations in data
* Computational efficiency
* Sensitivity to hyperparameters

## Performance Metrics
Performance Metrics Used :
- Accuracy
- Precision
- Recall
- F1 Score
- Macro F1 Score


## Experimental Setup
* The pre-processing phase of the study was conducted on a personal computer running the Windows 10 operating system.
* The computer is equipped with 16 gigabytes of RAM and features an AMD Ryzen 5 5600x processor with a clock speed of 3.70 GHz, boasting 6 cores and 12 threads.
* Python was selected as the primary programming language for this task.
* To read the crop data labels, the GDAL and GC libraries were employed.
* For the implementation of various machine learning and deep learning methods, we utilized the scikit-learn, TensorFlow, and Keras libraries.
* The code execution took place on Google Colab, a Python development environment that operates within a web browser and leverages the Google Cloud infrastructure.

## Results
Based on the train score, it is evident that the Bat Algorithm Optimization and the Convolutional Neural Network models outperform the rest in terms of training performance metrics.
The Bat Algorithm Optimization achieves an impressive accuracy of 91.04%, with high precision, recall, F1 score, and macro F1, indicating its excellence in classification tasks. The Convolutional Neural Network model closely follows, with an accuracy of 90.22% and consistently high scores across all other metrics. The Simple Neural Network and Feedforward Neural Network models also perform well. However, the Recurrent Neural Network lags behind in all metrics, indicating that it may not be as suitable for this particular task. The Bat Algorithm Optimization model clearly stands out as the top performer in this comparison.

Also, based on the test score, a striking distinction comes to the forefront, echoing the patterns seen in the training data. Once again, the Bat Algorithm Optimization and the Convolutional Neural Network models shine as the front-runners in terms of test performance metrics. The Convolutional Neural Network maintains a solid accuracy of 90.13%, while the Bat Algorithm Optimization model takes the lead with an impressive 91.02% accuracy. These two models consistently exhibit high precision, recall, F1 scores, and macro F1 values, underscoring their dependability and effectiveness in classifying the test data. The Simple Neural Network and Feedforward Neural Network also deliver commendable results, boasting accuracy scores in the mid-80s and well-balanced precision, recall, and F1 metrics. Conversely, the Recurrent Neural Network consistently lags behind in all metrics, signaling its limited suitability for this specific task.

## Limitations and Challenges
* Selected satellite images have approximately 2%-5% cloud cover, which may still affect data quality.
* Availability of cloud-free imagery for certain regions and timeframes can be a limitation in practical applications.
* Fine-tuning hyperparameters for different models could potentially yield even better results.
* Temporal factors, such as seasonal variations and crop growth stages can be crucial for accurate classification.

## Conclusion
* Bat Algorithm Optimization Dominance
* Traditional Neural Networks Competence
* CNN Spatial Expertise
* RNN Sequential Lag
* Generalizability of the Bat Algorithm


## Future Research
* Implications for Precision Agriculture
* Cost-Effective Solutions
* Hybrid Model Investigation
* Optimization Exploration
* Diversification into Other Agricultural Applications











