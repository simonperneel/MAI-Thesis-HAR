# Thesis Human Activity Recognition
This thesis investigates the optimisation of activity recognition systems using wearable MIMU sensors, focusing on efficient, orientation-independent features. Magnetometer data showed little added value, while ankle placement provided the best results. The final system achieved high accuracy (0.97) and F1-score (0.98) using 20 features from the accelerometer and gyroscope.

This repository contains the dataset, software and documentation. The thesis was completed as part of the master Artificial Intelligence at KU Leuven. The report can be found [```here```](./Report/report.pdf).

The software code includes tools and libraries to access data files from XSens Awinda IMUs to do preprocessing, feature extraction, classification and more. 

## Folder structure

    .
    ├── Data                    # Folder with all data 
    ├── Plots                   # Folder where the plots are written to
    ├── Report                  # Folder containing the thesis report
    ├── ann_classifier.py       # classification with neural net 
    ├── main.py                 # main file: pre-processing, feature extraction, classifcation, evaluation, ...
    ├── utils.py                # functions for feature extraction
    ├── requirements.txt        # required packages and their versions
    └── README.md

## Installation (Windows)
- Python version 3.8 is recommended and can be installed here: https://www.python.org/downloads/release/python-3810/
- (Mini)Conda for easy installation of the required packages: https://docs.conda.io/en/latest/miniconda.html 


1. Dowload the code and data from Github on https://github.com/simonperneel/MAI-Thesis-HAR or with following command in the terminal: \
```git clone https://github.com/simonperneel/MAI-Thesis-HAR ``` 

2. Required packages and there versions  for running the code are listed in ```requirements.txt```. They can easily be installed with conda in a virtual environment with following command in the terminal: \
```conda create --name <envname> --file requirements.txt``` 

3. Activate the enviromnent for running the python code: \
```conda activate <envname>```

3. Install one more package with pip (can't be installed with conda): \
```pip install tsfresh```

4. Run the code with your IDE or from the terminal: \
```python main.py``` \
```python ann_classifier.py```

## Code
[```Click here```](https://simonperneel.github.io/modules.html) for the documented source code 
### [```main.py```](./main.py)
The script automatically loads the csv files, and puts it in a pandas dataframe. 3 sensors are used, so at each time point, there are measurements from 3 sensors. All recorded data is stored in one data frame and exported in ```processed_data.csv```

The starting point are the CSV files exported by the MTw software. 

**Example**  
sensor 1: tp1-1-running-000_00B42D0F.csv  
sensor 2: tp1-1-running-000_00B42D71.csv \
sensor 3: tp1-1-running-000_00B42D95.csv

This is one trial of an activity by a subject. These files are read in and the measurements from each sensor are put next to each other for each timestamp. The csv file with all the data looks like this:

| Packet number | SampleTime | Acc_X_sensor1 | Acc_Y_sensor1 | ... | Acc_X_sensor2 | Acc_Y_sensor2 | ... | Acc_X_sensor3 | Acc_Y_sensor3 | ... | Activity label |
|---------------|------------|---------------|---------------|-----|---------------|---------------|-----|---------------|---------------|-----|----------------|
| 1             | 0          | -6.14         | 9.62          |     | 1.14          | -0.06         |     | 15.62         | 3.84          |     | running        |
| 2             | 0.01       | -1.44         | 8.45          |     | 5.89          | -1.13         |     | 17.77         | -2.02         |     | running        |
| ...           | ...        | ...           | ...           |     | ...           | ...           |     | ...           | ...           |     | ...            |
| 400           | 2          | 10.23         | 4.99          |     | 2.33          | -5.45         |     | 3.49          | 0.98          |     | running        |

### [```ann_classifier.py```](./ann_classifier.py)
classification on ***raw*** acceleration and gyroscope data with a MLP neural net. 

### [```utils.py```](./utils.py) 
file containing some self-defined functions for feature extraction



