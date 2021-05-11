# Thesis Human Activity Recognition - Simon Perneel

A set of tools and libraries written in Python code to access Xsens MTw data files and do processing, feature extraction, classification and more. Required packages and there versions are listed in ```requirements.txt```. They can easily be installed with conda in a virtual environment with following command:

```conda create --name <envname> --file requirements.txt``` \
Activate the enviromnent for running the python code:

```conda activate <envname>```

This project contains the code to access and load XsensData. 
The starting point are the CSV files exported by the MTw software. the naming of these files follow a fixed template ```tp{x}-{y}-{activity}-{sensor}.csv```. With x the ID of the subject, y the trial number, activity the name of the activity (running, walking, upstairs, downstairs, jumping, standing, sitting) and sensor the sensor ID of the XSens IMU

The script automatically loads the csv files, and puts it in a pandas dataframe. 3 sensors are used, so at each time point, there are measurements from 3 sensors. All recorded data is stored in one data frame and exported in ```processed_data.csv```

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




