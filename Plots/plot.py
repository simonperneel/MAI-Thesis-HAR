"""
Python scripts used for making the plots for the thesis report
author: Simon Perneel - simon.perneel@hotmail.com
"""

import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sns.set()

# General seettings
SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 20
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# figure sizes
WIDTH = 15  # inches
golden_mean = (math.sqrt(5) - 1.0) / 2.0  # aesthetic ratio
HEIGHT = 0.8 * WIDTH * golden_mean  # inches

# path to read the data from
path = r"C:\Users\simon\Documents\School\FIRW MAI\Thesis\Software\Data\Exported"
# sensor IDs
ID1 = '00B42D71'  # Wrist
ID2 = '00B42D0F'  # Waist
ID3 = '00B42D95'  # Ankle
IDs = [ID1, ID2, ID3]

def add_values(dataframe, subject_id, activity, timeseries_id, freq=100):
    """
    :param dataframe: dataframe containing data from one activity trial
    :param subject_id: id of the subject that performed the trial, to be added
    :param activity: activity label of the trial to be added
    :param timeseries_id: to be added, to identify a trial
    :param freq: sample frequency of the measurements
    :return: nothing, just adds information to the passed dataframe
    """
    # add new columns to the existing Dataframe
    # for each body part
    for bodypart in ['ankle', 'wrist', 'waist']:
        # norm of acceleration (gravity included)
        dataframe[f"Acc_norm_{bodypart}"] = [
           np.linalg.norm([x, y, z]) for x,y,z in zip(dataframe[f'Acc_X_{bodypart}'], dataframe[f'Acc_Y_{bodypart}'], dataframe[f'Acc_Z_{bodypart}'])]
        # norm of the free acceleration (no gravity)
        dataframe[f"FreeAcc_norm_{bodypart}"] = [
            np.linalg.norm([x, y, z]) for x,y,z in zip(dataframe[f'FreeAcc_X_{bodypart}'], dataframe[f'FreeAcc_Y_{bodypart}'], dataframe[f'FreeAcc_Z_{bodypart}'])]
        # magnitude of magnetometer values
        dataframe[f"Mag_X_{bodypart}"] = [abs(x) for x in dataframe[f"Mag_X_{bodypart}"]]
        dataframe[f"Mag_Y_{bodypart}"] = [abs(x) for x in dataframe[f"Mag_Y_{bodypart}"]]
        dataframe[f"Mag_Z_{bodypart}"] = [abs(x) for x in dataframe[f"Mag_Z_{bodypart}"]]
        # norm of magnetometer
        dataframe[f"Mag_norm_{bodypart}"] = [
            np.linalg.norm([x, y, z]) for x,y,z in zip(dataframe[f'Mag_X_{bodypart}'], dataframe[f'Mag_Y_{bodypart}'], dataframe[f'Mag_Z_{bodypart}'])]
        # norm of the gyroscope
        dataframe[f"Gyr_norm_{bodypart}"] = [
            np.linalg.norm([x, y, z]) for x,y,z in zip(dataframe[f'Gyr_X_{bodypart}'], dataframe[f'Gyr_Y_{bodypart}'], dataframe[f'Gyr_Z_{bodypart}'])]

    # id of the subject
    dataframe["Subject-id"] = subject_id
    # id for each timeseries (= trial of an activity)
    dataframe['Timeseries-id'] = timeseries_id
    # activity type
    dataframe["Activity"] = activity
    # Transform the labels from String to Integer via LabelEncoder
    #dataframe["ActivityEncoded"] = le.fit_transform(dataframe['Activity'])
    # Calculate sample time based on packet counter
    first_packet = dataframe.iloc[0, 0]  # number of first packet
    dataframe["SampleTimeFine"] = [(x - first_packet)/freq for x in dataframe["PacketCounter"]]

def readOne(path, IDs, tp_id, trial, activity, freq):
    """
    :param path: location on the system to read t
    :param tp_id: id of the subject
    :param trial: trial number of the activity (1-5)
    :param activity: activity performed by the subject ('running','standing', 'sitting",...)
    :param IDs: list containing the hardware ids of the sensors
    :param freq: sample frequency of the measurements
    :return: dataframe containing the timeseries data from one trial of an activity by a subject
    """
    csvs = []
    base = '-000_'
    for i in range(len(IDs)):  # merge 3 sensors measurements in one file
        filename = 'tp' + str(tp_id) + '-' + str(trial) + '-' + str(activity) + str(base) + str(IDs[i])
        file = open(path + "\\" + filename + ".csv")
        single_df = pd.read_csv(file, sep=',', header=4)
        file.close()
        csvs.append(single_df)

    csv_tmp = pd.merge(csvs[0], csvs[1], on=('PacketCounter', 'SampleTimeFine'), suffixes=('_waist', None))
    dataframe = pd.merge(csvs[2], csv_tmp, on=('PacketCounter', 'SampleTimeFine'), suffixes=('_ankle', '_wrist'))
    # add more features
    timeseries_id = 1
    add_values(dataframe, tp_id, activity, timeseries_id, freq)

    return dataframe

def single_plot(dataframe):
    """
    :param dataframe: contains time-series data from one activity trial
    :return: nothing, just plots some features of the timeseries data of one dataframe
    """
    figure, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    plt.autoscale(enable=True, axis='both', tight=None)
    # plot sensor measurements (stored in a dataframe)
    # norm of acceleration
    dataframe.plot(x='SampleTimeFine', y=['FreeAcc_norm_ankle'], label=['Wrist'], lineWIDTH=1.5, ax=ax)

    # plot layout
    activity = dataframe['Activity'].iloc[0]
    subject = dataframe['Subject-id'].iloc[0]
    figure.suptitle('activity: %s, subject: %s' % (activity, subject)), plt.legend()
    plt.subplots_adjust(hspace=0.4)
    ax.set_title('ankle')
    ax.set(xlabel="Time [s]", ylabel="Acceleration [m/s²]")
    #plt.show()
    plt.savefig('Plots/singleplot.svg', format='svg', bbox_inches = 'tight')

def compare_plot():
    """
    Plot the accelerometry and gyroscope data from 6 activity trials next side-by-side
    for comparison
    """
    df1 = readOne(path, IDs, tp_id=2, trial=2, activity='running', freq=100)  # read single activity from 1 csv file
    df3 = readOne(path, IDs, tp_id=1, trial=3, activity='standing', freq=100)  # read single activity from 1 csv file
    df6 = readOne(path, IDs, tp_id=1, trial=3, activity='sitting', freq=100)  # read single activity from 1 csv file
    df4 = readOne(path, IDs, tp_id=2, trial=2, activity='walking', freq=100)  # read single activity from 1 csv file
    df2 = readOne(path, IDs, tp_id=2, trial=2, activity='upstairs', freq=100)  # read single activity from 1 csv file
    df5 = readOne(path, IDs, tp_id=2, trial=2, activity='downstairs', freq=100)  # read single activity from 1 csv file

    figure, axes = plt.subplots(2, 3, figsize=(WIDTH, HEIGHT))
    plt.autoscale(enable=True, axis='both', tight=None)
    sns.set_palette(palette='tab10', n_colors=3)
    df1 = df1.iloc[0:300]
    df2 = df2.iloc[0:200]
    df3 = df3.iloc[0:300]
    df4 = df4.iloc[0:300]
    df5 = df5.iloc[0:200]
    df6 = df6.iloc[0:200]
    # norm of acceleration
    df1.plot(x='SampleTimeFine', y=['FreeAcc_norm_wrist','FreeAcc_norm_waist', 'FreeAcc_norm_ankle'], label=['wrist', 'thigh', 'ankle'], linewidth=1, ax=axes[0][0], color=sns.color_palette())
    df2.plot(x='SampleTimeFine', y=['FreeAcc_norm_wrist','FreeAcc_norm_waist', 'FreeAcc_norm_ankle'], label=['wrist', 'thigh', 'ankle'], linewidth=1, ax=axes[0][1], color=sns.color_palette())
    df3.plot(x='SampleTimeFine', y=['FreeAcc_norm_wrist','FreeAcc_norm_waist', 'FreeAcc_norm_ankle'], label=['wrist', 'thigh', 'ankle'], linewidth=1, ax=axes[0][2], color=sns.color_palette())
    df4.plot(x='SampleTimeFine', y=['FreeAcc_norm_wrist','FreeAcc_norm_waist', 'FreeAcc_norm_ankle'], label=['wrist', 'thigh', 'ankle'], linewidth=1, ax=axes[1][0], color=sns.color_palette())
    df5.plot(x='SampleTimeFine', y=['FreeAcc_norm_wrist','FreeAcc_norm_waist', 'FreeAcc_norm_ankle'], label=['wrist', 'thigh', 'ankle'], linewidth=1, ax=axes[1][1], color=sns.color_palette())
    df6.plot(x='SampleTimeFine', y=['FreeAcc_norm_wrist','FreeAcc_norm_waist', 'FreeAcc_norm_ankle'], label=['wrist', 'thigh', 'ankle'], linewidth=1, ax=axes[1][2], color=sns.color_palette())

    # plot layout
    """activity = dataframe['Activity'].iloc[0]
    subject = dataframe['Subject-id'].iloc[0]
    figure.suptitle('activity: %s, subject: %s' % (activity, subject)), plt.legend()"""
    axes[0][0].set_title('running')
    axes[0][1].set_title('going upstairs')
    axes[0][2].set_title('stance-to-sit')
    axes[1][0].set_title('walking')
    axes[1][1].set_title('going downstairs')
    axes[1][2].set_title('sit-to-stance')
    plt.subplots_adjust(hspace=0.4)
    for ax in axes.flat:
        ax.set(xlabel="Time [s]", ylabel="Norm of acceleration [m/s²]")
    #plt.show()
    plt.savefig('Plots/SomeActivities.pdf', format='pdf', bbox_inches = 'tight')

def dataDistribution():
    """
    data distribution along the subjects
    """
    df = pd.DataFrame(columns=["Activity","1", "2","3",
                               "4","5",
                               "6","7","8",'9'],
                      data=[["Walking",63,72,76,67,78,66,77,83,75],
                            ["Upstairs",45,46,48,4,7,49,35,41,36,45],
                            ["Downstairs",32,38,43,41,46,36,35,42,41],
                            ["Running",42,32,45,29,45,32,38,33,36],
                            ["Sitting",4,5,5,5,5,5,5,5,5],
                            ["Standing",5,5,5,5,5,5,5,5,5],
                            ["Jumping",5,5,9,5,5,5,5,5,5]])

    ax = df.set_index('Activity').reindex(df.set_index('Activity').sum().sort_values(ascending=False).index, axis=1)\
      .T.plot(kind='bar', stacked=True,
              colormap=ListedColormap(sns.color_palette("coolwarm", 7)),
              figsize=(18,10))
    plt.xticks(rotation=0)
    plt.xlabel('Subject nr.')
    plt.ylabel('count')

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(x+width/2,
                y+height/2,
                '{:.0f}'.format(height),
                horizontalalignment='center')

    plt.savefig('Plots/datadistribution.pdf', format='pdf')

def compareF1Scores():
    sns.color_palette(n_colors=3)
    d_acc = {'length': [0.5,0.75,1,1.5,2,3,4,5,6,7,8,9,10], '0': [0.87,0.8965,0.943,0.96,0.957,0.96,0.97,0.97,0.975,0.975, 0.98,0.98,0.975],
        '50': [0.88,0.8975,0.94,0.96,0.96,0.97,0.975,0.97,0.975,0.97,0.98,0.975,0.98], '25': [0.8725,0.8965,0.94,0.9575,0.965,0.97,0.97,0.975,0.975,0.98,0.98,0.98,0.985]}
    df_acc = pd.DataFrame(data=d_acc)

    d_f1 = {'length': [0.5,0.75,1,1.5,2,3,4,5,6,7,8,9,10], '0': [0.84,0.8775,0.915,0.95,0.95,0.967,0.97,0.97,0.9725,0.9725,0.975,0.9725,0.97],
    '50': [0.85,0.885,0.92,0.9425,0.95,0.9575,0.965,0.97,0.9725,0.975,0.98,0.975,0.975], '25': [0.84,0.875,0.9125, 0.9475,0.96,0.9675,0.9675,0.9725,0.97,0.975,0.975,0.98,0.98]}
    df_f1 = pd.DataFrame(data=d_f1)

    df_acc.plot(x='length', y=['0','50', '25'], label=['0% overlap', '50% overlap', '25% overlap'], linewidth=2, markersize=10, style='-o')
    plt.xticks(ticks=[1,2,3,4,5,6,7,8,9,10])
    plt.xlabel('window length [s]')
    plt.ylabel('accuracy [%]')
    plt.show()

def compareF1Scores2():
    sns.set_palette('PuBu', n_colors=2)
    df = pd.DataFrame({
    'Classifier': ['RF','SVM', 'kNN', 'DT', 'MLP','NB'],
    'L1O': [0.968, 0.97, 0.96, 0.89, 0.97, 0.94],
    'k-fold': [0.982, 0.98, 0.978, 0.934, 0.975, 0.95]
    })

    std = [0.069353273,	0.051639778,	0.063835727,	0.086467006,	0.064940619,	0.081528606,	0.041761226,
           0.022184679,	0.034871192,	0.150053324,	0.028635642,	0.050950957]
    fig, ax1 = plt.subplots(figsize=(WIDTH, HEIGHT))
    tidy = df.melt(id_vars='Classifier', var_name='Evaluation method', value_name='Value')
    print(tidy.head(12))
    g = sns.barplot(x='Classifier', y='Value', hue='Evaluation method', data=tidy, ax=ax1)
    plt.xlabel('Classifier')
    plt.ylabel('f1-score [%]')
    for i, p in enumerate(g.patches):
        g.annotate(format(p.get_height(), '.1%'),
                       (p.get_x() + p.get_width() / 2., 0),
                       ha = 'center', va = 'bottom', rotation=90,
                       xytext = (0, 9),
                       textcoords = 'offset points')

        plt.errorbar(x=(p.get_x()+(p.get_width()/2)), y=p.get_height(), yerr=std[i], ecolor='black', capsize=4)
    sns.despine(fig)
    plt.show()

def groupedBarPlot():
    df = pd.read_csv('../Evaluation/precisioncompared.csv', delimiter=',')
    print(df.head())

    sns.set_palette('Greens', n_colors=3)
    g = sns.catplot(
        data=df, kind="bar",
        x="Activity", y="precision", hue="sensor", alpha=.6, height=HEIGHT, aspect=1/golden_mean, legend=False)
    g.set_axis_labels("", "Precision [%]")
    plt.legend(loc='upper left')

    for ax in g.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.1%'),
                       (p.get_x() + p.get_width() / 2., 0),
                       ha = 'center', va = 'bottom', rotation=90,
                       xytext = (0, 9),
                       textcoords = 'offset points')
    plt.show()

def main():
    df = pd.read_pickle("../Data/data.pkl")  # read from saved dataframe file
    df1 = readOne(path, IDs, tp_id=2, trial=2, activity='walking', freq=100)  # read single activity from 1 csv file

    # comment/uncomment to toggle the plot function
    #compare_plot()
    #dataDistribution(df)
    #single_plot(df1.iloc[1:400])
    compareF1Scores()
    #compareF1Scores2()
    #groupedBarPlot()

if __name__ == "__main__":
    main()


