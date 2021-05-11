"""
    Python code for my master thesis -
    identification of inertial features for the optimal recogition of physical activities

    script to load MTw XSens generated csv file into a dataframe,
    divide the time series IMU data in time windows,
    add and select good features from it,
    classify the activities using a learning algorithm
    and evaluate the model

		Author: Simon Perneel - simon.perneel@hotmail.com
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns; sns.set(context='paper')

import math
import numpy as np
from statistics import mean, median
import collections
from collections import Counter
import time

import glob
import ntpath

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings, feature_calculators

from sklearn import preprocessing, svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, normalize
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneGroupOut, cross_validate, KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.stats import kurtosis, skew

# own defined functions
import utils


def show_df_info(df):
    """
    :param df: dataframe to print some information of
    :return: nothing, just print some information
    """
    pd.set_option('display.max_columns', None)
    print('First row of the data table: ')
    print(df.head(1))
    #print(df.tail(1))
    print("............................................................")
    print('Number of columns in the dataframe: %i' % (df.shape[1]))
    print('Number of rows in the dataframe: %i' % (df.shape[0]))
    # n = len(pd.unique(dataframe['Subject-id']))
    # print('amount of test persons: ', n)
    print("............................................................")

def create_segments(df, time_steps, step):
    """
    :param df: pandas dataframe containing a trial of an activity
    :param time_steps: # time steps of one segment, 100 time steps = 1s
    :param step: number of steps to advance in each iteration, if equal to time_steps → no overlap between segments
    :return: segments: list containing the segments of one trial
             labels: list cointaing the labels of the segments (eg. ['running','running','running'])
    """
    #print('total length dataframe: ', (df['SampleTimeFine'].max() - df['SampleTimeFine'].min()))
    #print('time interval length: ', time_steps*0.01)

    num_segments = len(df) // step
    # if a trial less than 'time_steps' samples , create a shorter time window anyway
    if num_segments == 0:
        num_segments = 1
    #print('number of segments ', num_segments)

    segments = []
    labels = []

    for i in range(0, num_segments):
        #print('start index ', i*step)
        #print('end index ', (i*step)+time_steps)
        segment = df.iloc[i*step:(i*step)+time_steps]
        label = df["Activity"].iloc[i*step]
        segments.append(segment)
        labels.append(label)
        #print('actual interval length: ', (segments[i]['SampleTimeFine'].max()-segments[i]['SampleTimeFine'].min()))
        #print(*labels)

    return segments, labels

def auto_feat_extract(all_segments_df):
    """
    :param all_segments_df: dataframe of all segments which are identified by their 'Segment-id'
    :return: X: dataframe containing a number of features for each segment (auto-calculated using tsfresh package)
    """

    pd.set_option('display.max_columns', None)
    #print(all_segments_df.head(1))
    # drop irrelevant columns for feature extraction
    #cols = [0,1,5,6,7,14,15,16,21,22,23,24,28,29,30,37,38,39,40,41,42,
     #       43,44,45,46,47,51,52,53,60,61,62,63,64,65,66,67,68,69,70]
    # only use norm as features, drop all XYZ axis features
    cols = [range(71)]

    all_segments_df = all_segments_df.drop(all_segments_df.columns[cols], axis=1)
    all_segments_df = all_segments_df.drop(['Subject-id', 'Timeseries-id','Activity','ActivityEncoded','Mag_norm_wrist',
                                            'Mag_norm_thigh','Mag_norm_ankle'], axis=1)

    pd.set_option('display.max_columns', None)
    print(all_segments_df.head(1))

    #print(all_segments_df.head())
    # extract features
    minimal_settings = settings.MinimalFCParameters()
    X = extract_features(all_segments_df, column_id='Segment-id', default_fc_parameters=minimal_settings, impute_function=impute)
    #print(X.shape)
    #pd.set_option('display.max_columns', None)
    #print("features table: ", X.head())

    return X

def orientation_difference(x):
    """
    :param x: a segment of the time series
    :return: difference in roll, pitch and yaw (in radians) from the orientation at the beginning vs. orientation at the
    end of the segment, from the sensor at the thigh.
    """
    d = {}
    roll, pitch, yaw = utils.orientationdiff(x['Quat_q0_thigh'], x['Quat_q1_thigh'], x['Quat_q2_thigh'], x['Quat_q3_thigh'])
    d['roll_diff_thigh'] = roll
    d['pitch_diff_thigh'] = pitch
    d['yaw_diff_thigh'] = yaw
    return pd.Series(d, index=['roll_diff_thigh', 'pitch_diff_thigh', 'yaw_diff_thigh'])

def feat_extract(all_segments_df, feature_set):
    """
    :param feature_set: the set of features that is used
    :param all_segments_df: dataframe containing all segments which are identified by their 'Segment-id'
    :return: X: dataframe contained handcrafted features for each segment
    """
    pd.set_option('display.max_columns', None)
    if feature_set == 'set1':
        X = all_segments_df.groupby('Segment-id').agg(
                # statistical features
                kurtosis_acc_wrist=('FreeAcc_norm_wrist', lambda x: kurtosis(x)),
                kurtosis_acc_thigh=('FreeAcc_norm_thigh', lambda x: kurtosis(x)),
                kurtosis_acc_ankle=('FreeAcc_norm_ankle', lambda x: kurtosis(x)),
                kurtosis_gyr_wrist=('Gyr_norm_wrist', lambda x: kurtosis(x)),
                kurtosis_gyr_thigh=('Gyr_norm_thigh', lambda x: kurtosis(x)),
                kurtosis_gyr_ankle=('Gyr_norm_ankle', lambda x: kurtosis(x)),
                skew_acc_wrist=('FreeAcc_norm_wrist', lambda x: skew(x)),
                skew_acc_thigh=('FreeAcc_norm_thigh', lambda x: skew(x)),
                skew_acc_ankle=('FreeAcc_norm_ankle', lambda x: skew(x)),
                skew_gyr_wrist=('Gyr_norm_wrist', lambda x: skew(x)),
                skew_gyr_thigh=('Gyr_norm_thigh', lambda x: skew(x)),
                skew_gyr_ankle=('Gyr_norm_ankle', lambda x: skew(x)),
                mean_acc_wrist=('Acc_norm_wrist', mean),
                mean_acc_thigh=('Acc_norm_thigh', mean),
                mean_acc_ankle=('Acc_norm_ankle', mean),
                mean_gyr_wrist=('Gyr_norm_wrist', mean),
                mean_gyr_thigh=('Gyr_norm_thigh', mean),
                mean_gyr_ankle=('Gyr_norm_ankle', mean),
                #max_mag_X_thigh=('Mag_X_thigh', lambda x: max(x)),
                #max_mag_Y_thigh=('Mag_Y_thigh', lambda x: max(x)),
                #max_mag_Z_thigh=('Mag_Z_thigh', lambda x: max(x)),
                #max_mag_X_wrist=('Mag_X_wrist', lafmbda x: max(x)),
                #max_mag_Y_wrist=('Mag_Y_wrist', lambda x: max(x)),
                #max_mag_Z_wrist=('Mag_Z_wrist', lambda x: max(x)),
                std_acc_wrist=('Acc_norm_wrist', lambda x: np.std(x)),
                std_acc_thigh=('Acc_norm_thigh', lambda x: np.std(x)),
                std_acc_ankle=('Acc_norm_ankle', lambda x: np.std(x)),
                max_acc_wrist=('FreeAcc_norm_wrist', lambda x: max(x)),
                max_acc_thigh=('FreeAcc_norm_thigh', lambda x: max(x)),
                max_acc_ankle=('FreeAcc_norm_ankle', lambda x: max(x)),
                # time features
                rms_acc_wrist=('FreeAcc_norm_wrist', lambda x: np.sqrt(mean(x)**2)),
                rms_acc_thigh=('FreeAcc_norm_thigh', lambda x: np.sqrt(mean(x)**2)),
                rms_acc_ankle=('FreeAcc_norm_ankle', lambda x: np.sqrt(mean(x)**2)),
                rms_gyr_wrist=('Gyr_norm_wrist', lambda x: np.sqrt(mean(x)**2)),
                rms_gyr_thigh=('Gyr_norm_thigh', lambda x: np.sqrt(mean(x)**2)),
                rms_gyr_ankle=('Gyr_norm_ankle', lambda x: np.sqrt(mean(x)**2)),
                autocorr_wrist_10=('Acc_norm_wrist', lambda x: x.autocorr(lag=10)),
                autocorr_thigh_10=('Acc_norm_thigh', lambda x: x.autocorr(lag=10)),
                autocorr_ankle_10=('Acc_norm_ankle', lambda x: x.autocorr(lag=10)),
                variance_acc_wrist=('FreeAcc_norm_wrist', lambda x: np.var(x)),
                variance_acc_thigh=('FreeAcc_norm_thigh', lambda x: np.var(x)),
                variance_acc_ankle=('FreeAcc_norm_ankle', lambda x: np.var(x)),
                variance_gyr_wrist=('Gyr_norm_wrist', lambda x: np.var(x)),
                variance_gyr_thigh=('Gyr_norm_wrist', lambda x: np.var(x)),
                variance_gyr_ankle=('Gyr_norm_wrist', lambda x: np.var(x)),
                # frequency features
                first_spectral_peak_thigh=('Acc_norm_thigh', lambda x: utils.DFT(x, 'peak')[1]),
                second_spectral_peak_thigh=('Acc_norm_thigh', lambda x: utils.DFT(x, 'peak')[2]),
                third_spectral_peak_thigh=('Acc_norm_thigh', lambda x: utils.DFT(x, 'peak')[3]),
                fourth_spectral_peak_thigh=('Acc_norm_thigh', lambda x: utils.DFT(x, 'peak')[4]),
                fifth_spectral_peak_thigh=('Acc_norm_thigh', lambda x: utils.DFT(x, 'peak')[5]),
                first_spectral_peak_freq_thigh=('Acc_norm_thigh', lambda x: utils.DFT(x, 'freq')[1]),
                second_spectral_peak_freq_thigh=('Acc_norm_thigh', lambda x: utils.DFT(x, 'freq')[2]),
                third_spectral_peak_freq_thigh=('Acc_norm_thigh', lambda x: utils.DFT(x, 'freq')[3]),
                fourth_spectral_peak_freq_thigh=('Acc_norm_thigh', lambda x: utils.DFT(x, 'freq')[4]),
                fifth_spectral_peak_freq_thigh=('Acc_norm_thigh', lambda x: utils.DFT(x, 'freq')[5]),
                first_spectral_peak_freq_wrist=('Acc_norm_wrist', lambda x: utils.DFT(x, 'freq')[1]),
                second_spectral_peak_freq_wrist=('Acc_norm_wrist', lambda x: utils.DFT(x, 'freq')[2]),
                third_spectral_peak_freq_wrist=('Acc_norm_wrist', lambda x: utils.DFT(x, 'freq')[3]),
                fourth_spectral_peak_freq_wrist=('Acc_norm_wrist', lambda x: utils.DFT(x, 'freq')[4]),
                fifth_spectral_peak_freq_wrist=('Acc_norm_wrist', lambda x: utils.DFT(x, 'freq')[5]),
                # label
                label=('Activity', 'first'),
            )
        # additional orientation feature, calculated with apply function
        Y = all_segments_df.groupby('Segment-id').apply(orientation_difference)
    # add orientation feature with the other features
    X = pd.merge(X, Y, on=('Segment-id'))

    if feature_set == 'set2':
    # for testing other combinations of features
        X = all_segments_df.groupby('Segment-id').agg(
            kurtosis_wrist=('Acc_norm_wrist', lambda x: kurtosis(x)),
            label=('Activity', 'first')
            )

    print(X.head())
    print('%i features are extracted' % len(X.columns))
    all_labels = X.pop('label')
    all_labels = np.array(all_labels)

    return X, all_labels

def normalize_df(dataframe):
    cols_to_norm = ['Acc_norm_ankle','Acc_norm_wrist','Acc_norm_thigh', 'FreeAcc_norm_ankle','FreeAcc_norm_wrist',
                    'FreeAcc_norm_thigh', 'Gyr_norm_ankle', 'Gyr_norm_wrist', 'Gyr_norm_thigh', 'Mag_norm_ankle',
                    'Mag_norm_wrist', 'Mag_norm_thigh']
    dataframe[cols_to_norm] = dataframe.groupby('Subject-id')[cols_to_norm].transform(lambda x: (x - x.min()) / x.max() - x.min())

    return dataframe

def standardize_subject_features(X):
    """
    :param X: feature vector + subject id for each column
    :return: standardize features at SUBJECT level
    accelerometry data is not directly comparable across subjects, standardize instead of normalization because of outliers
    """
    X_normalized = X.groupby('Subject-id').transform(lambda x: (x - x.mean()) / x.std())
    X.pop('Subject-id')

    return X_normalized

def plot(dataframe):
    """
    :param dataframe: contains time-series data from one activity trial
    :return: nothing, just plots some features of the timeseries data
    """
    width = 15  # inches
    golden_mean = (math.sqrt(5) - 1.0) / 2.0  # aesthetic ratio
    height = width * golden_mean  # inches
    figure, axes = plt.subplots(3, 1, figsize=(width, height))
    plt.autoscale(enable=True, axis='both', tight=None)
    # plot sensor measurements (stored in a dataframe)
    #dataframe.plot(x='SampleTimeFine', y=['FreeAcc_X_wrist','FreeAcc_Y_wrist','FreeAcc_Z_wrist'], label=['X','Y','Z'], linewidth=1.5, ax=axes[0])
    #dataframe.plot(x='SampleTimeFine', y=['FreeAcc_X_thigh','FreeAcc_Y_thigh','FreeAcc_Z_thigh'], label=['X','Y','Z'], linewidth=1.5, ax=axes[1])
    #dataframe.plot(x='SampleTimeFine', y=['FreeAcc_X_ankle','FreeAcc_Y_ankle','FreeAcc_Z_ankle'], label=['X','Y','Z'], linewidth=1.5, ax=axes[2])
    #dataframe.plot(x='SampleTimeFine', y=['FreeAcc_X_wrist', 'FreeAcc_Y_wrist', 'FreeAcc_Z_wrist'], label=['X mag.', 'Y mag.', 'Z mag.'], linewidth=1.5, ax=axes[0])
    #dataframe.plot(x='SampleTimeFine', y=['FreeAcc_X_thigh', 'FreeAcc_Y_thigh', 'FreeAcc_Z_thigh'], label=['X mag.', 'Y mag.', 'Z mag.'], linewidth=1.5, ax=axes[1])
    #dataframe.plot(x='SampleTimeFine', y=['FreeAcc_X_ankle', 'FreeAcc_Y_ankle','FreeAcc_Z_ankle'], label=['X mag.', 'Y mag.', 'Z mag.'], linewidth=1.5, ax=axes[2])
    #dataframe.plot(x='SampleTimeFine', y=['Mag_X_wrist', 'Mag_Y_wrist','Mag_Z_wrist'], label=['X', 'Y', 'Z'], linewidth=1.5, ax=axes[0])
    #dataframe.plot(x='SampleTimeFine', y=['Mag_X_thigh', 'Mag_Y_thigh','Mag_Z_thigh'], label=['X', 'Y', 'Z'], linewidth=1.5, ax=axes[1])
    #dataframe.plot(x='SampleTimeFine', y=['Mag_X_ankle', 'Mag_Y_ankle', 'Mag_Z_ankle'], label=['X', 'Y', 'Z'], linewidth=1.5, ax=axes[2])
    # norm of acceleration
    dataframe.plot(x='SampleTimeFine', y=['Acc_norm_wrist'], label=['Wrist'], linewidth=1.5, ax=axes[0])
    dataframe.plot(x='SampleTimeFine', y=['Acc_norm_thigh'], label=['Thigh'], linewidth=1.5, ax=axes[1])
    dataframe.plot(x='SampleTimeFine', y=['Acc_norm_ankle'], label=['Ankle'], linewidth=1.5, ax=axes[2])

    # plot layout
    activity = dataframe['Activity'].iloc[0]
    subject = dataframe['Subject-id'].iloc[0]
    figure.suptitle('activity: %s, subject: %s' % (activity, subject)), plt.legend()
    plt.subplots_adjust(hspace=0.4)
    axes[0].set_title('wrist')
    axes[1].set_title('thigh')
    axes[2].set_title('ankle')
    for ax in axes:
        ax.set(xlabel="Time [s]", ylabel="Acceleration [m/s²]")
    plt.show()
    plt.savefig('Plots/Activityplot.png', format='png')

def plot_comparison(df1, df2):
    width = 15  # inches
    golden_mean = (math.sqrt(5) - 1.0) / 2.0  # aesthetic ratio
    height = width * golden_mean  # inches
    figure, axes = plt.subplots(2, 1, figsize=(width, height))
    plt.autoscale(enable=True, axis='both', tight=None)
    #df1.plot(x='SampleTimeFine', y=['Acc_norm_wrist', 'Acc_norm_thigh', 'Acc_norm_ankle'], label=['wrist', 'thigh', 'ankle'], linewidth=1.5, ax=axes[0])
    #df2.plot(x='SampleTimeFine', y=['Acc_norm_wrist', 'Acc_norm_thigh', 'Acc_norm_ankle'], label=['wrist', 'thigh', 'ankle'], linewidth=1.5, ax=axes[1])
    df1.plot(x='SampleTimeFine', y=['FreeAcc_norm_thigh', 'Acc_norm_thigh'], label=['free acc', 'acc'], linewidth=1.5, ax=axes[0])
    df2.plot(x='SampleTimeFine', y=['FreeAcc_norm_thigh', 'Acc_norm_thigh'], label=['free acc', 'acc'], linewidth=1.5, ax=axes[1])

    plt.subplots_adjust(hspace=0.4)
    for ax in axes:
        ax.set(xlabel="Time [s]", ylabel="Acceleration norm [m/s²]")
    activity1 = df1['Activity'].iloc[0]
    activity2 = df2['Activity'].iloc[0]
    axes[0].set_title(activity1)
    axes[1].set_title(activity2)
    plt.show()

def plot_DT(tree, columns, dataframe):
    """
    :param tree: classifier object, containing the decision tree
    :param columns: column names of the feature vector
    :param dataframe: time series dataframe to have the activity names
    :return: nothing, plot and save the decision tree
    """
    fig = plt.figure(figsize=(10, 7.5))
    _ = plot_tree(tree, feature_names=columns.columns, class_names=dataframe["Activity"].unique(), filled=True)
    fig.savefig("Plots/DT.png", format='png', bbox_inches='tight')

def plot_tsne(feature_vector, all_labels):
    """
    :param feature_vector: dataframe containing the multi-D features from each segment
    :param all_labels: list containing activity type labels of each segment
    :return: nothing
    """
    tsne = TSNE(perplexity=25, n_components=2, init='random', n_iter=1000)
    feature_vector_2d = tsne.fit_transform(np.array(feature_vector))

    # normalize the feature vector to be between 0 and 1
    feature_vector_2d -= feature_vector_2d.min(axis=0)
    feature_vector_2d /= feature_vector_2d.max(axis=0)

    # separate out the X and Y points
    x = feature_vector_2d[:, 0]
    y = feature_vector_2d[:, 1]

    # create scatter plot to show where activities are embedded
    sns.set_palette('Set1', n_colors=7)
    #fig = plt.figure(figsize=[12, 12])
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(*zip(*feature_vector_2d))

    fig, ax = plt.subplots(figsize=[12, 12])
    sns.scatterplot(x=x, y=y, hue=all_labels, style=all_labels, s=80)

    plt.title('t-SNE plot of the multi-dimensional feature space')
    plt.show()

def plot_features(X, all_labels):
    sns.scatterplot(x=X.rms_acc_thigh, y=X.rms_acc_wrist, hue=all_labels)
    plt.show()
    print('test')

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
    for bodypart in ['ankle', 'wrist', 'thigh']:
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

def segmentation(df, TIME_PERIODS, STEP_DISTANCE):
    """
    :param df: dataframe containing all trials of all subjects
    :param TIME_PERIODS: # time periods of a segment, 1 period = 0.01s
    :param STEP_DISTANCE: amount of steps to advance each iteration
    :return: all_segments: list containing segmented data
             all_labels: list containing the activity label of each segment
    """
    timeseries = df.groupby('Timeseries-id')  # get each trial and divide in segments
    all_segments = []
    all_labels = []

    # for each trial of a subject
    # divide timeseries in segments of TIME_PERIODS/freq seconds
    for i in range(len(timeseries)):
        one_timeseries = timeseries.get_group(i)
        segments, labels = create_segments(one_timeseries, TIME_PERIODS, STEP_DISTANCE)
        all_segments.extend(segments)
        all_labels.extend(labels)

    # add segment id for each segment in list
    for i in range(len(all_segments)):
        segment = all_segments[i]
        segment.insert(80, "Segment-id", i, True)

    return all_segments, all_labels

def show_confusion_matrix(validations, predictions, normalized=False):
    """
    :param validations: list containing the true label of the activities
    :param predictions: the predicted labels by the model
    :param normalized: whether or not to make a normalized matrix
    :return: nothing, plots and saves a confusion matrix
    """
    matrix = confusion_matrix(validations, predictions)
    if normalized:
        matrix = normalize(matrix, axis=1, norm='l1')
        fmt = '.1%'
    else:
        fmt = 'd'
    width = 12  # inches
    golden_mean = (math.sqrt(5) - 1.0) / 2.0  # aesthetic ratio
    height = width * golden_mean  # inches
    plt.figure(figsize=(width, height))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=0.5,
                xticklabels=['Downstairs', 'Jumping', 'Running', 'Sitting down', 'Standing up', 'Upstairs', 'Walking'],
                yticklabels=['Downstairs', 'Jumping', 'Running','Sitting down', 'Standing up', 'Upstairs', 'Walking'],
                annot=True,
                fmt=fmt
                )
    plt.yticks(rotation=0)
    if normalized:
        plt.title('Normalized Confusion Matrix')
    else:
        plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("Plots/confusionmatrixDT.png", format='png')
    plt.show()


def preprocess(path, freq=100):
    """
    :param path: location of the csv files exported out of the Mtw software suite
    :param freq: sample frequency of the measurements
    :return: dataframe containing data from 3 sensors
    """
    files = glob.glob(path + "/*.csv")
    li = []
    print('csv files in path: %s are loading' % path)
    print(len(files), ' files')

    # iterate per 3 files, combine 3 csv files from each sensor
    for index in range(0, len(files) - 1, 3):
        csv_1 = pd.read_csv(files[index], sep=',', header=4)  # csv file from sensor 1
        csv_2 = pd.read_csv(files[index + 1], sep=',', header=4)  # sensor 2
        csv_3 = pd.read_csv(files[index + 2], sep=',', header=4)  # sensor 3

        filename = ntpath.basename(files[index])
        # retrieve parameters from filename
        tp = filename.find('tp')
        separator = filename.find('-')
        subject_id = int(filename[tp + 2:separator])
        activity_end = filename.find('-000')
        activity = filename[separator + 3:activity_end]
        trial = int(filename[separator + 1:separator + 2])
        # sensorID = filename.find('000_')
        # sensor = filename[sensorID+4:sensorID+12]

        # merge data from 3 the sensors in one row for each sample
        csv_tmp = pd.merge(csv_1, csv_2, on=('PacketCounter', 'SampleTimeFine'), suffixes=('_thigh', None))
        csv = pd.merge(csv_3, csv_tmp, on=('PacketCounter', 'SampleTimeFine'), suffixes=('_ankle', '_wrist'))
        pd.set_option('display.max_columns', None)
        timeseries_id = int(index / 3)
        add_values(csv, subject_id, activity, timeseries_id, freq)
        li.append(csv)
        print('%.1f %% loaded' % (index/len(files)*100))

    print('100 % loaded')
    dataframe = pd.concat(li, axis=0, ignore_index=True)
    # add encoding for the activity classes
    le = preprocessing.LabelEncoder()
    dataframe["ActivityEncoded"] = le.fit_transform(dataframe["Activity"].values.ravel())
    show_df_info(dataframe)

    return dataframe


def calc_feat_importances(X_train, classifier):
    """
    :param X_train: feature vector of the training set
    :param classifier: the decision tree object
    :return: importance of all used features of the classifier
    """
    feat_importances = dict(zip(X_train.columns, classifier.feature_importances_))  # get feature importance

    return feat_importances

def plot_feat_importances(feat_importances):
    """
    :param feat_importances: dictionary containing feature names and their importance score
    :return: nothing, makes a barplot
    """
    feat_importances = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)  # rank importances descending
    w = 12
    golden_mean = (math.sqrt(5) - 1.0) / 2.0  # aesthetic ratio
    h = w * golden_mean  # inches
    fig, ax = plt.subplots(figsize=(w,h))
    x, y = zip(*feat_importances)
    x, y = list(x), list(y)
    sns.barplot(x=x[:], y=y[:])  # plot 10 most important features
    plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    plt.title('Features ranked by importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig('Plots/feature_importance.png', bbox_inches="tight", format='png')

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

    csv_tmp = pd.merge(csvs[0], csvs[1], on=('PacketCounter', 'SampleTimeFine'), suffixes=('_thigh', None))
    dataframe = pd.merge(csvs[2], csv_tmp, on=('PacketCounter', 'SampleTimeFine'), suffixes=('_ankle', '_wrist'))
    # add more features
    timeseries_id = 1
    add_values(dataframe, tp_id, activity, timeseries_id, freq)

    return dataframe

def main():
    # define sensor IDs
    ID1 = '00B42D71'  # Wrist
    ID2 = '00B42D0F'  # Thigh
    ID3 = '00B42D95'  # Ankle
    IDs = [ID1, ID2, ID3]
    LABELS = ['Downstairs', 'Jumping', 'Running', 'Sitting', 'Standing', 'Upstairs', 'Walking']

    # ------------------------ HYPER-PARAMETERS ---------------------------------------
    # ---------parameters for reading in a single activity trail ---------------------
    # relative path containing the csv files, filenames follow template (see readme.md)
    path = r".\Data\Exported"

    # to read in one csv file
    tp_id = 1
    trial = 4
    activity = 'jumping'
    freq = 100  # Hz, sample rate of the sensors
    # -----------parameters for segmentation----------------------------------
    # The number of steps within one time segment
    TIME_PERIODS = 400  # 1 period = 0.01 s
    # The steps to take from one segment to the next; if this value is equal to
    # TIME_PERIODS, then there is no overlap between the segments
    STEP_DISTANCE = 200
    # ---------------evaluation method---------------------------------------------------
    eval_method = 'L1O'  # 'L1O' or 'k-Fold' cross-validation


    # Read in csv files to one big dataframe containing all activities
    dataframe = preprocess(path, freq)  # read from path
    #df1 = readOne(path, IDs, tp_id=5, trial=2, activity='standing', freq=100)  # read single activity from 1 csv file
    #df2 = readOne(path, IDs, tp_id=5, trial=2, activity='sitting', freq=100)  # read single activity from 1 csv file

    #dataframe = pd.read_pickle("Data/data.pkl")  # read from saved pre-process dataframe file

    #all_segments_df = pd.read_pickle("Data/all_segments.pkl")
    print("XSens data imported correctly")

    # save and export read data
    #dataframe.to_pickle("Data/data.pkl")  # save the dataframe
    #dataframe.to_csv("Data/processed_data.csv", index=False)

    # output some info
    #show_df_info(dataframe)  # basic info
    #print(dataframe['Subject-id'].value_counts(normalize='True')) # data distribution along the subjects

    # plot some activities
    #df1 = dataframe.loc[dataframe['Timeseries-id'] == 12]
    #plot(df1)
    #plot_comparison(df1, df2)


    # segmentation of the data in time windows
    all_segments, all_labels = segmentation(dataframe, TIME_PERIODS, STEP_DISTANCE)
    print('%i seconds of activity divided in %i segments of %.1f seconds' % ((len(dataframe)//100), len(all_segments), TIME_PERIODS/freq))
    print("segments have %i %% overlap" % ((1-STEP_DISTANCE/TIME_PERIODS)*100))

    all_segments_df = pd.concat(all_segments, ignore_index=True)  # put all segments back in a dataframe

    # filter to keep only certain activities (for testing)
    #all_segments_df = all_segments_df[(all_segments_df['Activity'] == 'standing') | (all_segments_df['Activity'] == 'sitting')]
    #all_segments_df = all_segments_df.loc[all_segments_df['Activity'] == 'standing']
    #all_segments_df = all_segments_df.loc[all_segments_df['Activity'] == 'sitting']

    #segment = all_segments[113]
    #plot(segment)

    # get each segment
    segments = all_segments_df.groupby('Segment-id')
    all_labels = np.array(all_labels)
    groups = []
    # make list of subject-ids to make groups for cv
    for segment_name, segment in segments:
        subject = segment["Subject-id"].iloc[0]  #find the subject of each segment
        groups.append(subject)

    # ------------------ FEATURE EXTRACTION -----------------------
    print('features are extracted from the segments...')
    # X is the vector containing all feature values
    # automatic feature extraction with tsfresh or handcrafted features
    #X = auto_feat_extract(all_segments_df)
    X, all_labels = feat_extract(all_segments_df, 'set1')

    #plot_features(X, all_labels)

    # exploratory data analysis
    # visualization of the features in 2D
    #plot_tsne(X, all_labels)
    print('Done')

    # -------------------- FEATURE SELECTION --------------------------
    """"rf = RandomForestClassifier(n_estimators=10, max_depth=20, n_jobs=-1)
    kn = KNeighborsClassifier(n_neighbors=3)
    n_features = 10
    sfs = SequentialFeatureSelector(kn, n_features_to_select=n_features)
    sfs.fit(X, all_labels)
    print(f"Top {n_features} features selected by forward sequential selection:{list(X.columns[sfs.get_support()])}")"""

    # ----------- NORMALIZATION/STANDARDIZATION---------------
    X['Subject-id'] = groups  # add column with subject-id for subject-level normalization/standardization
    X = standardize_subject_features(X)
    print('Features are normalized')

    # ------------------- PCA --------------------------------------
    """# standardization needed with PCA
    n = 30  # number of features to reduce to
    pca = PCA(n_components=n)
    print(f'features reduced to {n} components')
    X = pd.DataFrame(pca.fit_transform(X))"""

    # ------------- CLASSIFICATION AND CROSS-VALIDATION -------------
    # variables to keep track of general accuracy
    all_y_test = []
    all_predicted_y = []
    sum_feat_importances = {}

    if eval_method == 'L1O':
        print('Leave-One-Out-Cross-Validation')
        logo = LeaveOneGroupOut()
        for i_fold, (train_index, test_index) in enumerate(logo.split(X, all_labels, groups=groups)):
            print('Subject %i left out of training set and used as test set' % (i_fold+1))
            #print("TRAIN", train_index, "TEST", test_index)
            #print("train samples: ", len(train_index), ", test samples: ", len(test_index))
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = all_labels[train_index], all_labels[test_index]
            y_train, y_test = y_train.tolist(), y_test.tolist()
            #print('train labels', *y_train)
            #clf = svm.SVC(kernel='rbf', probability=True, decision_function_shape='ovo')
            #clf = svm.LinearSVC(max_iter=10000)
            #clf = DecisionTreeClassifier(criterion='gini')
            clf = RandomForestClassifier(max_depth=20)
            #clf = GaussianNB(priors=[0.192, 0.032,0.18,0.021,0.021,0.211,0.343])
            #clf = MLPClassifier(random_state=1, max_iter=300)
            #clf = GradientBoostingClassifier(n_estimators=20)
            #clf = KNeighborsClassifier(n_neighbors=5)
            clf.fit(X_train, y_train)
            predicted_y = clf.predict(X_test)
            all_y_test.extend(y_test)
            all_predicted_y.extend(predicted_y)

            report = classification_report(y_test, predicted_y, output_dict=True)
            print('> weighted average precision: %.2f' % report.get('weighted avg').get('precision'))
            #show_confusion_matrix(y_test, predicted_y, normalized=False)
            feat_importances = calc_feat_importances(X_train, clf)  # calculates and plots the feature importances
            sum_feat_importances = Counter(sum_feat_importances) + Counter(feat_importances)  # accumulate feature importances

    if eval_method == 'k-fold':
        print('K-fold Cross-Validation')
        k_fold = KFold(n_splits=10, shuffle=True)
        for i_fold, (tr, tst) in enumerate(k_fold.split(X, all_labels)):
            print('fold number %i : ' % i_fold)
            X_train, X_test = X.iloc[tr], X.iloc[tst]
            y_train, y_test = all_labels[tr], all_labels[tst]
            #clf = DecisionTreeClassifier(criterion='gini')
            clf = RandomForestClassifier(max_depth=20)
            #clf = KNeighborsClassifier(n_neighbors=3)
            #clf = svm.SVC(kernel='linear', decision_function_shape='ovo', C=1)
            #clf = GaussianNB(priors=[0.192, 0.032,0.18,0.021,0.021,0.211,0.343])
            clf.fit(X_train, y_train)
            predicted_y = clf.predict(X_test)
            all_y_test.extend(y_test)
            all_predicted_y.extend(predicted_y)

            report = classification_report(y_test, predicted_y, output_dict=True)
            print('> weighted average precision: %.2f' % report.get('weighted avg').get('precision'))
            #show_confusion_matrix(y_test, predicted_y, normalized=False)
            #feat_importances = calc_feat_importances(X_train, clf)  # calculates and plots the feature importances
            #sum_feat_importances = Counter(sum_feat_importances) + Counter(feat_importances)  # accumulate feature importances


    # report general accuracy of the model
    print(classification_report(all_y_test, all_predicted_y))
    show_confusion_matrix(all_y_test, all_predicted_y, normalized=False)

    #plot_DT(clf, X, dataframe)  # plot visual representation a DT classifier
    plot_feat_importances(sum_feat_importances)  # plot accumulated feat. importances of each trained model

    print('Data processed correctly')
    print('............................................................')


if __name__ == "__main__":
    main()

    # DRAFT

    # train-test split
    """

    # Decision tree
    clf = DecisionTreeClassifier(criterion='gini')
    # Random forest
    #clf = RandomForestClassifier(max_depth=20)
    # Nearest neighbours
    #clf = KNeighborsClassifier(n_neighbors=3)
    # Support Vector Machine
    #clf = svm.SVC(kernel='linear', decision_function_shape='ovo', C=1)

    clf.fit(X_train, y_train)
    
    # model evaluation
    #print(classification_report(y_test, clf.predict(X_test)))
    #show_confusion_matrix(y_test, clf.predict(X_test))

    """
    # fft transform
    """
    _, axes = plt.subplots(3, 1, figsize=(10, 8))
    yf_wrist = rfft(np.array(segment['Acc_norm_wrist']))
    maximum = max((np.abs(yf_wrist[1:])))
    print("max", maximum)
    yf_thigh = rfft(np.array(segment['Acc_norm_thigh']))
    yf_ankle = rfft(np.array(segment['Acc_norm_ankle']))
    xf = rfftfreq(400, (1/freq))
    print(len(xf))
    axes[0].stem(xf[:], np.abs(yf1[:]), '-.')
    axes[1].stem(xf[:], np.abs(yf2[:]),'-.')
    axes[2].stem(xf[:], np.abs(yf3[:]),'-.')
    plt.show()

    """
    """
            # normalization of the features
        min = dataframe[f"FreeAcc_norm_{bodypart}"].min()
        max = dataframe[f"FreeAcc_norm_{bodypart}"].max()
        dataframe[f"FreeAcc_norm_{bodypart}"] = [normalize(x, min, max) for x in dataframe[f"FreeAcc_norm_{bodypart}"]]
        dataframe[f"Gyr_norm_{bodypart}"] = [normalize(x, min, max) for x in dataframe[f"Gyr_norm_{bodypart}"]
    
    """
    # quaternion stuff
    """
    quaternions = df1[['Quat_q0_thigh','Quat_q1_thigh','Quat_q2_thigh','Quat_q3_thigh']]
    quat1 = quaternions.iloc[0].values
    quat2 = quaternions.iloc[100].values
    print(quat1)
    print(quat2)
    multi_quat = quaternion_multiply(quat1, quat2)
    print(multi_quat)
    """


