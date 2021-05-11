"""
    time series classification with neural nets
    Multi-layer perceptron architecture
    no feature extraction - classification on 'raw' data

    Author: Simon Perneel - simon.perneel@hotmail.com
"""
from __future__ import print_function
from matplotlib import pyplot as plt

import numpy as np
import random
import pandas as pd
import seaborn as sns
import math
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.preprocessing import LabelEncoder, normalize

import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import gc
import time


# Some global parameters upfront
# Same labels will be reused throughout the program
LABELS = ['walking',
          'running',
          'sitting',
          'standing',
          'upstairs',
          'downstairs',
          'jumping']

# The number of steps within one time segment
TIME_PERIODS = 200  # 1 period = 0.01 s
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 100
EVAL_METHOD = 'L1O'  # 'k-fold' or 'L1O'


def create_segments(df, time_steps, step, label_name):
  N_FEATURES = 6  # acc+gyro norm. ankle, wrist, thigh.

  # Number of steps to advance in each iteration,
  # if equal to time_steps, no overlap between segments
  # step = time_steps
  segment = []
  labels = []
  subject = []
  num_segments = len(df) // step

  if num_segments == 0:
    num_segments = 1


  for i in range(0, num_segments):
    #print('start index', i*step)
    #print('end index ', i*step+time_steps)
    a = df['FreeAcc_norm_ankle'].iloc[i*step:(i*step)+time_steps].values
    b = df['FreeAcc_norm_wrist'].iloc[i*step:(i*step)+time_steps].values
    c = df['FreeAcc_norm_thigh'].iloc[i*step:(i*step)+time_steps].values
    d = df['Gyr_norm_ankle'].iloc[i*step:(i*step)+time_steps].values
    e = df['Gyr_norm_wrist'].iloc[i*step:(i*step)+time_steps].values
    f = df['Gyr_norm_thigh'].iloc[i*step:(i*step)+time_steps].values
    # pad with zeros if segment is shorter than {time_steps} samples
    a = np.pad(a, (0, time_steps - len(a)), mode='constant')
    b = np.pad(b, (0, time_steps - len(b)), mode='constant')
    c = np.pad(c, (0, time_steps - len(c)), mode='constant')
    d = np.pad(d, (0, time_steps - len(d)), mode='constant')
    e = np.pad(e, (0, time_steps - len(e)), mode='constant')
    f = np.pad(f, (0, time_steps - len(f)), mode='constant')

    # Retrieve the label of the segment
    label = df[label_name].iloc[0]
    subjectId = df['Subject-id'].iloc[0]
    segment.append([a, b, c, d, e, f])
    labels.append(label)
    subject.append(subjectId)

  # Bring segments in better shape
  segments = np.asarray(segment, dtype=np.float).reshape(-1, time_steps, N_FEATURES)
  labels = np.asarray(labels)

  return segments, labels, subject

def make_model(input_shape, num_classes):
  # Setting up the neural net
  model_m = Sequential()
  model_m.add(Reshape((TIME_PERIODS, 6), input_shape=(input_shape,)))
  model_m.add(Dense(100, activation='relu'))
  model_m.add(Dense(100, activation='relu'))
  model_m.add(Dense(100, activation='relu'))
  model_m.add(Flatten())
  model_m.add(Dense(num_classes, activation='softmax'))
  #print(model_m.summary())  # print layers
  model_m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model_m

def reset_seeds():
    np.random.seed(1)
    random.seed(2)
    if tf.__version__[0] == '2':
        tf.random.set_seed(3)
    else:
        tf.set_random_seed(3)
    #print("RANDOM SEEDS RESET")

def plotLearningCurve(history, model_m, x_tr, y_tr):
  plt.figure(figsize=(6, 4))
  plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
  plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
  plt.plot(history.history['loss'], 'r--', label='Loss of training data')
  plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
  plt.title('Model accuracy and loss')
  plt.ylabel('Accuracy and Loss')
  plt.xlabel('Training Epoch')
  plt.ylim(0)
  plt.legend()
  plt.show()

  #y_pred_train = model_m.predict(x_tr)
  # Take class with highest probability from train predictions
  #max_y_pred_train = np.argmax(y_pred_train, axis=1)
  #print(classification_report(y_tr, max_y_pred_train))


def show_confusion_matrix(validations, predictions, normalized=False):
    """
    :param validations: list containing the true label of the activities
    :param predictions: the predicted labels by the model
    :param normalized: boolean for normalized rows
    :return: nothing, plots and saves a confusion matrix
    """
    matrix = metrics.confusion_matrix(validations, predictions)
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
                linewidths=1,
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
    plt.savefig("Plots/ANNconfusionmatrix.png", format='png')
    plt.show()


def main():
  # Set some standard parameters upfront
  pd.options.display.float_format = '{:.1f}'.format
  pd.set_option('display.max_columns', None)
  sns.set()  # Default seaborn look and feel
  plt.style.use('ggplot')
  print('keras version ', keras.__version__)

  # load the preprocessed data from main.py
  df = pd.read_pickle("Data/data.pkl")  # enter filename
  # output first lines of the dataset
  # print(df.head())

  # Define column name of the label vector
  LABEL = 'ActivityEncoded'
  # Transform the labels from String to Integer via LabelEncoder
  le = LabelEncoder()
  # Add new column to the existing DataFrame with the encoded values
  df[LABEL] = le.fit_transform(df['Activity'].values.ravel())
  #print(df.head())

  # normalize features for training data set (values between 0 and 1)
  # surpress warning
  pd.options.mode.chained_assignment = None
  df["FreeAcc_norm_wrist"] = df["FreeAcc_norm_wrist"] / df["FreeAcc_norm_wrist"].max()
  df["FreeAcc_norm_ankle"] = df["FreeAcc_norm_ankle"] / df["FreeAcc_norm_ankle"].max()
  df["FreeAcc_norm_thigh"] = df["FreeAcc_norm_thigh"] / df["FreeAcc_norm_thigh"].max()
  df["Gyr_norm_wrist"] = df["Gyr_norm_wrist"] / df["Gyr_norm_wrist"].max()
  df["Gyr_norm_ankle"] = df["Gyr_norm_ankle"] / df["Gyr_norm_ankle"].max()
  df["Gyr_norm_thigh"] = df["Gyr_norm_thigh"] / df["Gyr_norm_thigh"].max()

  # round numbers
  df = df.round({"FreeAcc_norm_ankle": 8, "FreeAcc_norm_wrist": 8, "FreeAcc_norm_thigh": 8})
  df = df.round({"Gyr_norm_ankle": 8, "Gyr_norm_wrist": 8, "Gyr_norm_thigh": 8})

  # Segmentation of the data
  timeseries = df.groupby('Timeseries-id')  # get each trial and divide in segments

  #x_tr, y_tr, x_test, y_test = [], [], [], []
  X = []
  all_labels = []
  subjects = []

  for i in range(len(timeseries)):
    one_timeseries = timeseries.get_group(i)       # get an activity trial to divide in segments
    # return divided segments, their label and the id of the subject
    segments, labels, subject = create_segments(one_timeseries, TIME_PERIODS, STEP_DISTANCE, LABEL)
    X.extend(segments)
    all_labels.extend(labels)
    subjects.extend(subject)

  print('%i seconds of activity divided in %i segments of %.1f seconds' % (len(df), len(X), TIME_PERIODS/100))
  print("segments have %i %% overlap" % ((1-STEP_DISTANCE/TIME_PERIODS)*100))

  X = np.asarray(X)
  all_labels = np.asarray(all_labels)
  subjects = np.asarray(subjects)

  print('x shape: ', X.shape)
  print(X.shape[0], 'training samples')
  print('y shape: ', all_labels.shape)

  num_time_periods, num_sensors = X.shape[1], X.shape[2]
  print('num time periods: ', num_time_periods)
  print('num sensors:', num_sensors)
  num_classes = le.classes_.size
  print('activity classes', list(le.classes_))

  input_shape = (num_time_periods * num_sensors)

  # reshape to flattened representation of time slices as input to neural net
  X = X.reshape(X.shape[0], input_shape)
  print('x shape: ', X.shape)
  print('input shape: ', input_shape)

  X = X.astype('float32')
  all_labels = all_labels.astype('float32')

  # one hot encoding of the labels
  y_hot = np_utils.to_categorical(all_labels, num_classes)
  #print(y_hot, 'y hot')
  print('New y shape:', y_hot.shape)

  # Hyper-parameters
  BATCH_SIZE = 64  # amount of samples that will be propagated trough the network
  EPOCHS = 20  # Max amount of epochs
  acc_per_fold, loss_per_fold = [], []
  all_test_y, all_predicted_y = [], []

  callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
  ]

  # -------------------------------------------------------------------
  #                   TRAIN AND EVALUATE NET
  # -------------------------------------------------------------------
  if EVAL_METHOD == 'k-fold':
      k_fold = KFold(n_splits=10, shuffle=True)
      print('10-fold cross validation')
      for i_fold, (tr, tst) in enumerate(k_fold.split(X, y_hot)):
          print('testing with fold %i, training net with other folds' % (i_fold+1))
          # enable validation to use ModelCheckpoint and EarlyStopping callbacks
          X_train, X_test = X[tr], X[tst]
          y_train, y_test = y_hot[tr], y_hot[tst]

          model_m = make_model(input_shape, num_classes)

          history = model_m.fit(x=X_train,
                                y=y_train,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                callbacks=callbacks_list,
                                validation_data=(X_test, y_test),  # test data to validate
                                #validation_split=0.2,
                                verbose=0)

          y_pred_test = model_m.predict(X_test)

          # take class with highest probability from the test predictions
          max_y_pred_test = np.argmax(y_pred_test, axis=1)
          max_y_test = np.argmax(y_test, axis=1)
          all_test_y.extend(max_y_test)
          all_predicted_y.extend(max_y_pred_test)

          # evaluate model on test data with evaluate()
          scores = model_m.evaluate(X_test, y_test, verbose=1)
          print(f'Score for fold {i_fold+1}: {model_m.metrics_names[0]} of {scores[0]:.2f}; {model_m.metrics_names[1]} of {scores[1]*100:.2f}%')
          acc_per_fold.append(scores[1] * 100)
          loss_per_fold.append(scores[0])

          # remove parameters and weights for next fold
          del model_m
          K.clear_session()
          reset_seeds()


  if EVAL_METHOD == 'L1O':
      logo = LeaveOneGroupOut()
      print('Leave-One-Subject-Out cross validation')
      for i_fold, (tr, tst) in enumerate(logo.split(X, y_hot, groups=subjects)):
            print('Subject %i left out of training set and used as test set' % (i_fold+1))
            #print("train segments: ", len(tr), ", test segments: ", len(tst))
            X_train, X_test = X[tr], X[tst]
            y_train, y_test = y_hot[tr], y_hot[tst]

            model_m = make_model(input_shape, num_classes)
            history = model_m.fit(x=X_train,
                                  y=y_train,
                                  batch_size=BATCH_SIZE,
                                  epochs=EPOCHS,
                                  callbacks=callbacks_list,
                                  validation_data=(X_test, y_test),
                                  #validation_split=0.2,
                                  verbose=0)

            y_pred_test = model_m.predict(X_test)
            # take class with highest probability from the test predictions
            max_y_pred_test = np.argmax(y_pred_test, axis=1)
            max_y_test = np.argmax(y_test, axis=1)
            all_test_y.extend(max_y_test)
            all_predicted_y.extend(max_y_pred_test)

            # evaluate model on test data with evaluate()
            scores = model_m.evaluate(X_test, y_test, verbose=0)
            print(f'> Score for subject {(i_fold+1)}: {model_m.metrics_names[0]} of {scores[0]:.2f}; {model_m.metrics_names[1]} of {scores[1]*100:.2f}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            #show_confusion_matrix(max_y_test, max_y_pred_test)

            #plotLearningCurve(history, model_m, X_train, y_train)

            # remove parameters and weights for next fold
            del model_m
            K.clear_session()
            reset_seeds()

  # average scores
  print('Average scores for all folds:')
  print(f'> Accuracy: {np.mean(acc_per_fold):.2f} (+- {np.std(acc_per_fold):.2f})')
  print(f'> Loss: {np.mean(loss_per_fold):.2f}')
  show_confusion_matrix(all_test_y, all_predicted_y, normalized=True)
  print(classification_report(all_test_y, all_predicted_y))


if __name__ == "__main__":
  main()
