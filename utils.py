import numpy as np
import math
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

def zero_crossing_rate(signal):
    """
    :param signal: time signal to calculate zcr from
    :return: zcr: number that the signal crosses the zero line
    """
    signal = np.array(signal)
    zcr = ((signal[:-1] * signal[1:]) < 0).sum()

    return zcr

def count_peaks(signal):
    peaks, properties = find_peaks(signal, threshold=2)

    return len(peaks)

def MAD(x):
    """
    :param x: a pandas column
    :return: Median Absolute Deviation of the column
    """
    med = np.median(x)
    x = abs(x-med)
    MAD = 1.4826 * np.median(x)

    return MAD

def quaternion_multiply(quaternion1, quaternion0):
    """
    :param quaternion1: fi
    :param quaternion0:
    :return: multiplied quaternion (not communitative! q1*q0 =/= q0*q1
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1

    return np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                     w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                     w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], dtype=np.float64)

def quaternion_difference(quaternion1, quaternion0):
    """
    :param quaternion1: second quaternion
    :param quaternion0: first quaternion
    :return: rotation quaternion to get from orientation 0 to orientation 1
    """
    w0, x0, y0, z0 = quaternion0

    conjugate_q0 = w0, -x0, -y0, -z0

    diff_quaternion = quaternion_multiply(quaternion1, conjugate_q0)

    return diff_quaternion


def euler_from_quaternion(quaternion):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    w, x, y, z = quaternion
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

def DFT(signal, type):
    """
    :param signal: signal to transform to the frequency domain
    :param type: peak or frequency to return
    :return: the five largest signal peaks or their corresponding frequency
    """
    power_spectrum = False
    yf = rfft(np.array(signal))
    yf = np.abs(yf)
    if power_spectrum:
        abs_yf = np.abs(yf)
        yf = np.square(abs_yf)
    indices = yf.argsort()[::-1]  # indices of yf, sorted from large to small
    peak_indices = indices[:6]  # indices of the five largest peaking (excluding peak at 0 Hz, always largest)
    xf = rfftfreq(len(signal), (1/100))   # get corresponding frequencies

    if type =='peak':
        res = yf[peak_indices]
    if type =='freq':
        res = xf[peak_indices]

    return res

def orientationdiff(w, x, y, z):
    """
    :param w: array with first part of all quaternions in the segment (real part)
    :param x: array with second part of all quaternions in the segment
    :param y: array with third part of all quaternions in the segment
    :param z: array with fourth part of all quaternions in the segment
    :return: difference in orientation (roll, pitch, yaw) at the beginning vs. the end of the segment
    """

    quaternion0 = w.iloc[0], x.iloc[0], y.iloc[0], z.iloc[0]  # orientation at the beginning of the segment
    quaternion1 = w.iloc[-1], x.iloc[-1], y.iloc[-1], z.iloc[-1]  # orientation at the end of the segment
    q_diff = quaternion_difference(quaternion1, quaternion0)

    roll, pitch, yaw = euler_from_quaternion(q_diff)

    return [roll, pitch, yaw]
