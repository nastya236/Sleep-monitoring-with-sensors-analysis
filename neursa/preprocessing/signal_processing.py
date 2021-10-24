import numpy as np
from scipy import signal
import math
import scipy


def AdaptBP_weight(X, f0, beta, delta, mu, fe):
    f0 = f0 / fe
    N = X.shape[0]
    nsig = 1
    weights = np.zeros((N, nsig))
    Y = X
    alpha = np.ones(N + 1) * np.cos(2 * np.pi * f0)
    b = 0.5 * (1 - beta) * np.array([1, 0, -1])
    a = np.array([1, -alpha[1] * (beta + 1), beta])
    V = signal.lfilter(b, a, Y)
    Q = np.mean(V[2:99]*(V[1:98] + V[3:100]))
    P = np.mean(V[1:100]**2)
    J = V[3:100]-2 * alpha[1] * V[2:99]+V[1: 98]
    J = np.mean(J * J)
    S = np.mean((X[1:100])** 2)
    W = S / J
    weights[1, :] = W / np.sum(W)
    weights[2, :] = W / np.sum(W)
    for n in np.arange(3, N):
        Y[n] = alpha[n] * (beta + 1) * Y[n - 1] - beta * Y[n - 2] + 0.5 * (1 - beta) * (X[n] - X[n - 2])
        Q = delta * Q + (1 - delta) * (Y[n - 1] * (Y[n] + Y[n - 2]))
        P = delta * P + (1 - delta) * Y[n - 1] * Y[n - 1]
        J = mu * J + (1 - mu) * (Y[n] - 2 * alpha[n] * Y[n - 1]+Y[n - 2])** 2
        S = mu * S + (1 - mu) * X[n]** 2
        W = S / J
        weights[n, :] = W / np.sum(W)
        alpha[n + 1] = 0.5 * weights[n,:]*(Q / P)
    k = alpha.shape[0]
    alpha = alpha[1: k-1]
    alpha = alpha * (abs(alpha) < 1) + 1.0 * (alpha >= 1) - 1.0 * (alpha <= -1)
    IF = fe * np.real(np.arccos(alpha)) / 2 / np.pi
    return IF


def remove_trend(signal, a=1, cutoff=0.3, fs=256, numcoef=1025, pass_zero=False):
    b = scipy.signal.firwin(numcoef, cutoff, fs=fs, pass_zero=pass_zero)
    cleared_signal = scipy.signal.filtfilt(b, a, signal)
    return cleared_signal


def min_max_scaler(X):
    X_std = (X - np.min(X)) / (np.max(X) - np.min(X))

    return X_std


def get_rr_interval(peaks):
    rr_intervals = np.diff(peaks)

    return rr_intervals


def get_ecg_features(ecg_window,
                     distance=0.8*256,
                     length=10):
    without_trend = remove_trend(ecg_window)
    # scaled_z = min_max_scaler(without_trend)
    peaks = scipy.signal.find_peaks(without_trend, distance=distance)
    # rr_intervals = get_rr_interval(peaks[0])

    logs = np.zeros(ecg_window.shape[0])
    for i in peaks[0]:
        logs[i-length:i+length+1] = 1

    return logs, peaks

from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

