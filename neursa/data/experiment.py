import datetime
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
import scipy
from neursa.preprocessing.preparation import prepare_experiment
from neursa.preprocessing.signal_processing import butter_bandpass_filter, get_ecg_features
from neursa.utils import get_windows


class SensorsExperiment:
    """
    Attributes
    ----------

    """

    def __init__(self, path_to_folder,
                 info):
        self.path_to_folder = path_to_folder
        self.info = info
        self.load()

    def load(self):
        self.ECG = None
        sensors, time, labels, points = prepare_experiment(self.path_to_folder,
                                                     self.info)
        self.time = time
        self.labels_ = labels['labels']
        xp = labels['time_labels'].astype('datetime64[ns]').astype(int)
        x = self.time.astype('datetime64[ns]').astype(int)
        fp = labels['labels']
        result = np.interp(x, xp, fp)
        self.labels = result.astype(int)
        self.time_labels = labels['time_labels']
        for key, value in sensors.items():
            if key.find('piezo')!=-1:
                new_value = butter_bandpass_filter(value, 0.1, 100, 256)
                setattr(self, key, new_value)
            elif key.find('Thorax') != -1 or key.find('Abdominal') != -1:
                new_value = butter_bandpass_filter(value, 0.1, 15, 256)
                setattr(self, key, new_value)
            else:
                setattr(self, key, value)
        if self.path_to_folder == '/data/anvlfilippova/Institution/Recording 0721001/':
            self.ECG = self.Piezo12
        if self.path_to_folder == '/data/anvlfilippova/Institution/Recording 0721003/':
            self.ECG = self.ECG*-1
        self.ECG = butter_bandpass_filter(self.ECG, 0.3, 70, 256)
    def compile_items(self, window_size, step_size):
        items = []
        segments = zip(
            # get_windows(torch.Tensor(self.piezo1), window_size, step_size),
            # get_windows(torch.Tensor(self.piezo2), window_size, step_size),
            get_windows(torch.Tensor(self.piezo3), window_size, step_size),
            get_windows(torch.Tensor(self.piezo4), window_size, step_size),
            get_windows(torch.Tensor(self.piezo5), window_size, step_size),
            get_windows(torch.Tensor(self.piezoB), window_size, step_size),
            get_windows(torch.Tensor(self.piezoA), window_size, step_size),
            get_windows(torch.Tensor(self.piezoEF), window_size, step_size),
            get_windows(torch.Tensor(self.piezoDC), window_size, step_size),
            get_windows(torch.Tensor(self.ECG), window_size, step_size),
            # get_windows(torch.Tensor(self.Thorax), window_size, step_size),
            # get_windows(torch.Tensor(self.Abdomen), window_size, step_size),
            # get_windows(torch.Tensor(self.NasalPressure), window_size, step_size),
            # get_windows(torch.Tensor(self.Plethysmogram), window_size, step_size),
            get_windows(torch.Tensor(self.labels), window_size, step_size),
            torch.Tensor(self.labels_)
        )
        # ecg, thorax, abdomen, \
        #             nasal_pressure, plethysmogram,
        for piezo3, piezo4, piezo5, piezoB, \
            piezoA, piezoEF, piezoDC,ecg, \
            label, true_label in segments:
        # for ecg, thorax,abdomen, label, true_label in segments:
            item = {
                # 'piezo1': piezo1,
                # 'piezo2': piezo2,
                'piezo3': piezo3,
                'piezo4': piezo4,
                'piezo5': piezo5,
                'piezoB': piezoB,
                'piezoA': piezoA,
                'piezoEF': piezoEF,
                'piezoDC': piezoDC,
                'ecg': ecg,
                # 'thorax': thorax,
                # 'abdomen': abdomen,
                # 'nasal_pres': nasal_pressure,
                # 'pleth': plethysmogram,
                'logs':torch.Tensor(get_ecg_features(ecg.numpy())[0]),
                # 'peaks': torch.Tensor(get_ecg_features(ecg.numpy())[1][0]),
                'label': torch.mode(label)[0].long(),
                'true_label': true_label
            }
            items.append(item)
        return items

