from torch.utils.data import Dataset
from tqdm import tqdm
from neursa.data.experiment import SensorsExperiment
from neursa.preprocessing.preparation import parse_table_info

class SegmentsDataset(Dataset):
    def __init__(self, paths,
                 segment_params={'window_size': 256*30, 'step_size':256*30},
                 info_path = '/data/anvlfilippova/Institution/SleepSensor_Recordings list.xlsx'):
        self.paths = paths
        self.segment_params = segment_params
        self.info = parse_table_info(info_path)

    def process_experiments(self, limit=None):
        self.experiments = []
        self._items = []
        self._experiment_indices = [0]
        indices = range(len(self.paths))[:limit]
        indices_iterator = tqdm(indices, desc='Processing experiments')
        for i in indices_iterator:
            experiment = SensorsExperiment(self.paths[i], self.info)
            self.experiments.append(experiment)
            items = experiment.compile_items(**self.segment_params)
            self._items.extend(items)
            self._experiment_indices.append(len(self._items))
        return self

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def get_processed_experiment(self, index):
        if self._items is None:
            raise RuntimeError('experiments are not processed yet')
        items_slice = slice(self._experiment_indices[index], self._experiment_indices[index + 1])
        return self._items[items_slice]