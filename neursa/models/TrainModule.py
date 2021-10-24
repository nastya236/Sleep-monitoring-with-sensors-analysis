import numpy as np
import os
import pandas as pd
import torch

from abc import ABC, abstractmethod
from tqdm import tqdm

from acclib.utils import AverageValueMeter, Logger
from acclib.data import get_parameters


class BaseModel(ABC):
    def __init__(self, backbone, device, target):
        super().__init__()
        self.backbone = backbone
        self.to(device)
        self.target = target

    @abstractmethod
    def _construct_prediction(self, out, batch_item, preprocess_params, process_params, segment_params):
        pass

    @abstractmethod
    def _initialize_evaluation(self, logger):
        pass

    @abstractmethod
    def _evaluate_prediction(self, prediction, experiment, logger, segment_params, selected_labels_metric=None):
        pass

    @abstractmethod
    def _compute_loss(self, criterion, pred_batch, y_batch):
        pass

    def to(self, device):
        self.device = device
        self.backbone = self.backbone.to(device)

    # def save(self, path):
    #     folder_path = os.path.dirname(path)
    #     if folder_path and not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #     torch.save(self.backbone, path)

    def train(self, criterion, optimizer, epochs, train_dataloader,
              valid_dataloader=None):
        # logger.log(self.backbone.get_parameters())
        best_loss = float('inf')
        for epoch in range(epochs):
            logger.log({'epochs': epoch + 1})
            if verbose:
                print(f'Epoch {epoch + 1}/{epochs}:', flush=True)
            mean_loss = self._run(criterion, optimizer, train_dataloader, train_mode=True,
                                  logger=logger, verbose=verbose)
            if valid_dataloader is not None:
                mean_loss = self._run(criterion, optimizer, valid_dataloader, train_mode=False,
                                      logger=logger, verbose=verbose)
            # if logger:
            #     self.save(f'{logger.path}/last.model')
            if mean_loss < best_loss:
                # self.save(f'{logger.path}/best.model')
                best_loss = mean_loss
        return best_loss

    # def predict(self, items, preprocess_params, process_params, segment_params):
    #     self.backbone.eval()
    #     batch_item = {name: torch.stack([item[name] for item in items]) for name in items[0].keys()}
    #     out = self._batch_result(batch_item).detach().cpu()
    #     return self._construct_prediction(out, batch_item, preprocess_params, process_params, segment_params)

    # def evaluate(self, dataset, logger=Logger(None), verbose=True, selected_labels_metric=None):
    #     self._initialize_evaluation(logger)
    #     all_metrics = []
    #     iterator = tqdm(dataset.experiments, desc='Evaluating model', disable=(not verbose))
    #     for index, experiment in enumerate(iterator):
    #         items = dataset.get_processed_experiment(index)
    #         prediction = self.predict(items, **dataset.get_params())
    #         metrics = self._evaluate_prediction(prediction, experiment, logger, dataset.get_params()['segment_params'],
    #                                             selected_labels_metric)
    #         if metrics is not None:
    #             experiment_params = get_parameters(dataset.storage, index)
    #             metrics.update(experiment_params)
    #             all_metrics.append(metrics)
    #     all_metrics = pd.DataFrame(all_metrics)
    #
    #     if all_metrics.empty:
    #         return (None, None)
    #     all_stats = all_metrics.describe().T
    #     all_stats['std'] /= np.sqrt(all_stats['count'] - 1)
    #     if logger:
    #         logger.log(all_stats['mean'].to_dict())
    #         all_stats.to_csv(f'{logger.path}/metrics.csv', index=False)
    #     return all_metrics, all_stats

    def _run(self, criterion, optimizer,
             dataloader, train_mode,
             logger, verbose):
        self.backbone.train(train_mode)
        # logger.train(train_mode)
        criterion_name = type(criterion).__name__
        iterator_desc = 'Train run' if train_mode else 'Valid run'
        iterator = tqdm(dataloader, desc=iterator_desc, disable=(not verbose))
        for item in iterator:
            pred_batch, loss = self._batch_update(item, criterion, optimizer, train_mode)
            loss = loss.cpu().detach().numpy()
            meter.add(loss)
        return meter.mean

    def _batch_update(self, item, criterion, optimizer, train_mode):
        if train_mode:
            pred_batch, loss = self._batch_result(item, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred_batch, loss = self._batch_result(item, criterion)
        return pred_batch, loss

    def _batch_result(self, item, criterion=None):
        x_batch = torch.cat((item['acc_s'], item['gyro_s']), dim=-1).permute(0, 2, 1).to(self.device)
        pred_batch = self.backbone(x_batch)
        if criterion is None:
            return pred_batch
        else:
            y_batch = item[self.target].to(self.device)
            loss = self._compute_loss(criterion, pred_batch, y_batch)
            return pred_batch, loss
