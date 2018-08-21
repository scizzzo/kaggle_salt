import os
import shutil
import torch

from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from os.path import join
from collections import defaultdict

from dataset import SaltTrainDataset as CustomDataset



RESULT_PATH = './models/'


class PyTrainer:
    def __init__(self, model_fn,  optimizer_fn, lr, criterion, metrics):
        self.model_fn = model_fn
        self.criterion = criterion
        self.optimizer_fn = optimizer_fn
        self.metrics = metrics
        self.lr = lr

    def _train_model(self, epochs, train_dataloader, val_dataloader, output_dirpath):
        self.model.train()

        best_val_loss = float('inf')
        it = 0
        val_it = 0
        best_val_metrics = {}
        for epoch in range(epochs):
            n = len(train_dataloader)
            tq = tqdm(total=n)
            tq.set_description('Epoch {}, lr {}'.format(epoch + 1, self.lr))
            tq.refresh()
            epoch_progress = 0
            for inputs, targets in train_dataloader:
                it += 1
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                postfix = {'loss': '{:.3f}'.format(loss.item())}
                for name, metric in self.metrics.items():
                    metric_value = metric(outputs, targets)
                    postfix[name] = metric_value
                tq.set_postfix(**postfix)
                tq.update(1)

                epoch_progress += 1

            val_it += 1
            model_dirpath = join(output_dirpath, 'model.pt')
            torch.save(self.model.state_dict(), model_dirpath)
            metric_values = self._evaluate(val_dataloader,
                                    metrics={'loss': self.criterion, **self.metrics})
            self.model.train()
            val_loss = metric_values['loss']
            if val_loss < best_val_loss:
                best_val_metrics = metric_values
                best_val_loss = val_loss
                best_model_dirpath = join(output_dirpath, 'bestModel.pt')
                shutil.copy(model_dirpath, best_model_dirpath)
            tq.close()
        return best_val_metrics

    def _evaluate(self, dataloader, metrics={}):
        self.model.eval()
        tq = tqdm(total=len(dataloader))
        tq.set_description('Evaluating')
        tq.refresh()
        metric_values = defaultdict(int)
        samples = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.model(inputs)
                batch_size = inputs.size(0)
                for name, metric in metrics.items():
                    metric_value = metric(outputs, targets)
                    if isinstance(metric_value, torch.Tensor):
                       metric_value = metric_value.item()
                    metric_values[name] += metric_value * batch_size
                samples += batch_size
                tq.update(1)
        tq.close()
        for name in metric_values:
            metric_values[name] /= samples
        print('Eval. Metrics {}'.format(str(dict(metric_values))))
        return metric_values

    def train_folds(self, folds, input_size, epochs, batch_size):
        val_metrics = []
        for i, fold in enumerate(folds):
            print('Start training fold: {}'.format(i))
            print('Will construct model')
            self.model = self.model_fn().cuda()
            self.optimizer = self.optimizer_fn(self.model.parameters(), lr=self.lr)
            print('Model construted')

            train_dataloader = DataLoader(CustomDataset(fold['train'], input_size, True),
                                          shuffle=True,
                                          batch_size=batch_size,
                                          num_workers=4)
            val_dataloader = DataLoader(CustomDataset(fold['val'], input_size, False),
                                        shuffle=False,
                                        batch_size=batch_size,
                                        num_workers=4)
            output_path = join(RESULT_PATH, '{}_fold'.format(i))
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            best_val_metrics = self._train_model(epochs, train_dataloader, val_dataloader, output_path)
            print('Fold {}. Best val metrics:{}'.format(i, best_val_metrics))
            val_metrics.append(best_val_metrics)
        print('Train results:')
        print(val_metrics)

