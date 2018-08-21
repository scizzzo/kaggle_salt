import torch
import os
import numpy as np

from tqdm import tqdm
from os.path import join
from torch.utils.data.dataloader import DataLoader
from dataset import SaltInferenceDataset as CustomDataset


PREDICTIONS_PATH = './results'


class PyInferer:
    def __init__(self, data, model, model_folds):
        self.data = data
        self.model = model.cuda()
        self.model_folds = model_folds
        self.fold_results = None

    @staticmethod
    def flip_tensor_lr(batch):
        columns = batch.data.size()[-1]
        return batch.index_select(3, torch.LongTensor(list(reversed(range(columns)))).cuda())

    @staticmethod
    def flip_tensor_ud(batch):
        rows = batch.data.size()[-2]
        return batch.index_select(2, torch.LongTensor(list(reversed(range(rows)))).cuda())

    def load_fold_weights(self, weights_path):
        self.model.load_state_dict(torch.load(join(weights_path, 'bestModel.pt')))

    def make_prediction(self, input, tta):
        pred1 = self.model(input)
        if 'h' in tta:
            pred2 = self.flip_tensor_lr(self.model(self.flip_tensor_lr(input)))
            masks = [pred1, pred2]
            if 'v' in tta:
                pred3 = self.flip_tensor_ud(self.model(self.flip_tensor_ud(input)))
                pred4 = self.flip_tensor_ud(self.flip_tensor_lr(self.model(self.flip_tensor_ud(self.flip_tensor_lr(input)))))
                masks.extend([pred3, pred4])
            new_mask = torch.mean(torch.stack(masks, 0), 0)
            return new_mask
        return pred1

    def process_output(self, output, meta, fold):
        output = output.data.cpu().numpy()
        out_fp = meta['img_name'][0].split('.')[0] + '.npy'
        np.save(join(PREDICTIONS_PATH, fold, out_fp), output)

    def make_inference(self, input_size, tta=''):
        for i, fold in enumerate(os.listdir(self.model_folds)):
            if not os.path.exists(join(PREDICTIONS_PATH, fold)):
                os.makedirs(join(PREDICTIONS_PATH, fold))
            print('Make predictions {} fold'.format(i))
            fold_path = join(self.model_folds, fold)
            self.load_fold_weights(fold_path)
            self.model.eval()
            dataloader = DataLoader(CustomDataset(self.data, input_size),
                                    shuffle=False,
                                    batch_size=1)
            for input, meta in tqdm(dataloader):
                input = input.cuda()
                output = self.make_prediction(input, tta)
                self.process_output(output, meta, fold)
