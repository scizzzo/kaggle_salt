import numpy as np
import argparse
import os

from functools import partial
from trainer import PyTrainer
from metrics import IoU, BCEDICELoss
from unet import construct_unet
from torch.optim import Adam
from inference import PyInferer


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--output', type=str)
parser.add_argument('--pretrained', dest='pretrained', action='store_true')
parser.set_defaults(pretrained=False)
parser.add_argument('--input_size', type=str, default=128)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=int, default=0.001)
parser.add_argument('--batch_size', type=int, default=24)


def extract_folds(dir_path, train_files):
    folds = []
    for file in sorted(train_files, key=lambda x: int(x.split('_')[0])):
        fold = {}
        fold['train'] = os.path.join(dir_path, file)
        fold['val'] = os.path.join(dir_path, file.split('_')[0] + '_val.npy')
        folds.append(fold)
    return folds


def processed_folds(folds):
    pr_folds = []
    for fold in folds:
        pr_fold = {}
        train_data = np.load(fold['train'])
        val_data = np.load(fold['val'])
        pr_fold['train'] = train_data
        pr_fold['val'] = val_data
        pr_folds.append(pr_fold)
    print('Folds has been readed.')
    return pr_folds


def main(args):
    if args.mode == 'train':
        files = os.listdir(args.data)
        train_files = [file for file in files if 'train' in file]
        folds = extract_folds(args.data, train_files)
        folds = processed_folds(folds)

        metrics = {'iou': IoU()}
        loss_weights = {
            'bce': 0.5,
            'dice': 0.5
        }
        criterion = BCEDICELoss(loss_weights)

        model_fn = partial(construct_unet, n_cls=1, pretrained=args.pretrained)

        trainer = PyTrainer(model_fn, Adam, args.lr, criterion, metrics)

        trainer.train_folds(folds, args.input_size, args.epochs, args.batch_size)

    elif args.mode == 'inference':
        inference_data = np.load(args.data)

        print('Will construct model.')
        model = construct_unet(n_cls=1, pretrained=False)
        print('Model constructed')

        inferer = PyInferer(inference_data, model, './models')
        inferer.make_inference(input_size=args.input_size)


if __name__ == '__main__':
    main(parser.parse_args())