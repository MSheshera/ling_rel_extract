from __future__ import unicode_literals
from __future__ import print_function
import argparse, os, sys
import math, time, json, codecs, pickle

import numpy as np
import torch

import model_utils as mu
import sentslstm as slstm
import train

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def run_model(int_mapped_path, embedding_path, checkpoint_path=None, use_toy=True):
    """

    :return:
    """
    # np.random.seed(4186)
    # torch.manual_seed(4186)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(4186)

    # Load training and dev data.
    if use_toy:
        set_str = 'small'
    else:
        set_str = 'full'
    train_path = os.path.join(int_mapped_path, 'train-im-{:s}.json'.format(set_str))
    with open(train_path, 'r') as fp:
        X_train, y_train = json.load(fp)  # l of l of l, l
    dev_path = os.path.join(int_mapped_path, 'dev-im-{:s}.json'.format(set_str))
    with open(dev_path, 'r') as fp:
        X_dev, y_dev = json.load(fp)
    map_path = os.path.join(int_mapped_path, 'word2idx-{:s}.json'.format(set_str))
    with open(map_path, 'r') as fp:
        word2idx = json.load(fp)
    data = {'X_train': X_train, 'y_train': y_train,
            'X_dev': X_dev, 'y_dev': y_dev}

    testX, testy = X_dev[:10], y_dev[:10]
    test_batcher = mu.Batcher(testX, testy)
    X, y = test_batcher.full_batch()
    # Initialize model.
    model = slstm.SentsLSTM(word2idx, embedding_path, num_classes=6,
                            max_batch_size=64, num_layers=1,
                            embedding_dim=200, hidden_dim=50, dropout=0.3,
                            cuda=torch.cuda.is_available())
    print(model)
    print(model.objective(X, y))


    # # train the model
    # trainer = train.LSTMTrainer(model=model, data=data, update_rule='adam',
    #                             num_epochs=5, learning_rate=0.001,
    #                             print_every=10, max_batch_size=64,
    #                             checkpoint_path=checkpoint_path)
    # trainer.train()
    # loss_hist, dev_hist = trainer.loss_history, trainer.dev_score_history


def main():
    parser = argparse.ArgumentParser(description='PyTorch Vanilla LSTM binary')
    parser.add_argument('--toy', action='store_true',
                        help='if you want to use the toy data instead of real data')
    parser.add_argument('--save', action='store_true',
                    help='want to save the models')
    # hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=50,
                    help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate for optimizer')
    parser.add_argument('--dropout', type=float, default=0.5,
                    help='droput')
    parser.add_argument('--embed_dim', type=int, default=50,
                    help='embedding dimension')
    parser.add_argument('--hid_dim', type=int, default=50,
                    help='hidden dimension')
    # path to save to
    parser.add_argument('--fname', type=str, default='stat',
                    help='filename to save the statistics to')
    args = parser.parse_args()

if __name__ == '__main__':
    #main()
    run_model(int_mapped_path=sys.argv[2], embedding_path=sys.argv[1],
              checkpoint_path=None, use_toy=True)
