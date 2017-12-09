"""
Call code from everywhere, read data, initialize model, train model and make
sure training is doing something meaningful.
"""
from __future__ import unicode_literals
from __future__ import print_function
import argparse, os, sys
import math, time, codecs

import utils
import model_utils as mu
import sentslstm as slstm
import trainer
import evaluate

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def train_model(int_mapped_path, embedding_path, model_path,
                result_path, use_toy=True):
    """
    Read the int training and dev data, initialize and train the model.
    :return:
    """
    # np.random.seed(4186)
    # torch.manual_seed(4186)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(4186)

    # Load training and dev data.
    X_train, y_train, X_dev, y_dev, X_test, y_test, word2idx = \
        utils.load_intmapped_data(int_mapped_path, use_toy)
    data = {'X_train': X_train,
            'y_train': y_train,
            'X_dev': X_dev,
            'y_dev': y_dev}

    # Initialize model.
    model = slstm.SentsLSTM(word2idx, embedding_path, num_classes=6,
                            num_layers=1, embedding_dim=200, hidden_dim=50,
                            dropout=0.3)
    # Initialize the trainer.
    lstmtrainer = trainer.LSTMTrainer(
        model=model, data=data, batcher=mu.Batcher, batch_size=16,
        update_rule='adam', num_epochs=5, learning_rate=0.001, check_every=10,
        print_every=1, model_path=model_path)

    # Train and save the best model to model_path.
    lstmtrainer.train()
    # Make predictions and evaluate.
    data['X_test'] = X_test
    data['y_test'] = y_test
    everything = evaluate.make_predictions(data=data, batcher=mu.Batcher,
                                           model=model,
                                           model_name=lstmtrainer.model_name,
                                           result_path=result_path)
    y_test_true, y_test_preds, y_dev_true, y_dev_preds, \
        y_train_true, y_train_preds = everything
    for split, true, pred in [('test', y_test_true, y_test_preds),
                              ('dev', y_dev_true, y_dev_preds),
                              ('train', y_train_true, y_train_preds)]:
        print(split)
        evaluate.evaluate_preds(true, pred)


def main():
    pass

if __name__ == '__main__':
    #main()
    train_model(int_mapped_path=sys.argv[1], embedding_path=sys.argv[2],
                model_path=sys.argv[3], result_path=sys.argv[4])
