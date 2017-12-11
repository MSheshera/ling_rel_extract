"""
Call code from everywhere, read data, initialize model, train model and make
sure training is doing something meaningful.
"""
from __future__ import unicode_literals
from __future__ import print_function
import argparse, os, sys
import math, time, codecs, pprint

import utils
import model_utils as mu
import sentslstm as slstm
import trainer
import evaluate

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def train_model(int_mapped_path, embedding_path, run_path,
                model_hparams, train_hparams, use_toy=True):
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
    # Unpack args.
    edim, hdim, dropp = model_hparams['edim'], model_hparams['hdim'], model_hparams['dropp']
    bsize, epochs, lr = train_hparams['bsize'], train_hparams['epochs'], train_hparams['lr']
    print('Model hyperparams:')
    pprint.pprint(model_hparams)
    print('Train hyperparams:')
    pprint.pprint(train_hparams)

    # Initialize model.
    model = slstm.SentsLSTM(word2idx, embedding_path, num_classes=6,
                            num_layers=1, embedding_dim=edim, hidden_dim=hdim,
                            dropout=dropp)
    # Initialize the trainer.
    lstmtrainer = trainer.LSTMTrainer(
        model=model, data=data, batcher=mu.Batcher, batch_size=bsize,
        update_rule='adam', num_epochs=epochs, learning_rate=lr, check_every=10,
        print_every=10, model_path=run_path)

    # Train and save the best model to model_path.
    lstmtrainer.train()
    # Plot training time stats.
    utils.plot_train_hist(lstmtrainer.loss_history, lstmtrainer.checked_iters,
                          fig_path=run_path, ylabel='Batch loss')
    utils.plot_train_hist(lstmtrainer.dev_score_history,
                          lstmtrainer.checked_iters,
                          fig_path=run_path, ylabel='Dev F1')

    # Make predictions and evaluate on all the data.
    data['X_test'] = X_test
    data['y_test'] = y_test
    everything = evaluate.make_predictions(data=data, batcher=mu.Batcher,
                                           model=model, result_path=run_path,
                                           batch_size=128, write_preds=True)
    y_test_true, y_test_preds, y_dev_true, y_dev_preds, \
        y_train_true, y_train_preds = everything
    for split, true, pred in [('test', y_test_true, y_test_preds),
                              ('dev', y_dev_true, y_dev_preds),
                              ('train', y_train_true, y_train_preds)]:
        print(split)
        evaluate.evaluate_preds(true, pred)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=u'subcommand',
                                       help=u'The action to perform.')
    # Train the model.
    train_args = subparsers.add_parser(u'train_model')
    # Where to get what.
    train_args.add_argument(u'--int_mapped_path', required=True,
                            help=u'Path to the int mapped dataset.')
    train_args.add_argument(u'--embedding_path', required=True,
                            help=u'Path to the glove embeddings directory.')
    train_args.add_argument(u'--run_path', required=True,
                            help=u'Path to directory to save all run items to.')
    # Model hyper-parameters.
    train_args.add_argument(u'--edim', required=True, type=int,
                            choices=[200], help=u'Embedding dimension.')
    train_args.add_argument(u'--hdim', required=True, type=int,
                            choices=range(1, 201, 1),
                            help=u'LSTM hidden dimension.')
    train_args.add_argument(u'--dropp', required=True, type=float,
                            choices=[0.3], help=u'Dropout probability.')
    # Training hyper-parameters.
    train_args.add_argument(u'--bsize', required=True, type=int,
                            choices=[8, 16, 32, 64, 128, 256],
                            help=u'Batch size.')
    train_args.add_argument(u'--epochs', required=True, type=int,
                            choices=range(1, 11, 1),
                            help=u'Number of training epochs.')
    train_args.add_argument(u'--lr', required=True, type=float,
                            choices=[0.1, 0.01, 0.001, 0.0001, 0.00001],
                            help=u'Learning rate.')
    cl_args = parser.parse_args()

    if cl_args.subcommand == 'train_model':
        model_hparams, train_hparams = {}, {}
        model_hparams['edim'], model_hparams['hdim'], model_hparams['dropp'] = \
            cl_args.edim, cl_args.hdim, cl_args.dropp
        train_hparams['bsize'], train_hparams['epochs'], train_hparams['lr'] = \
            cl_args.bsize, cl_args.epochs, cl_args.lr
        train_model(int_mapped_path=cl_args.int_mapped_path,
                    embedding_path=cl_args.embedding_path,
                    run_path=cl_args.run_path,
                    model_hparams=model_hparams, train_hparams=train_hparams,
                    use_toy=False)

if __name__ == '__main__':
    main()
