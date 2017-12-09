"""
Train the passed model given the data and the batcher and save the best to disk.
"""
from __future__ import print_function
import sys, os
import time

from sklearn import metrics
import torch
import torch.optim as optim


class LSTMTrainer():
    def __init__(self, model, data, batcher, batch_size, update_rule, num_epochs, learning_rate,
                 check_every, print_every, model_path, verbose=True):
        """

        :param model: pytorch model.
        :param data: dict{'X_train':, 'y_train':, 'X_dev':, 'y_dev':}
        :param batcher: a model_utils.Batcher class.
        :param batch_size: int; number of docs to consider in a batch.
        :param update_rule: string;
        :param num_epochs: int; number of passes through the training data.
        :param learning_rate: float;
        :param check_every: int; check model dev_f1 check_every iterations.
        :param print_every: int; print training numbers print_every iterations.
        :param model_path: string; directory to which model should get saved.
        :param verbose: boolean;
        """
        # Model, batcher and the data.
        self.model = model
        self.batcher = batcher
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_dev = data['X_dev']
        self.y_dev = data['y_dev']

        # Book keeping
        self.verbose = verbose
        self.print_every = print_every
        self.num_train = len(self.X_train)
        self.num_dev = len(self.X_dev)
        self.batch_size = batch_size
        self.model_path = model_path  # Save predictions, model and checkpoints.
        self.model_name = None

        # Optimizer args.
        self.update_rule = update_rule
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.check_every = check_every

        # Initialize optimizer.
        if self.update_rule == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.learning_rate)
        else:
            sys.stderr.write('Unknown update rule.\n')
            sys.exit(1)

        # Train statistics.
        self.loss_history = []
        self.dev_score_history = []

    def train(self):
        """
        Make num_epoch passes throught the training set and train the model.
        :return:
        """
        # Pick the model with the least loss.
        best_dev_f1 = 0.0
        best_params = self.model.state_dict()
        best_epoch, best_iter = 0, 0
        # Initialize the dev batcher and get padded and sorted dev data.
        dev_batcher = self.batcher(full_X=self.X_dev, full_y=self.y_dev,
                                   shuffle=False)
        dev_X, dev_y = dev_batcher.full_batch()
        dev_y = dev_y.numpy()

        train_start = time.time()
        print('num_train: {:d}; num_dev: {:d}'.format(self.num_train,
                                                      self.num_dev))
        print('Training {:d} epochs'.format(self.num_epochs))
        for epoch in xrange(self.num_epochs):
            # Initialize batcher. Shuffle one time before the start of every
            # epoch.
            epoch_batcher = self.batcher(full_X=self.X_train, full_y=self.y_train,
                                         batch_size=self.batch_size, shuffle=True)
            iters_for_epoch = epoch_batcher.num_batches
            # Get the next padded and sorted training batch.
            iters_start = time.time()
            for iter, (batch_X, batch_y) in enumerate(epoch_batcher.next_batch()):
                # Clear all gradient buffers.
                self.optimizer.zero_grad()
                # Compute objective.
                objective = self.model.objective(batch_X, batch_y)
                # Gradients wrt the parameters.
                objective.backward()
                # Step in the direction of the gradient.
                self.optimizer.step()
                loss = float(objective.data.numpy())
                if self.verbose and iter % self.print_every == 0:
                    print('Epoch: {:d}; Iteration: {:d}/{:d}; Loss: {:.4f}'.format(
                        epoch, iter, iters_for_epoch, loss))
                self.loss_history.append(loss)
                # Check every blah iterations how you're doing on the dev set.
                if iter % self.check_every == 0:
                    dev_preds = self.model.predict(batch_X=dev_X)
                    # Find f1 weighted by support of class.
                    dev_f1 = metrics.f1_score(dev_y, dev_preds,
                                              average='weighted')
                    self.dev_score_history.append(dev_f1)
                    if dev_f1 > best_dev_f1:
                        best_dev_f1 = dev_f1
                        best_params = self.model.state_dict()
                        best_epoch = epoch
                        best_iter = iter
                        if self.verbose:
                            print('Current best model; Epoch {:d}; Iteration '
                                  '{:d}; Dev F1: {:.4f}'.format(epoch, iter,
                                                                best_dev_f1))
            epoch_time = time.time()-iters_start
            print('Epoch {:d} time: {:.4f}s'.format(epoch, epoch_time))
            print()

        # Update model parameters to be best params.
        print('Best model; Epoch {:d}; Iteration {:d}; Dev F1: {:.4f}'.
              format(best_epoch, best_iter, best_dev_f1))
        self.model.load_state_dict(best_params)
        train_time = time.time()-train_start
        print('Training time: {:.4f}s'.format(train_time))

        # Save the learnt model.
        self.model_name = 'slstm_best_{:s}'.format(time.strftime('%m_%d-%H_%M_%S'))
        model_file = os.path.join(self.model_path, self.model_name)
        # TODO: Look into this save restore stuff and fix it. --high-pri.
        # https://stackoverflow.com/a/43819235/3262406
        torch.save(self.model.state_dict(), model_file)
        print('Wrote: {:s}'.format(model_file))
