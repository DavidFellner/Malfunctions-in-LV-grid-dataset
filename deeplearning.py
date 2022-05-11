from util import load_model, export_model, save_model, load_data, plot_samples, choose_best, save_result, create_dataset
import logging, sys, os
import torch
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
import plotting

class Deeplearning:

    def __init__(self, config, learning_config, run):

        self.config = config
        self.learning_config = learning_config
        self.run = run

        logger, device = Deeplearning.init(self)

        # Load data
        logger.info("Loading Data ...")
        if self.learning_config["mode"] == 'train':
            self.train_loader = load_data('train')
        self.test_loader = load_data('test')
        logger.info(f"Loaded data.")

        # dataset, X, y = load_dataset()
        if self.learning_config["plot samples"] and self.learning_config["mode"] == 'train':
            for i, (X, y, X_raw) in enumerate(self.train_loader):
                plot_samples(X_raw, y, X)
                break

        ''' 
        deprecated by use of data loaders!
        
        if self.learning_config['baseline']:
            Deeplearning.baseline(self, X, y)
            '''

        print('X data with zero mean per sample and scaled between -1 and 1 based on training samples used')

        self.path = os.path.join(self.config.models_folder, self.learning_config['classifier'])
        self.model, self.epoch, self.loss = load_model(self.learning_config, run)

    def init(self):
        level = 'INFO'
        self.logger = logging.getLogger('main')
        self.logger.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(ch)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using {device}.")

        return self.logger, device

    def baseline(self, X, y):

        clf_baseline = SGDClassifier()
        scores = cross_validate(clf_baseline, X, y, scoring=self.learning_config["metrics"], cv=10, n_jobs=1)
        print("########## Linear Baseline: 10-fold Cross-validation ##########")
        for metric in self.learning_config["cross_val_metrics"]:
            print("%s: %0.2f (+/- %0.2f)" % (metric, scores[metric].mean(), scores[metric].std() * 2))

        return

    def training_or_testing(self, k):

        if not self.learning_config["cross_validation"]:

            # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=learning_config['train test split'])
            # X_train, X_test = model.preprocess(X_train, X_test)

            print("\n########## Training ##########")
            if self.learning_config["do grid search"]:
                runs = len(self.learning_config["grid search"][1])
            else:
                runs = 1
            for i in range(runs):
                if self.learning_config["mode"] == 'train':
                    self.logger.info("Training classifier ..")
                    if self.learning_config["do grid search"]: self.logger.info(
                        "Value of {}: {}".format(self.learning_config["grid search"][0], self.learning_config["grid search"][1][i]))
                    if self.learning_config["do hyperparameter sensitivity analysis"]: self.logger.info(
                        "Value of {}: {}".format(self.learning_config["hyperparameter tuning"][0], self.learning_config["hyperparameter tuning"][1][k]))

                    clfs, losses, lrs = self.model.fit(self.train_loader, self.test_loader,
                                                  early_stopping=self.learning_config['early stopping'],
                                                  control_lr=self.learning_config['LR adjustment'], prev_epoch=self.epoch,
                                                  prev_loss=self.loss, grid_search_parameter = self.learning_config["grid search"][1][i])

                    self.logger.info("Training finished!")
                    self.logger.info('Finished Training')
                    plotting.plot_2D([losses, [i[1] for i in clfs]], labels=['Training loss', 'Validation loss'],
                                     title='Losses after each epoch', x_label='Epoch',
                                     y_label='Loss')  # plot training loss for each epoch
                    plotting.plot_2D(lrs, labels='learning rate', title='Learning rate for each epoch', x_label='Epoch',
                                     y_label='Learning rate')
                    clf, epoch = choose_best(clfs)
                    self.model.state_dict = clf[0]  # pick weights of best model found

                y_pred, outputs, y_test = self.model.predict(test_loader=self.test_loader)
                if self.learning_config["mode"] == 'eval':
                    clf = self.model
                    score = self.model.score(y_test, y_pred)
                    print("\n########## Metrics ##########")
                    print(
                        "Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}".format(score[0],
                                                                                         score[1][
                                                                                             0],
                                                                                         score[1][
                                                                                             1],
                                                                                         score[1][
                                                                                             2], ))
                else:
                    score = self.model.score(y_test, y_pred) + [clf[1]]
                    print("\n########## Metrics ##########")
                    print(
                        "Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}\nLowest validation loss: {4}".format(score[0],
                                                                                                                      score[1][
                                                                                                                          0],
                                                                                                                      score[1][
                                                                                                                          1],
                                                                                                                      score[1][
                                                                                                                          2],
                                                                                                                      score[2]))

                if self.learning_config["training time sweep"]:
                    epoch = 0
                    for clf in clfs:
                        epoch_model = self.model
                        epoch_model.state_dict = clf[0]
                        y_pred, outputs, y_test = epoch_model.predict(test_loader=self.test_loader)
                        score = epoch_model.score(y_test, y_pred) + [clf[1]]
                        if self.learning_config["save_result"]:
                            save_result(score, i, k, epoch)
                        epoch += 1
                    plotting.plot_time_sweep()



                if self.learning_config["save_model"] and self.learning_config["mode"] == 'train':
                    save_model(self.model, epoch, clf[1], i, k)

                if self.learning_config["save_result"]:
                    save_result(score, i, k, self.learning_config["number of epochs"])

                if self.learning_config["export_model"]:
                    export_model(self.model, self.learning_config, i, k)

            if self.learning_config['do grid search']:
                plotting.plot_grid_search()

        """
        deprecated by use of data loaders!
        
        if self.learning_config["cross_validation"]:
            print("\n########## k-fold Cross-validation ##########")
            model, scores = self.cross_val(X, y, model)
            print("########## Metrics ##########")
            for score in scores:
                print("%s: %0.2f (+/- %0.2f)" % (score, np.array(scores[score]).mean(), np.array(scores[score]).std() * 2))

            if learning_config["save_model"] and learning_config["mode"] == 'train':
                save_model(model, epoch, clf[1], learning_config)

            if learning_config["save_result"]:
                save_result(scores, learning_config)

            if learning_config["export_model"]:
                export_model(model, learning_config)"""

        """def cross_val(self, X, y, model):

            kf = KFold(n_splits=self.learning_config['k folds'])
            best_clfs = []
            scores = []

            for train_index, test_index in kf.split(X):
                print('Split #%d' % (len(scores) + 1))
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = list(np.array(y)[train_index]), list(np.array(y)[test_index])

                X_train, X_test = model.preprocess(X_train, X_test)

                clfs, losses, lrs = model.fit(X_train, y_train, X_test, y_test,
                                              early_stopping=self.learning_config['early stopping'],
                                              control_lr=self.learning_config['LR adjustment'])
                best_model = choose_best(clfs)
                best_clfs.append(best_model)
                model.state_dict = best_model[0]
                y_pred, outputs = model.predict(X_test)
                scores.append(model.score(y_test, y_pred) + [best_model[1]])

            very_best_model = choose_best(best_clfs)
            model.state_dict = very_best_model[0]

            scores_dict = {'Accuracy': [i[0] for i in scores], 'Precision': [i[1][0] for i in scores],
                           'Recall': [i[1][1] for i in scores], 'FScore': [i[1][2] for i in scores],
                           'Lowest validation loss': [i[2] for i in scores]}

            return model, scores_dict"""