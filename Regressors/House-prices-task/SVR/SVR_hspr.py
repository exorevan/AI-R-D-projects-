import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, cross_validate,
                                     train_test_split)
from sklearn.svm import SVR

warnings.filterwarnings('ignore')


class DataWork:
    def load_data(self):
        train_data = pd.read_csv('SVR_train.csv', sep=',')
        train_answers = train_data.iloc[:, -1]
        train_data = train_data.iloc[:, 1:-1]

        test_data = pd.read_csv('SVR_test.csv', sep=',')
        test_answers = pd.read_csv('SVR_sample_submission.csv', sep=',')
        test_answers = test_answers.iloc[:, -1]
        test_data = test_data.iloc[:, 1:]

        return test_data, test_answers, train_data, train_answers

    def prepare_data(self):
        test_data, test_answers, train_data, train_answers = self.load_data()

        features = ['OverallQual', 'YearRemodAdd', 'Fireplaces', 'GarageCars', 'GrLivArea',
                    'ExterQual', 'TotalBsmtSF', 'KitchenQual', 'TotRmsAbvGrd', 'GarageArea']
        prpr = Preprocessing()
        cthnd = Categories_handler()

        train_X = train_data[features]
        train_X = cthnd.utilities_handler(train_X)
        train_X = cthnd.kitchenqua_handler(train_X)
        train_X = cthnd.pavecar_handler(train_X)
        train_X = cthnd.fence_handler(train_X)

        xmean = []
        for i in range(np.array(train_X).shape[1]):
            xmean.append(np.mean(np.array(train_X)[:, i]))

        xvar = []
        for i in range(np.array(train_X).shape[1]):
            xvar.append(np.var(np.array(train_X)[:, i]))

        train_X = prpr.preprocessing_x(train_X, xmean, xvar)
        ymean = np.mean(np.array(train_answers))
        yvar = np.var(np.array(train_answers))
        train_y = prpr.preprocessing_y(train_answers, ymean, yvar)

        val_X = test_data[features]
        val_X = cthnd.utilities_handler(val_X)
        val_X = cthnd.kitchenqua_handler(val_X)
        val_X = cthnd.pavecar_handler(val_X)
        val_X = cthnd.fence_handler(val_X)
        val_X = prpr.preprocessing_x(val_X, xmean, xvar)
        val_y = prpr.preprocessing_y(test_answers, ymean, yvar)

        return train_X, train_y, val_X, val_y


class Categories_handler:
    def utilities_handler(self, data):
        new_data = data.replace(
            {'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3})

        return new_data

    def kitchenqua_handler(self, data):
        new_data = data.replace(
            {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        return new_data

    def pavecar_handler(self, data):
        new_data = data.replace({'N': 0, 'P': 1, 'Y': 2})

        return new_data

    def fence_handler(self, data):
        new_data = data.replace(
            {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4})

        return new_data


class Preprocessing:
    def del_nans(self, data):
        nans = data.isnull().values.T
        nans_col = []

        for i in range(len(nans)):
            if True in nans[i]:
                nans_col.append(i)

        for i in nans_col:
            for j in range(len(data)):
                if math.isnan(data.iloc[j, i]):
                    data.iloc[j, i] = 0

    def normalization_x(self, data, xmean, xvar):
        for i in range(data.shape[1]):
            data[:, i] = (data[:, i] - xmean[i]) / xvar[i]

    def preprocessing_x(self, data, xmean, xvar):
        new_data = data

        self.del_nans(new_data)
        new_data = np.array(data, dtype=np.float64)
        self.normalization_x(new_data, xmean, xvar)

        return new_data

    def normalization_y(self, data, ymean, yvar):
        data = (data - np.mean(data)) / np.var(data)

    def preprocessing_y(self, data, ymean, yvar):
        new_data = np.array(data, dtype=np.float64)
        self.normalization_y(new_data, ymean, yvar)

        return new_data

    def postprocessing_y(self, data, ymean, yvar):
        new_data = data * yvar + ymean

        return new_data


class ML:
    def grid_search(self, kernel, C, gamma, train_X, train_y):
        svr = SVR()
        parameters = {'kernel':kernel, 'C':C, 'gamma':gamma}
        clf = GridSearchCV(svr, parameters, cv=50, verbose=3)
        clf.fit(train_X, train_y)

        return clf.best_params_

    def cross_validation(self, regressor):
        scores = cross_validate(
            estimator = regressor,
            X = train_X,
            y = train_y,
            scoring = 'r2',
            cv = 3,
            return_estimator = True
        )

        return scores


class Plots:
    def correlations(self):
        window_plot_size = (8, 8)
        fig1 = plt.figure(figsize=window_plot_size)
        prpr = Preprocessing()
        cthnd = Categories_handler()
        features = ['LotFrontage', 'LotArea', 'Utilities', 'OverallQual', 'YearRemodAdd', 'BsmtCond', 'HeatingQC', 'Fireplaces', 'ExterCond', 'GarageCars', 'GrLivArea', 'ExterQual', 'MasVnrArea', 'TotalBsmtSF',
                    'ExterCond', 'OverallCond', 'KitchenQual', 'YrSold', 'TotRmsAbvGrd', 'GarageArea', 'GarageCond', 'GarageQual', 'PavedDrive', 'WoodDeckSF', 'PoolArea', 'PoolQC', 'Fence', 'MiscVal']  # , 'SalePrice']

        X_corr = pd.read_csv('SVR_train.csv', sep=',')[features]
        X_corr = cthnd.utilities_handler(X_corr)
        X_corr = cthnd.kitchenqua_handler(X_corr)
        X_corr = cthnd.pavecar_handler(X_corr)
        X_corr = cthnd.fence_handler(X_corr)
        prpr.del_nans(X_corr)

        hm = sns.heatmap(X_corr.corr(), cmap="YlGnBu")
        plt.show()

    def result(self, val_predictions, val_y):
        window_plot_size = (8, 8)
        fig = plt.figure(figsize=window_plot_size)

        plt.style.use('dark_background')
        plt.grid(c='w', alpha = .2, linestyle = '-')

        plt.plot(list(range(1459)), val_predictions, c='blue')
        plt.plot(list(range(1459)), val_y, c='red')
        plt.xlabel("red - true, blue - predictions")
        plt.show()


plots = Plots()
plots.correlations()

data_work = DataWork()
train_X, train_y, val_X, val_y = data_work.prepare_data()

ml = ML()
best_params = ml.grid_search(['rbf', 'linear'], [0.1, 1, 10, 100], [0.1, 0.01, 0.001, 0.0001], train_X, train_y)
print(f"\nbest params are {best_params}\n")

regressor_example = SVR(kernel='linear', C=100, gamma=0.1)
cross_validation_results = ml.cross_validation(regressor_example)
print(f"\nbest models are {cross_validation_results}\n")

regressor_from_cv = cross_validation_results['estimator'][0]
validation_predictions = regressor_from_cv.predict(val_X)
print(f"\nr2_score is {r2_score(val_y, validation_predictions)}\n")

plots.result(validation_predictions / 3, val_y)