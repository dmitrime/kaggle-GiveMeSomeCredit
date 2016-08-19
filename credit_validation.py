import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc


def predict_column(train, target, est=500):
    # all columns except col
    cols = list(set(train.columns)-set([target]))
    # predict with GBM
    gbm = GradientBoostingClassifier(n_estimators=est).fit(
        train[cols], train[target])
    # return probabilities
    return gbm.predict_proba(train[cols])[:,1]


def add_features(data):
    # remove unnecessary first column
    data.drop(data.columns[0], axis=1, inplace=True)

    # NaN dependents, no dependends and the rest columns
    data['NaNDependents'] = pd.isnull(data['NumberOfDependents'])
    data['NoDependents'] = data['NumberOfDependents'] == 0
    data['NumberOfDependents'].fillna(0, inplace=True)

    # NaN income, no income, low income, and log of income
    data['NaNMonthlyIncome'] = data['MonthlyIncome'] == np.nan
    data['NoMonthlyIncome'] = data['MonthlyIncome'] == 0
    data['LowMonthlyIncome'] = data['MonthlyIncome'] < 100
    data['LogMonthlyIncome'] = np.log(data['MonthlyIncome'])
    data.loc[~np.isfinite(data['LogMonthlyIncome']), 'LogMonthlyIncome'] = 0
    data.loc[pd.isnull(data['LogMonthlyIncome']), 'LogMonthlyIncome'] = 0

    # log of income per person
    data['LogIncomePerPerson'] = data['LogMonthlyIncome'] / data['NumberOfDependents']
    data.loc[~np.isfinite(data['LogIncomePerPerson']), 'LogIncomePerPerson'] = 0

    # log of RevolvingUtilizationOfUnsecuredLines
    data['LogRevolvingUtilizationOfUnsecuredLines'] = np.log(data['RevolvingUtilizationOfUnsecuredLines'])
    data.loc[~np.isfinite(data['LogRevolvingUtilizationOfUnsecuredLines']), 'LogRevolvingUtilizationOfUnsecuredLines'] = 0

    # age related
    data['YoungAge'] = data['age'] < 21
    data['OldAge'] = data['age'] > 65
    data['ZeroAge'] = data['age'] == 0
    data['LogAge'] = np.log(data['age'])
    data.loc[data['ZeroAge'] == True, 'LogAge'] = 0

    ## restore debt and take log
    data['LogDebt'] = np.log(data['DebtRatio'] * data['LogMonthlyIncome'])
    data.loc[~np.isfinite(data['LogDebt']), 'LogDebt'] = 0
    data['LogDebtPerPerson'] = data['LogDebt'] / data['NumberOfDependents']
    data.loc[~np.isfinite(data['LogDebtPerPerson']), 'LogDebtPerPerson'] = 0

    # drop unneeded orginal columns
    data.drop(['MonthlyIncome', 'age', 'RevolvingUtilizationOfUnsecuredLines'], axis=1, inplace=True)

    # binary columns late or not
    data['NoPastDue30-59'] = data['NumberOfTime30-59DaysPastDueNotWorse'] == 0
    data['NoPastDue60-89'] = data['NumberOfTime60-89DaysPastDueNotWorse'] == 0
    data['NoLateOver90'] = data['NumberOfTimes90DaysLate'] == 0

    # predict probability of being late over 90 days and use it as a feature
    data['PredictedLateOver90'] = predict_column(data, 'NoLateOver90')

    print(data.columns)


def fit_classifiers(train, test, validate=False):
    # all columns except the target (X columns).
    cols = [c for c in train.columns if c not in ['SeriousDlqin2yrs']]

    classifiers = dict()
    # fit random forest to the train data
    classifiers['rf'] = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    # fit gbm to the train data
    classifiers['gbm1'] = GradientBoostingClassifier(n_estimators=1000)
    # fit gbm with adaboost to the train data
    classifiers['gbm2'] = GradientBoostingClassifier(n_estimators=1000, loss='exponential')

    # stack the classifiers by building a new dataset where 
    # each classifier's predictions correspond to a feature column.
    stacked_train = np.zeros((train.shape[0], len(classifiers)))
    stacked_test = np.zeros((test.shape[0], len(classifiers)))

    # make np arrays from pandas dataframe
    X, y = train.as_matrix(columns=cols), train['SeriousDlqin2yrs'].as_matrix()

    N_folds = 5
    folds = list(StratifiedKFold(y, N_folds))
    for clf_idx, clf in enumerate(classifiers.values()):
        stacked_test_clf = np.zeros((test.shape[0], N_folds))
        for fold_idx, (fold_train, fold_test) in enumerate(folds):
            clf.fit(X[fold_train], y[fold_train])
            # clf_idx-th classifier's predictions on the current fold
            stacked_train[fold_test, clf_idx] = clf.predict_proba(X[fold_test])[:,1]
            stacked_test_clf[:, fold_idx] = clf.predict_proba(test[cols])[:,1]
        stacked_test[:, clf_idx] = stacked_test_clf.mean(axis=1)

    # fit logistic regression to the stacked predictions
    logistic = LogisticRegression().fit(stacked_train, train['SeriousDlqin2yrs'])
    probs = logistic.predict_proba(stacked_test)[:,1]
    if validate:
        # compute AUC from the ROC curve
        fpr, tpr, _ = roc_curve(test['SeriousDlqin2yrs'], probs)
        print('stacked: {}'.format(auc(fpr, tpr)))

    return probs

if __name__ == '__main__':
    # set the random seed
    np.random.seed(0)
    # read in the data into pandas dataframe
    data = pd.read_csv('cs-training.csv')
    # add new features based on the old ones
    add_features(data)
    # split into train and validat
    train, validate = train_test_split(data, test_size=0.25)

    result = fit_classifiers(train, validate, validate=True)
    # average probabilities for all classifiers
    fpr, tpr, _ = roc_curve(validate['SeriousDlqin2yrs'], result)
    print auc(fpr, tpr)
