import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc

from credit_validation import add_features, fit_classifiers
#from credit_validation_old import add_features, fit_classifiers

def output(probs, fname):
    with open(fname, 'w') as f:
        f.write('Id,Probability\n')
        for idx, pr in zip(xrange(1, len(probs)+1), probs):
            f.write('{},{:.9f}\n'.format(idx, pr))

if __name__ == '__main__':
    # read in the data into pandas dataframe
    data  = pd.read_csv('cs-training.csv')
    test  = pd.read_csv('cs-test.csv')
    # remove unnecessary y-column
    test.drop('SeriousDlqin2yrs', axis=1, inplace=True)
    # add new features
    add_features(data)
    add_features(test)

    # predict the probabilities on the test data
    probs = fit_classifiers(data, test)
    # write results to file
    output(probs, 'submission.csv')
