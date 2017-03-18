# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:39:37 2016

@author: lchlebak
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix

#Turn file into dataframe.
df = pd.DataFrame.from_csv('/home/lchlebak/Downloads/training_and_test_data.csv')

#Drop all of the doctor-dependent columns EXCEPT for Doctor and Final Triage.
df = df.drop(['Doctor Dependent 1', 'Doctor Dependent 2', 'Doctor Dependent 3'],1)

#Drop the final doctor-dependent columns, Final Triage and Doctor, to form a new 
#dataframe.
dn = df.drop(['Final Triage','Doctor'],1)

#Run k-Nearest Neighbors algorithm.

#Choose training and testing data sets. Here Doctor 0 is the "consensus" doctor.
test = df.groupby(['Doctor']).get_group(0)
train = df[df['Doctor'] != 0]
test = test.drop(['Doctor'],1)
train = train.drop(['Doctor'],1)

#Pick out the X_i parameters we want to use in order to predict Y. Notice we
#don't use any of the doctor-dependent parameters.
attribute_list = list(dn.columns.values)
predictors = train[attribute_list].copy()
X = predictors.as_matrix()
predictors_test = test[attribute_list].copy()
X_test = predictors_test.as_matrix()

#Run Gridsearch and cross validation.
parameters = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
opt = GridSearchCV(KNeighborsClassifier(),parameters,cv=5)
opt.fit(predictors.as_matrix(),train['Final Triage'].as_matrix())

#Final results.
best_params = opt.best_params_
clf = opt.best_estimator_
scr = clf.score(X_test, test['Final Triage'].as_matrix())

#Look at the confusion matrix to see how we did.
confusion = confusion_matrix(test['Final Triage'].as_matrix(),
                             clf.predict(X_test))
    