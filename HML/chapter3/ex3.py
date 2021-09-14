import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score,\
        f1_score, precision_recall_curve

train_data = pd.read_csv("./titanic/train.csv")
x_test = pd.read_csv("./titanic/test.csv")
y_test = pd.read_csv("./titanic/gender_submission.csv")
gender_submission = pd.read_csv("./titanic/gender_submission.csv")

train_data.describe()

# Find out where the empty/null/nan values are
print("null test:")
for c in train_data.columns:
    if(any(train_data[c].isnull())):
        print(c + "is null sometimes")
print("nan test:")
for c in train_data.columns:
    if(any(train_data[c].isna())):
        print(c + "is nan sometimes")
# age can be null -> atm replace by mean age of 30 ??
# cabin can null -> value is dropped anyway, so no problem right now.
# embarked can be null -> let's fill int he most likely case
# based on the frequency

# We could drop all the row where either "Age" or "Embarked" is missing.
# However we end up with only 60% success rate on the test data if we do this.
# While by simply filling in the average or most frequent we get around 80%.
train_data_without_na = train_data.dropna()

y_train = train_data["Survived"]
x_train = train_data\
        .drop(columns=["Survived"])

collum_trans = ColumnTransformer(transformers=[
        # remove the answer from the training data
        # ('remove answer', 'drop',  'Survived'), # done beforehand...
        # Remove the following collumn's as I haven't figured out
        # how to convert them into numerical values. They could
        # be used to help fill in the nan's.
        ('name', 'drop', ['Name', 'Fare', 'Cabin', 'Ticket']),
        ("Fill in age", SimpleImputer(strategy='mean'), ['Age']),
        ("Embarked one hot encoder",\
         make_pipeline(\
                       SimpleImputer(strategy='most_frequent'),\
                       OneHotEncoder()),\
         ['Embarked']),
        ("Sex ordinal", OrdinalEncoder(), ['Sex']),
        ], remainder='passthrough')
pipeline = Pipeline([
        ('cols', collum_trans),
        ('scale', StandardScaler()),
        ('cf', SVC())
])

# parameters of pipelines can be set using '__' seperated parameter names:
# 'cf__kernel': ['linear', 'poly', 'sigmoid', 'precomputed', 'rbf']
# param_grid = {
#         'cf__kernel': ['linear', 'rbf', 'poly'],
#         'cf__degree': [1, 2, 3, 4, 5],
#         'cf__gamma': ['scale', 'auto'],
#         'cf__shrinking': [True, False]
# }

# putting best param grid here, as the other takes a while to run.
best_param_grid = {
        'cf__kernel': ['rbf'],
        'cf__degree': [1],
        'cf__gamma': ['scale'],
        'cf__shrinking': [True]
}

search = GridSearchCV(pipeline, best_param_grid, verbose=3)
est = search.fit(x_train, y_train)

# Bad way to measure correctness, but it should at least be good.
y_train_pred = est.predict(x_train)
error_on_train_set = y_train_pred - y_train
success_rate = sum(0 if x != 0 else 1 for x in error_on_train_set)\
        / len(error_on_train_set)
print(("success rate on training set={:.0%}").format(success_rate))

error_on_test_set = est.predict(x_test) - y_test["Survived"].to_numpy()
success_rate_test = sum(0 if x != 0 else 1 for x in error_on_test_set)\
        / len(error_on_test_set)
print(("success rate on test set={:.0%}").format(success_rate_test))

# conclussion:
# We get a succes rate of 83% on the test set.
# possible improvements:
# 1. Better estimate missing ages, now we take the average but
# we can do better. Names with "miss" in it are generally younger
# then those without... this kind of stuff can help
# 2. Better estimate missing Embarked, I wonder if I can use some
# kind of clustering or correlation thing to find out what the value
# most likely is.
# 3. Use a different kind of classifier, such as SGDClassifier.

# 4. Is this a skewed dataset?

# Confusion matrix:
conf_matrix = confusion_matrix(y_train, y_train_pred)
# SVM:
# negative: 94% | 6% -> looks really good
# positive: 30% | 70% -> seems it clasifies quiet
#                        a lot of positives as negatives

TP = conf_matrix[1, 1]
FP = conf_matrix[1, 0]
TN = conf_matrix[0, 0]
FN = conf_matrix[0, 1]

# Accuracy of positive precision.
# precision = TP/(TP+FP)
precision = precision_score(y_train, y_train_pred)
print("precision={:.0%}".format(precision))

# Ration of positive that are correctly detected.
# recall = TP/(TP+FN)
recall = recall_score(y_train, y_train_pred)
print("recall={:.0%}".format(recall))

# F1 = precision*recall/(precision + recall)
F1 = f1_score(y_train, y_train_pred)
print("F1 score={:.0%}".format(F1))

# We could tune the recall/precision by using a manual treshhold.
# y_scores = cross_val_predict(est, x_train, y_train,
#                              cv=5, method='decision_function')
# precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
# thresholds[np.argmax(precisions >= 0.90)]
# -> est.decision_function(..) will return score prediction, we can use this
#                       with treshhold we got here.
