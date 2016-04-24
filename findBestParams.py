import csv

# current time
import time
current_milli_time = lambda: int(round(time.time() * 1000))

from sklearn import svm, grid_search

training_X = []
training_y = []

testing_X = []
testing_y = []

# parameters = {'C': [0.3, 1, 10]}

with open('trainging_feature_vector_103950.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        row = [float(s) for s in row]
        training_X.append(row)
for X in training_X:
    training_y.append(X.pop(0))

with open('test.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        row = [float(s) for s in row]
        testing_X.append(row)
for X in testing_X:
    testing_y.append(X.pop(0))


# clf = svm.LinearSVC(C=0.3)
clf = grid_search.GridSearchCV(svm.LinearSVC(), parameters, cv=3)
clf.fit(training_X, training_y)

print("Best parameters set found on development set:")
print(clf.best_params_)
print('\n')

# print("Grid scores on development set:")
# for params, mean_score, scores in clf.grid_scores_:
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean_score, scores.std() * 2, params))
# print('\n')

print("Start predicting ... at {}".format(current_milli_time()))
clf.predict(testing_X)
print("Finish predicting at {}".format(current_milli_time()))

