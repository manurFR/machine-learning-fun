from math import sqrt, pi, exp
from numpy import mean, std
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB


COLUMN_NAMES = ['pregnancies', 'glucose', 'blood pressure', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'diabetic']
filename = 'pima-indians-diabetes.data'


def load_csv(fname):
    return pandas.read_csv(fname, names=COLUMN_NAMES)

dataset = load_csv(filename)
print 'Loaded data file {0} with {1} rows'.format(filename, len(dataset))

classes = dataset.pop('diabetic')

train, test, train_classes, test_classes = train_test_split(dataset, classes, train_size=0.67, random_state=666)
print 'Split {0} rows into train with {1} and test with {2}'.format(len(dataset), len(train), len(test))

# 2.1 Separate data by class
train_by_class = dict.fromkeys(pandas.unique(train_classes), None)
for i in range(len(train)):
    if train_by_class[train_classes[i]] is None:
        train_by_class[train_classes[i]] = []
    train_by_class[train_classes[i]].append(train[i])

# 2.2 Calculate mean and stddev
means_by_class = {clazz: mean(train_by_class[clazz], axis=0) for clazz in train_by_class}
stddev_by_class = {clazz: std(train_by_class[clazz], axis=0, ddof=1) for clazz in train_by_class}


def gaussian_probability(x, mean_property, stddev_property):
    return (1 / (sqrt(2*pi) * stddev_property)) * exp(-(pow(x - mean_property, 2)/(2*pow(stddev_property, 2))))


def class_probability(instance):
    probabilities = {clazz: 1 for clazz in train_by_class}
    for clazz in probabilities:
        for j in range(len(instance)):
            probabilities[clazz] *= gaussian_probability(instance[j], means_by_class[clazz][j], stddev_by_class[clazz][j])
    return probabilities


def predict_class(instance):
    probabilities = class_probability(instance)
    best_class, best_proba = None, -1
    for value, proba in probabilities.iteritems():
        if best_proba is None or proba > best_proba:
            best_class = value
            best_proba = proba
    return best_class

# predictions for all test set instances
predictions = [predict_class(instance) for instance in test]

# accuracy
correct_predictions = 0
for i in range(len(test_classes)):
    if predictions[i] == test_classes[i]:
        correct_predictions += 1
print 'Accuracy: {0}'.format(100.0*correct_predictions/float(len(test_classes)))