from utils import *
connect_models_file()

from models.knn import KNN

training_data = [
    [150, 45],  # height (cm), weight (kg)
    [155, 48],
    [160, 52],
    [158, 50],
    [162, 55],
    [165, 58],
    [168, 60],
    [170, 62],

    [172, 68],
    [175, 70],
    [178, 75],
    [180, 78],
    [182, 82],
    [185, 85],
    [177, 72],
    [169, 64]
]

training_labels = [
    'Teen', 'Teen', 'Teen', 'Teen', 'Teen', 'Teen', 'Teen', 'Teen',
    'Adult', 'Adult', 'Adult', 'Adult', 'Adult', 'Adult', 'Adult', 'Adult'
]


test_point = [231, 34]

k = 3

knn = KNN()

predict = knn.predict(training_data, training_labels, test_point, k)
print(predict)