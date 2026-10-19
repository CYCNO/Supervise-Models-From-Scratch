from utils import *
connect_models_file()

import numpy as np
import pandas as pd
from models.decisionTree import DecisionTree

# Recreating your dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong']
}
X = pd.DataFrame(data)
y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]) # 0: No, 1: Yes

tree = DecisionTree(X, y)
tree.print_tree()

# [Outlook]
# ├── If Overcast ──> Outcome: 1
# ├── If Rain
# │   [Wind]
# │   ├── If Strong ──> Outcome: 0
# │   └── If Weak ──> Outcome: 1
# └── If Sunny
#     [Humidity]
#     ├── If High ──> Outcome: 0
#     └── If Normal ──> Outcome: 1

row = {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'}
print(tree.predict(row))
# 0
