# bruh the easiest model, we just find the distance between new data to old(training) data
# and the least distance from the training data it will be, it would become data from 
# that category, we use euclidean distance formula or minkowski formula
# more on https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/

import numpy as np
from collections import Counter

class KNN:
    def __init__(self):
        pass

    # not using this because it's very slow, instead we using Counter 
    # def _most_count(self, nearest_labels):
    #     count_table = {}
    #     for i in nearest_labels:
    #         if i in count_table:
    #             count_table[i] += 1
    #         else:
    #             count_table[i] = 1
    #     return max(count_table, key=count_table.get)

    def _euclidean_distance(self, point_1, point_2):
        return np.sqrt(sum((np.array(point_1) - np.array(point_2))**2)) # np.sqrt((p1[0]-p2[0]**2 + p1[1]-p2[1]**2)
    
    def predict(self, training_data, training_labels, test_point, k):
        distances =[]
        for i in range(len(training_data)):
            dist = self._euclidean_distance(test_point, training_data[i])
            distances.append((dist, training_labels[i]))

        distances.sort()
        nearest_labels = [label for _, label in distances[:k]]
        return Counter(nearest_labels).most_common(1)[0][0]
