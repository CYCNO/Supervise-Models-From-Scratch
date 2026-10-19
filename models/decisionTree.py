import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        # Automatically build the tree as soon as the class is created
        self.tree = self.build_tree(self.X, self.y)

    def gini_impurity(self, p, q):
        return 1 - (p**2 + q**2)

    def tokenize_column(self, X, y, name):
        counter = {}
        for i in range(len(y)):
            value = X[name].iloc[i]
            label = y.iloc[i] if hasattr(y, "iloc") else y[i]

            if value not in counter:
                counter[value] = {"total": 0, "p": 0}
            counter[value]["total"] += 1

            if label == 1:
                counter[value]["p"] += 1
        return counter

    def gini_wt_column(self, X, y, name):
        # Notice how we use self. to call other functions inside the class
        counter = self.tokenize_column(X, y, name)
        n_samples = len(y)

        gini_weight = 0.0
        for val, stats in counter.items():
            total = stats["total"]
            if total == 0:
                continue

            p = stats["p"] / total
            q = 1.0 - p
            weight = total / n_samples
            gini_weight += weight * self.gini_impurity(p, q)

        return gini_weight

    def get_low_imp(self, X, y):
        low_imp = {"name": "", "value": float("inf")}

        for col in X.columns:
            curr_imp = self.gini_wt_column(X, y, col)
            if curr_imp < low_imp["value"]:
                low_imp = {"name": col, "value": curr_imp}

        return low_imp

    def build_tree(self, X, y):
        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            return int(unique_classes[0])

        if len(X.columns) == 0:
            counts = np.bincount(y)
            return int(np.argmax(counts))

        best_feat = self.get_low_imp(X, y)["name"]

        tree = {best_feat: {}}
        for category in np.unique(X[best_feat]):
            mask = X[best_feat] == category
            X_sub = X[mask].drop(columns=[best_feat])
            y_sub = y[mask]

            # self.build_tree for recursion
            tree[best_feat][category] = self.build_tree(X_sub, y_sub)
        return tree

    def print_tree(self, tree=None, indent=""):
        # If no tree is passed, default to the one we built in __init__
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            print(f" ──> Outcome: {tree}")
            return

        for node_name in tree:
            print(f"\n{indent}[{node_name}]")
            branches = tree[node_name]

            for i, (category, subtree) in enumerate(branches.items()):
                is_last = (i == len(branches) - 1)
                prefix = "└── " if is_last else "├── "

                print(f"{indent}{prefix}If {category}", end="")
                self.print_tree(subtree, indent + ("    " if is_last else "│   "))

    def predict(self, row):
        # Automatically use the tree saved inside the class
        current_layer = self.tree

        while isinstance(current_layer, dict):
            node = list(current_layer.keys())[0]
            value = row[node]
            current_layer = current_layer[node][value]

        return current_layer
