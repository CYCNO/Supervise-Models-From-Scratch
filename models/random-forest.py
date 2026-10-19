import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self, X, y, max_features="sqrt", random_state=None):
        self.X = X
        self.y = y
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)

        self.tree = self.build_tree(X, y)

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
        counter = self.tokenize_column(X, y, name)
        n_samples = len(y)

        if n_samples == 0:
            return 0.0

        gini_weight = 0.0

        for stats in counter.values():
            total = stats["total"]

            if total == 0:
                continue

            p = stats["p"] / total
            q = 1.0 - p
            weight = total / n_samples

            gini_weight += weight * self.gini_impurity(p, q)

        return gini_weight

    def get_low_imp(self, X, y):
        columns = list(X.columns)

        if not columns:
            return {"name": None, "value": float("inf")}

        # Randomly select features to consider at this split
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(len(columns))))
        elif self.max_features is None:
            k = len(columns)
        else:
            k = max(1, min(int(self.max_features), len(columns)))

        features = self.rng.choice(
            columns, size=k, replace=False
        )

        low_imp = {"name": None, "value": float("inf")}

        for col in features:
            curr_imp = self.gini_wt_column(X, y, col)

            if curr_imp < low_imp["value"]:
                low_imp = {"name": col, "value": curr_imp}

        return low_imp

    def build_tree(self, X, y):
        if len(y) == 0:
            return 0

        unique_classes = np.unique(y)

        # Pure node: all samples have the same class
        if len(unique_classes) == 1:
            return int(unique_classes[0])

        # No features left: return majority class
        if len(X.columns) == 0:
            return int(y.mode().iloc[0])

        best_feat = self.get_low_imp(X, y)["name"]

        if best_feat is None:
            return int(y.mode().iloc[0])

        tree = {best_feat: {}}

        for category in X[best_feat].unique():
            mask = X[best_feat] == category

            X_sub = X.loc[mask].drop(columns=[best_feat])
            y_sub = y.loc[mask] if hasattr(y, "loc") else y[mask]

            tree[best_feat][category] = self.build_tree(
                X_sub, y_sub
            )

        return tree

    def print_tree(self, tree=None, indent=""):
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            print(f" ──> Outcome: {tree}")
            return

        for node_name, branches in tree.items():
            print(f"\n{indent}[{node_name}]")

            items = list(branches.items())

            for i, (category, subtree) in enumerate(items):
                is_last = i == len(items) - 1
                prefix = "└── " if is_last else "├── "

                print(f"{indent}{prefix}If {category}", end="")

                next_indent = indent + (
                    "    " if is_last else "│   "
                )

                self.print_tree(subtree, next_indent)

    def predict(self, row):
        current_layer = self.tree

        while isinstance(current_layer, dict):
            node = next(iter(current_layer))
            value = row[node]

            # Handle unseen categories by returning the
            # majority class of the training data.
            if value not in current_layer[node]:
                return int(self.y.mode().iloc[0])

            current_layer = current_layer[node][value]

        return int(current_layer)


class RandomForest:
    def __init__(
        self, data, n_trees=10, max_features="sqrt",
        random_state=None
    ):
        self.data = data
        self.n_trees = n_trees
        self.max_features = max_features
        self.random_state = random_state
        self.forest = {}

    def fit(self):
        self.forest = {}

        rng = np.random.default_rng(self.random_state)

        for n in range(self.n_trees):
            # Bootstrap sampling: sample rows with replacement
            indices = rng.integers(
                0, len(self.data), size=len(self.data)
            )

            boot_data = self.data.iloc[indices].copy()

            X_boot = boot_data.drop(columns=["Loan_Approved"])

            y_boot = boot_data["Loan_Approved"].map(
                lambda x: 1 if x == "Yes" else 0
            )

            # Each tree gets its own random feature-selection seed
            tree_seed = int(rng.integers(0, 2**32 - 1))

            tree = DecisionTree(
                X_boot,
                y_boot,
                max_features=self.max_features,
                random_state=tree_seed
            )

            self.forest[n] = tree

            if n%10 == 0:
                print(f"{n}/{self.n_trees} Trees Trained")
        print("All Trees Have Been Trained.")

    def predict(self, row):
        if not self.forest:
            raise ValueError("Call fit() before predict().")

        votes = [
            tree.predict(row)
            for tree in self.forest.values()
        ]

        # Majority voting; ties predict class 0
        return int(sum(votes) > len(votes) / 2)

    def predict_votes(self, row):
        """Return the number of votes for each class."""
        if not self.forest:
            raise ValueError("Call fit() before predict_votes().")

        votes = {0: 0, 1: 0}

        for tree in self.forest.values():
            prediction = tree.predict(row)
            votes[prediction] += 1

        return votes

    def print_forest(self):
        for n, tree in self.forest.items():
            print("-" * 30)
            print(f"Tree: {n + 1}")
            tree.print_tree()
