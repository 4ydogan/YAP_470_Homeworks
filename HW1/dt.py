class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.label = None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def calculate_gini(self, labels):
        total_samples = len(labels)
        gini = 1.0

        classes = set(labels)
        for cls in classes:
            p = labels.count(cls) / total_samples
            gini -= p ** 2

        return gini

    def split_dataset(self, dataset, feature_index, feature_value):
        left_subset = []
        right_subset = []

        for data in dataset:
            if data[feature_index] <= feature_value:
                left_subset.append(data)
            else:
                right_subset.append(data)

        return left_subset, right_subset

    def get_best_split(self, dataset):
        best_gini = float('inf')
        best_feature_index = -1
        best_feature_value = None
        num_features = len(dataset[0]) - 1

        for feature_index in range(num_features):
            feature_values = set(data[feature_index] for data in dataset)

            for feature_value in feature_values:
                left_subset, right_subset = self.split_dataset(dataset, feature_index, feature_value)
                gini = (len(left_subset) / len(dataset)) * self.calculate_gini([data[-1] for data in left_subset]) \
                        + (len(right_subset) / len(dataset)) * self.calculate_gini([data[-1] for data in right_subset])

                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_feature_value = feature_value

        return best_feature_index, best_feature_value

    def create_tree(self, dataset, depth):
        labels = [data[-1] for data in dataset]

        if len(set(labels)) == 1:
            node = Node(dataset)
            node.label = labels[0]
            return node

        if self.max_depth is not None and depth >= self.max_depth:
            most_common_label = max(set(labels), key=labels.count)
            node = Node(dataset)
            node.label = most_common_label
            return node

        feature_index, feature_value = self.get_best_split(dataset)
        left_subset, right_subset = self.split_dataset(dataset, feature_index, feature_value)

        node = Node(dataset)
        node.data = [feature_index, feature_value]
        node.left = self.create_tree(left_subset, depth + 1)
        node.right = self.create_tree(right_subset, depth + 1)

        return node

    def fit(self, X, y):
        dataset = [list(x) + [label] for x, label in zip(X, y)]
        self.tree = self.create_tree(dataset, 0)

    def predict_sample(self, sample, node):
        if node.label is not None:
            return node.label

        if sample[node.data[0]] <= node.data[1]:
            return self.predict_sample(sample, node.left)
        else:
            return self.predict_sample(sample, node.right)

    def predict(self, X):
        predictions = []
        for sample in X:
            prediction = self.predict_sample(sample, self.tree)
            predictions.append(prediction)
        return predictions
