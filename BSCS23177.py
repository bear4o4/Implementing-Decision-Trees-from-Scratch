from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pprint
from sklearn.metrics import accuracy_score



iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels



#task1
classes, counts = np.unique(y, return_counts=True)
probabilities = counts / counts.sum()
entropy = -np.sum(probabilities * np.log2(probabilities))
print(f"Initial Entropy of the dataset: {entropy:.4f}")
print("####################################")


#task2
petal_length = X[:, 2]
sorted_values = np.sort(np.unique(petal_length))
thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2
print("Candidate thresholds for petal length:")
print(thresholds)
print("####################################")



#task3
feature_index = 2
def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))
feature_values = X[:, feature_index]
sorted_values = np.sort(np.unique(feature_values))
thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2
original_entropy = entropy(y)
best_threshold = None
best_info_gain = -1
for threshold in thresholds:

    left_indices = feature_values <= threshold
    right_indices = feature_values > threshold

    y_left = y[left_indices]
    y_right = y[right_indices]


    if len(y_left) == 0 or len(y_right) == 0:
        continue


    H_left = entropy(y_left)
    H_right = entropy(y_right)


    H_split = (len(y_left) / len(y)) * H_left + (len(y_right) / len(y)) * H_right

    info_gain = original_entropy - H_split

    if info_gain > best_info_gain:
        best_info_gain = info_gain
        best_threshold = threshold
print(f"Best threshold for petal length: {best_threshold}")
print(f"Information Gain: {best_info_gain:.4f}")
print("####################################")




#task4
def best_split(X, y):
    n_features = X.shape[1]
    best_feature = None
    best_threshold = None
    best_info_gain = -1
    original_entropy = entropy(y)

    for feature_index in range(n_features):
        feature_values = X[:, feature_index]
        sorted_values = np.sort(np.unique(feature_values))
        thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2

        for threshold in thresholds:
            left_indices = feature_values <= threshold
            right_indices = feature_values > threshold
            y_left = y[left_indices]
            y_right = y[right_indices]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            H_left = entropy(y_left)
            H_right = entropy(y_right)

            H_split = (len(y_left) / len(y)) * H_left + (len(y_right) / len(y)) * H_right
            info_gain = original_entropy - H_split

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold, best_info_gain
feature, threshold, gain = best_split(X, y)
print(f"Best Feature Index: {feature}")
print(f"Best Threshold: {threshold}")
print(f"Information Gain: {gain:.4f}")
print("####################################")



#task5
def build_tree(X, y, depth=0, max_depth=5):

    if len(np.unique(y)) == 1:
        return {"type": "leaf", "class": y[0]}
    if depth >= max_depth:

        values, counts = np.unique(y, return_counts=True)
        majority_class = values[np.argmax(counts)]
        return {"type": "leaf", "class": majority_class}


    feature, threshold,_ = best_split(X, y)

    if feature is None:

        values, counts = np.unique(y, return_counts=True)
        majority_class = values[np.argmax(counts)]
        return {"type": "leaf", "class": majority_class}


    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    left_subtree = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right_subtree = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)

    return {
        "type": "node",
        "feature": feature,
        "threshold": threshold,
        "left": left_subtree,
        "right": right_subtree
    }
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tree = build_tree(X_train, y_train,max_depth=3)
pprint.pprint(tree)
print("####################################")




#task6
def predict_sample(tree, sample):

    if tree["type"] == "leaf":
        return tree["class"]


    feature_value = sample[tree["feature"]]
    if feature_value <= tree["threshold"]:
        return predict_sample(tree["left"], sample)
    else:
        return predict_sample(tree["right"], sample)
def predict(tree, X):
    return np.array([predict_sample(tree, sample) for sample in X])
y_pred = predict(tree, X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("####################################")



#task7
def calculate_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = (correct / total) * 100
    return accuracy
y_pred = predict(tree, X_test)
accuracy = calculate_accuracy(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}%")
print("##################################")



#task8
def trace_decision_path(tree, sample, feature_names, sample_index=0):
    print(f"\nTracing Decision Path for Sample {sample_index}:")

    node = tree
    while node["type"] != "leaf":
        feature_index = node["feature"]
        threshold = node["threshold"]
        feature_value = sample[feature_index]
        feature_name = feature_names[feature_index]

        if feature_value <= threshold:
            print(f"Checking feature: {feature_name} ≤ {threshold:.2f}")
            print(f"→ Going left (value = {feature_value:.2f})")
            node = node["left"]
        else:
            print(f"Checking feature: {feature_name} ≤ {threshold:.2f}")
            print(f"→ Going right (value = {feature_value:.2f})")
            node = node["right"]

    print(f"Final Prediction: {node['class']} ({iris.target_names[node['class']]})")
sample_index = 3
sample = X_test[sample_index]
feature_names = iris.feature_names
trace_decision_path(tree, sample, feature_names, sample_index)
print("####################################")

