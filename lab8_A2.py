import pandas as pd
import numpy as np

# Read the dataset
data = pd.read_excel("C:\Users\akhil\Downloads\DATA (3) (1).zip")  # Replace "your_dataset.csv" with the actual filename

# Define function for information gain calculation
def calculate_information_gain(data, feature_name, target_name):
    """
    Calculate the information gain for a given feature.
    """
    # Calculate entropy of the parent node
    total_entropy = calculate_entropy(data[target_name])

    # Calculate weighted average of entropy for child nodes
    weighted_entropy = 0
    for value in data[feature_name].unique():
        subset = data[data[feature_name] == value]
        weighted_entropy += (len(subset) / len(data)) * calculate_entropy(subset[target_name])

    # Calculate information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Define function to calculate entropy
def calculate_entropy(target):
    """
    Calculate the entropy of a target variable.
    """
    entropy = 0
    classes = target.unique()
    total_count = len(target)
    for c in classes:
        count = len(target[target == c])
        p = count / total_count
        entropy -= p * np.log2(p)
    return entropy

# Define function for binning continuous features
def bin_continuous_feature(data, feature_name, num_bins, binning_type="equal_width"):
    """
    Convert a continuous feature to categorical by binning.
    """
    if binning_type == "equal_width":
        bins = np.linspace(data[feature_name].min(), data[feature_name].max(), num_bins + 1)
    elif binning_type == "frequency":
        bins = np.percentile(data[feature_name], np.arange(0, 100, 100 / num_bins))
        bins[-1] = data[feature_name].max()  # Ensure the last bin includes max value
    else:
        raise ValueError("Invalid binning type. Use 'equal_width' or 'frequency'.")

    binned_feature = pd.cut(data[feature_name], bins, labels=range(num_bins))
    return binned_feature

# Function to find the best split feature for the root node
def find_root_node(data, target_name, binning_type="equal_width", num_bins=5):
    """
    Find the best split feature for the root node using information gain.
    """
    features = data.columns.tolist()
    features.remove(target_name)

    best_feature = None
    max_information_gain = -float('inf')

    for feature in features:
        if data[feature].dtype == 'object':
            information_gain = calculate_information_gain(data, feature, target_name)
        else:
            # Convert continuous feature to categorical
            binned_feature = bin_continuous_feature(data, feature, num_bins, binning_type)
            data_binned = data.copy()
            data_binned[feature] = binned_feature
            information_gain = calculate_information_gain(data_binned, feature, target_name)

        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_feature = feature

    return best_feature

# Test the function with the provided dataset
root_node = find_root_node(data, "Chg%")
print("Root node:", root_node)
