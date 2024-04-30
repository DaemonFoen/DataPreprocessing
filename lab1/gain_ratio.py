import numpy as np


def gain_ratio(data, target_column, attribute_column):
    entropy_before = entropy(data[target_column])

    attribute_values = data[attribute_column].unique()
    entropy_after = 0

    for value in attribute_values:
        subset = data[data[attribute_column] == value]
        subset_size = len(subset)
        subset_entropy = entropy(subset[target_column])
        entropy_after += (subset_size / len(data)) * subset_entropy

    information_gain = entropy_before - entropy_after

    split_info = entropy(data[attribute_column])

    if split_info == 0:
        return 0

    return information_gain / split_info


def entropy(column):
    column = column.astype(str)
    values, counts = np.unique(column, return_counts=True)
    probabilities = counts / len(column)
    return -np.sum(probabilities * np.log2(probabilities))
