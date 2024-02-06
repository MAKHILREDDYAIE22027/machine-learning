def label_encode_categorical(data):
    """
    Converts categorical variables to numeric using label encoding.

    Parameters:
    data (list or numpy array): A list or numpy array containing categorical variables.

    Returns:
    encoded_data (list): List of numeric labels corresponding to the categorical data.
    label_mapping (dict): Dictionary mapping each unique category to its corresponding numeric label.
    """

    # Initialize an empty dictionary to store label mappings
    label_mapping = {}

    # Initialize an empty list to store encoded data
    encoded_data = []

    # Iterate through the unique categories in the data
    for category in data:
        # If the category has not been assigned a label yet
        if category not in label_mapping:
            # Assign a new label to the category
            label_mapping[category] = len(label_mapping)
        # Append the label corresponding to the category to the encoded data list
        encoded_data.append(label_mapping[category])

    return encoded_data, label_mapping
