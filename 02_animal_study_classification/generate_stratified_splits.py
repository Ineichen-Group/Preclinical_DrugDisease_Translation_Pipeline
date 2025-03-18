import os
import pandas as pd
import numpy as np
import re
import csv
from sklearn.model_selection import train_test_split

def map_labels(df):

    # TODO double-check the classes are correct

    label_map = {
        'REST': [
            'Clinical-study-protocol',
            'Human-systematic-review',
            'Non-systematic-review',
            'Human-RCT-non-drug-intervention',
            'Human-RCT-drug-intervention',
            'Human-RCT-non-intervention',
            'Human-case-report',
            'Human-non-RCT-non-drug-intervention',
            'Human-non-RCT-drug-intervention',
            'In-vitro-study',
            'Remaining'
        ],
        'ANIMAL': [
            'Animal-systematic-review',
            'Animal-non-drug-intervention',
            'Animal-drug-intervention',
            'Animal-other'
        ]
    }

    label_mapping_multi = {
        'Remaining': 0,
        'Non-systematic-review': 1,
        'Human-non-RCT-non-drug-intervention': 2,
        'Human-non-RCT-drug-intervention': 3,
        'Human-case-report': 4,
        'Animal-other': 5,
        'Animal-drug-intervention': 6,
        'Human-systematic-review': 7,
        'In-vitro-study': 8,
        'Human-RCT-non-drug-intervention': 9,
        'Animal-non-drug-intervention': 10,
        'Human-RCT-drug-intervention': 11,
        'Clinical-study-protocol': 12,
        'Human-RCT-non-intervention': 13
    }


    def map_to_label(col, label_map):
        """ 
        Function to map flattened_labels to the corresponding label 
        """
        for label, sublabels in label_map.items():
            if col in sublabels:
                if label == 'REST':
                    return 0  # Binary label for 'REST'
                else:
                    return 1  # Binary label for 'ANIMAL'
        return None

    # Encode multiclass labels
    df['multi_label'] = df['accepted_label'].map(label_mapping_multi)
    # Encode binary labels
    df['binary_label'] = df.apply(lambda row: map_to_label(row['accepted_label'], label_map), axis=1)

    # drop na
    df.dropna(subset=['multi_label'], inplace=True)
    df.dropna(subset=['binary_label'], inplace=True)
    # Convert float labels to integers
    df['multi_label'] = df['multi_label'].astype(int)
    df['binary_label'] = df['binary_label'].astype(int)
    
    return df


def get_data(df):
    return df[['pmid', 'journal_name', 'title', 'abstract', 'accepted_label', 'multi_label', 'binary_label']]


def save_splits(train_data, val_data, test_data, save_dir):
    os.makedirs(save_dir, exist_ok=True)  # create folder only if does not exist
    train_data.to_csv(os.path.join(save_dir, 'train.csv'), index=False, columns=['idx', 'pmid', 'journal_name', 'title', 'abstract', 'accepted_label', 'multi_label', 'binary_label'])
    val_data.to_csv(os.path.join(save_dir, 'val.csv'), index=False, columns=['idx', 'pmid', 'journal_name', 'title', 'abstract', 'accepted_label', 'multi_label', 'binary_label'])
    
    # Save test data only if test split exists
    if test_data is not None:
        test_data.to_csv(os.path.join(save_dir, 'test.csv'), index=False, columns=['idx', 'pmid', 'journal_name', 'title', 'abstract', 'accepted_label', 'multi_label', 'binary_label'])
    
    print("Data splits have been saved successfully.")

    # Log class distribution for each split
    with open(os.path.join(save_dir, 'split_info.log'), 'a') as f:
        f.write('Train multi:\n' + str(train_data['multi_label'].value_counts()) + '\n\n')
        f.write('Val multi:\n' + str(val_data['multi_label'].value_counts()) + '\n\n')
        
        if test_data is not None:
            f.write('Test multi:\n' + str(test_data['multi_label'].value_counts()) + '\n\n')
        
        f.write('Train binary:\n' + str(train_data['binary_label'].value_counts()) + '\n\n')
        f.write('Val binary:\n' + str(val_data['binary_label'].value_counts()) + '\n\n')
        
        if test_data is not None:
            f.write('Test binary:\n' + str(test_data['binary_label'].value_counts()) + '\n\n')


def custom_train_test_split(data, test_size, val_size, random_state):
    # Identify unique classes
    unique_classes = data['multi_label'].unique()
    
    # Initialize lists to store indices of samples for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Shuffle the dataset
    data_shuffled = data.sample(frac=1, random_state=random_state)
    
    # Split the dataset ensuring at least one sample for each class in each split
    for label in unique_classes:
        label_data = data_shuffled[data_shuffled['multi_label'] == label]
        
        # Split based on whether test_size is greater than 0
        if test_size > 0:
            train_label, val_test_label = train_test_split(label_data, test_size=test_size + val_size, random_state=random_state)
            val_label, test_label = train_test_split(val_test_label, test_size=test_size / (test_size + val_size), random_state=random_state)
            test_indices.extend(test_label.index.tolist())
        else:
            train_label, val_label = train_test_split(label_data, test_size=val_size, random_state=random_state)
        
        train_indices.extend(train_label.index.tolist())
        val_indices.extend(val_label.index.tolist())
    
    # Create DataFrame for each split using the selected indices
    train_data = data.loc[train_indices]
    val_data = data.loc[val_indices]
    
    # Conditionally create test_data if test_size > 0
    test_data = data.loc[test_indices] if test_size > 0 else None

    # Add new column 'idx' starting from 1 for each split
    train_data['idx'] = np.arange(1, len(train_data) + 1)
    val_data['idx'] = np.arange(1, len(val_data) + 1)
    if test_data is not None:
        test_data['idx'] = np.arange(1, len(test_data) + 1)
    
    return train_data, val_data, test_data



def main():
    # old dataset
    # path_to_file = "./../../data/prodigy/annotated_output/final/full_combined_dataset_1996.csv"
    # enriched dataset
    path_to_file = './data/study_type_teller/full_enriched_dataset_2696.csv'
    save_dir = "./data/study_type_teller/9-1-0_all_classes_enriched" # new enriched dataset
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(path_to_file)
    print(df)

    # start logging
    with open(os.path.join(save_dir, 'split_info.log'), 'w') as f:
        f.write(f"\n\n{10*'*'} Creating stratified splits on the enriched dataset {10*'*'}\n\n")


    df = map_labels(df)
    data = get_data(df) # Get the necessary columns

    # Define split sizes and random state
    test_size = 0.0
    val_size = 0.1
    random_state = 42

    # Perform custom train-test split
    train_data, val_data, test_data = custom_train_test_split(data, test_size, val_size, random_state)

    save_splits(train_data, val_data, test_data, save_dir)


if __name__ == "__main__":
    main()
