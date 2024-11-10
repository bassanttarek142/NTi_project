# data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from text_preprocessing import data_cleaning
import pickle

def load_data(file_path):
    """Load the cleaned and combined dataset."""
    data = pd.read_csv(file_path)
    return data

def preprocess_text(data):
    """Apply text cleaning to the 'Text' column and save in 'cleaned_text'."""
    # Ensure the 'Text' column is of type string
    data['Text'] = data['Text'].astype(str)
    
    # Apply the data cleaning function
    data['cleaned_text'] = data['Text'].apply(data_cleaning)
    
    # Remove rows with empty 'cleaned_text'
    data = data[data['cleaned_text'].str.strip().astype(bool)]
    data = data.reset_index(drop=True)
    
    return data

def encode_labels(data):
    """Encode sentiment labels into integers."""
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['sentiment'])
    
    # Save the label encoder for future use
    with open("label_encoder.pkl", "wb") as le_file:
        pickle.dump(le, le_file)
    
    # Return only data, as label_encoder is saved
    return data

def shuffle_and_split(data, test_size=0.15, val_size=0.15, random_state=42):
    """Shuffle and split the data into train, validation, and test sets."""
    # Create a train-test split
    train_full, test = train_test_split(
        data, test_size=test_size, random_state=random_state, shuffle=True, stratify=data['label']
    )
    
    # Adjust validation size based on remaining data
    val_size_adjusted = val_size / (1 - test_size)
    
    # Create a train-validation split from the training set
    train, val = train_test_split(
        train_full, test_size=val_size_adjusted, random_state=random_state, stratify=train_full['label']
    )
    
    return train, val, test

def prepare_data(file_path):
    """Main function to prepare and shuffle/split data."""
    # Load data
    data = load_data(file_path)

    # Preprocess text
    data = preprocess_text(data)

    # Encode labels
    data = encode_labels(data)

    # Optionally, check class distribution
    print("Class distribution after preprocessing:")
    print(data['label'].value_counts())

    # Shuffle and split data into train, validation, and test sets
    train, val, test = shuffle_and_split(data)

    print("Data preparation and splitting complete.")

    return train, val, test

if __name__ == "__main__":
    # Prepare data and get training/validation/testing sets
    train, val, test = prepare_data("final_file.csv")

    # Inspect the data shapes
    print("Training data shape:", train.shape)
    print("Validation data shape:", val.shape)
    print("Test data shape:", test.shape)

    # Display sample cleaned text
    print("\nSample cleaned texts:")
    print(train['cleaned_text'].head())
