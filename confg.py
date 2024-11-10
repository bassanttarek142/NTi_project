import pandas as pd
import os

def load_data(file1_path, file2_path):
    """Load two CSV files into pandas DataFrames."""
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    return df1, df2

def load_txt_files_from_folder(folder_path, sentiment_label):
    """Load .txt files from a folder and return a DataFrame with text and sentiment."""
    texts = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read().strip())
    return pd.DataFrame({'Text': texts, 'sentiment': sentiment_label})

def remove_duplicates(df):
    """Remove duplicates based on the 'Text' column in a DataFrame."""
    return df.drop_duplicates(subset='Text', keep='first')

def combine_data(df1, df2):
    """Concatenate two DataFrames and remove duplicates."""
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df = remove_duplicates(combined_df)
    return combined_df

def save_data(df, output_path):
    """Save DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)

def main():
    # Define file paths
    file2 = "train_all_ext.csv"      
    output_file = "cleaned_combined_data.csv"  
    folder1 = "pos"
    folder2 = "neg"

    df1, df2 = load_data(file1, file2)

    pos_df = load_txt_files_from_folder(folder1, 'positive')
    neg_df = load_txt_files_from_folder(folder2, 'negative')

    # Combine all datasets
    combined_df = combine_data(df1, df2)
    combined_df = combine_data(combined_df, pos_df)
    combined_df = combine_data(combined_df, neg_df)
    
    save_data(combined_df, output_file)

    print("Data preprocessing complete. Final file saved as:", output_file)

if __name__ == "__main__":
    main()
