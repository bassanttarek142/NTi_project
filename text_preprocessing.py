import pandas as pd
import numpy as np
import re
import string
import emoji
import pyarabic.araby as ar
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

data = pd.read_csv("cleaned_combined_data.csv")

data = data.dropna(subset=['Text'])  
data = data.drop(columns=['Tweet_id'])

def data_cleaning(text):
    """Clean and preprocess Arabic text data."""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'[A-Za-z]+', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    text = ar.strip_tashkeel(text)
    text = ar.strip_tatweel(text)
    
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    all_punctuations = string.punctuation + arabic_punctuations
    translator = str.maketrans('', '', all_punctuations)
    text = text.translate(translator)
    
    text = ''.join([char for char in text if not emoji.is_emoji(char)])
    
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    text = text.replace("آ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("أ", "ا")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    
    text = text.strip()
    
    return text

data['Text'] = data['Text'].apply(data_cleaning)

data.to_csv("final_file.csv", index=False)

print("Data preprocessing complete. File saved as final_file.csv.")
