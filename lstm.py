import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import emoji
import pyarabic.araby as ar
import nltk
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dense, Dropout, 
    GlobalMaxPooling1D, Attention, Conv1D, MaxPooling1D, 
    BatchNormalization, GRU
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from data_preparation import prepare_data  # Ensure data_preparation.py and text_preprocessing.py are in the same directory
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import requests
import gzip
from gensim.models import KeyedVectors

# Download NLTK resources
nltk.download('stopwords')

def download_fasttext_arabic(destination_path='cc.ar.300.vec'):
    if os.path.exists(destination_path):
        print(f"'{destination_path}' already exists.")
        return

    fasttext_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.vec.gz'
    fasttext_zip = 'cc.ar.300.vec.gz'

    print("Downloading FastText Arabic embeddings (Approx. 1.5 GB)... This may take a while.")
    try:
        with requests.get(fasttext_url, stream=True) as response:
            response.raise_for_status()
            with open(fasttext_zip, 'wb') as f:
                total_downloaded = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        total_downloaded += len(chunk)
                        downloaded_mb = total_downloaded / (1024 * 1024)
                        print(f"Downloaded {downloaded_mb:.2f} MB", end='\r')
        print("\nDownload complete.")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        return

    print("Extracting 'cc.ar.300.vec' from the gzip file...")
    try:
        with gzip.open(fasttext_zip, 'rb') as f_in:
            with open(destination_path, 'wb') as f_out:
                for chunk in iter(lambda: f_in.read(1024 * 1024), b''):
                    f_out.write(chunk)
        print("Extraction complete.")
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        return

    # Remove the gzip file after extraction
    try:
        os.remove(fasttext_zip)
        print(f"'{fasttext_zip}' has been removed.")
    except Exception as e:
        print(f"An error occurred while removing the gzip file: {e}")

def verify_embeddings_file(destination_path='cc.ar.300.vec'):
    if not os.path.exists(destination_path):
        print(f"Error: '{destination_path}' does not exist.")
        return False
    file_size = os.path.getsize(destination_path)
    if file_size < 1000000:  # Less than ~1 MB likely indicates a problem
        print(f"Error: '{destination_path}' is too small ({file_size} bytes). It may be corrupted.")
        return False
    print(f"'{destination_path}' is present with size {file_size / (1024 * 1024):.2f} MB.")
    return True

def load_embeddings(file_path='cc.ar.300.vec'):
    embeddings_index = {}
    print("Loading pre-trained FastText Arabic embeddings using Gensim...")
    try:
        embeddings = KeyedVectors.load_word2vec_format(file_path, binary=False, unicode_errors='ignore')
        print(f"Loaded {len(embeddings.key_to_index)} word vectors.")
    except FileNotFoundError:
        print(f"Error: Embeddings file '{file_path}' not found.")
        return embeddings_index
    except Exception as e:
        print(f"An error occurred while loading embeddings: {e}")
        return embeddings_index
    
    # Convert Gensim KeyedVectors to a dictionary
    embeddings_index = {word: embeddings[word] for word in embeddings.key_to_index}
    return embeddings_index

def normalize_word(word):
    word = ar.normalize_hamza(word)
    word = ar.normalize_ligature(word)
    word = ar.strip_tashkeel(word)
    return word

def create_embedding_matrix(tokenizer, embeddings_index, max_num_words=20000, embedding_dim=300):
    embedding_matrix = np.zeros((max_num_words, embedding_dim))
    matched = 0
    for word, i in tokenizer.word_index.items():
        if i >= max_num_words:
            continue
        normalized_word = normalize_word(word)
        embedding_vector = embeddings_index.get(normalized_word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            matched += 1
    print(f"Number of words matched with embeddings: {matched}/{max_num_words}")
    return embedding_matrix

def preprocess_arabic_text(text):
    text = ar.normalize_hamza(text)
    text = ar.normalize_ligature(text)
    text = ar.strip_tashkeel(text)
    return text

def load_and_prepare_data(train, val, test, max_num_words=20000, max_sequence_length=150):
    # Apply preprocessing to all texts
    train['cleaned_text'] = train['cleaned_text'].apply(preprocess_arabic_text)
    val['cleaned_text'] = val['cleaned_text'].apply(preprocess_arabic_text)
    test['cleaned_text'] = test['cleaned_text'].apply(preprocess_arabic_text)
    
    X_train = train['cleaned_text'].astype(str).tolist()
    y_train = train['label'].tolist()
    X_val = val['cleaned_text'].astype(str).tolist()
    y_val = val['label'].tolist()
    X_test = test['cleaned_text'].astype(str).tolist()
    y_test = test['label'].tolist()

    tokenizer = Tokenizer(num_words=max_num_words, oov_token="<OOV>", lower=True)
    tokenizer.fit_on_texts(X_train)
    
    with open("tokenizer.pkl", "wb") as token_file:
        pickle.dump(tokenizer, token_file)

    X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_sequence_length, padding='post', truncating='post')
    X_val_padded = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=max_sequence_length, padding='post', truncating='post')
    X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_sequence_length, padding='post', truncating='post')

    return X_train_padded, np.array(y_train), X_val_padded, np.array(y_val), X_test_padded, np.array(y_test), tokenizer

def build_improved_model(max_num_words, max_sequence_length, num_classes, embedding_dim=300, l2_rate=1e-4):
    inputs = Input(shape=(max_sequence_length,))
    
    # Pre-trained Embedding Layer
    embedding_layer = Embedding(
        input_dim=max_num_words,
        output_dim=embedding_dim,
        input_length=max_sequence_length,
        embeddings_regularizer=l2(l2_rate),
        trainable=False
    )(inputs)  # Set trainable=False to keep embeddings fixed

    # Add a convolutional layer to extract local features
    conv = Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(l2_rate))(embedding_layer)
    conv = BatchNormalization()(conv)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Dropout(0.3)(conv)

    # Stacked Bidirectional LSTM layers without recurrent_dropout
    lstm = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(l2_rate)))(conv)
    lstm = BatchNormalization()(lstm)
    lstm = Dropout(0.3)(lstm)
    lstm = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(l2_rate)))(lstm)
    lstm = BatchNormalization()(lstm)
    lstm = Dropout(0.3)(lstm)

    # Attention Mechanism
    attention = Attention()([lstm, lstm])
    attention = GlobalMaxPooling1D()(attention)

    # Fully Connected Layers
    dense = Dense(256, activation='relu', kernel_regularizer=l2(l2_rate))(attention)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.4)(dense)
    dense = Dense(128, activation='relu', kernel_regularizer=l2(l2_rate))(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.4)(dense)
    
    outputs = Dense(num_classes, activation='softmax')(dense)

    # Optimizer: Trying Nadam for potentially better performance
    optimizer = Nadam(learning_rate=1e-4, clipnorm=1.0)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    DATA_FILE = "final_file.csv"
    MAX_NUM_WORDS = 20000
    MAX_SEQUENCE_LENGTH = 150  # Increased sequence length to capture more context
    NUM_CLASSES = 3
    EMBEDDING_DIM = 300  # Adjusted to match FastText Arabic embeddings
    BATCH_SIZE = 64
    EPOCHS = 30  # Adjusted to balance training time and performance

    print("Preparing data...")
    train, val, test = prepare_data(DATA_FILE)
    print("Data preparation and splitting complete.")
    print("Data cleaning and preprocessing complete.")

    print("Tokenizing and padding data...")
    X_train_padded, y_train, X_val_padded, y_val, X_test_padded, y_test, tokenizer = load_and_prepare_data(
        train, val, test,
        max_num_words=MAX_NUM_WORDS,
        max_sequence_length=MAX_SEQUENCE_LENGTH
    )
    print("Data tokenization and padding complete.")

    # Download and extract FastText Arabic embeddings if not present
    download_fasttext_arabic()

    # Verify the embeddings file
    if not verify_embeddings_file('cc.ar.300.vec'):
        print("Embeddings file verification failed. Exiting.")
        return

    # Load pre-trained FastText Arabic embeddings using Gensim
    embeddings_index = load_embeddings('cc.ar.300.vec')
    
    if len(embeddings_index) == 0:
        print("Error: No word vectors were loaded. Please check the embeddings file.")
        return

    # Create embedding matrix
    print("Creating embedding matrix...")
    embedding_matrix = create_embedding_matrix(tokenizer, embeddings_index, max_num_words=MAX_NUM_WORDS, embedding_dim=EMBEDDING_DIM)
    print("Embedding matrix created.")

    # Optionally, check embedding coverage
    total_words = min(MAX_NUM_WORDS, len(tokenizer.word_index))
    matched_words = np.count_nonzero(np.any(embedding_matrix, axis=1))
    print(f"Embedding matrix covers {matched_words}/{MAX_NUM_WORDS} words ({(matched_words/MAX_NUM_WORDS)*100:.2f}%)")

    print("\nBuilding the model...")
    model = build_improved_model(
        max_num_words=MAX_NUM_WORDS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        l2_rate=1e-4
    )
    # Set the pre-trained embeddings
    model.layers[1].set_weights([embedding_matrix])
    model.layers[1].trainable = False  # Freeze embeddings to prevent overfitting
    print("\nModel Summary:")
    model.summary()

    # Calculate class weights to handle class imbalance
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights}")

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        'best_improved_model.h5', monitor='val_accuracy', save_best_only=True, mode='max'
    )

    # Train Model
    print("\nStarting Training...")
    history = model.fit(
        X_train_padded, y_train,
        validation_data=(X_val_padded, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        class_weight=class_weights,
        verbose=1
    )

    # Load best weights saved by ModelCheckpoint
    model.load_weights('best_improved_model.h5')

    # Evaluate on Test Set
    print("\nEvaluating on Test Set...")
    test_loss, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=1)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # Predictions and Classification Report
    y_pred_probs = model.predict(X_test_padded)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred_classes, digits=4))

    # Plot Accuracy
    plt.figure(figsize=(10,6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()

    # Plot Loss
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Negative', 'Neutral', 'Positive'], 
                yticklabels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Save the Tokenizer
    with open("tokenizer.pkl", "wb") as token_file:
        pickle.dump(tokenizer, token_file)
    print("Tokenizer saved as 'tokenizer.pkl'.")

if __name__ == "__main__":
    main()
