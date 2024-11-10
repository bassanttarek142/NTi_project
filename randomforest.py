import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preparation import prepare_data
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def main():
    train_df, val_df, test_df = prepare_data("final_file.csv")  # Direct unpacking

    X_train = train_df['cleaned_text']
    y_train = train_df['label']

    X_val = val_df['cleaned_text']
    y_val = val_df['label']

    X_test = test_df['cleaned_text']
    y_test = test_df['label']

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)

    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  

    model.fit(X_train_tfidf, y_train)

    y_val_pred = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    y_test_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_test_pred))

    with open("random_forest_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    with open("tfidf_vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    print("\nModel and vectorizer saved.")

if __name__ == "__main__":
    main()
