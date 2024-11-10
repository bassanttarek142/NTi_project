import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preparation import prepare_data
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_test_pred))

    with open("random_forest_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    print("\nRandom Forest model saved as 'random_forest_model.pkl'.")

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    import xgboost as xgb
    from sklearn.metrics import classification_report, accuracy_score

    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_test_pred))

    with open("xgboost_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    print("\nXGBoost model saved as 'xgboost_model.pkl'.")

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

    # Save the vectorizer for future use
    with open("tfidf_vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    print("\nVectorizer saved as 'tfidf_vectorizer.pkl'.")

    # Train Random Forest
    train_random_forest(X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf, y_test)

    # Train XGBoost
    train_xgboost(X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf, y_test)

    print("\nAll models have been trained and saved.")

if __name__ == "__main__":
    main()
