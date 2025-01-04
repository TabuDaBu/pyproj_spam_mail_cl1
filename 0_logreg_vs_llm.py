import pandas as pd
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def preprocess_data(df):
    if 'Email No.' in df.columns:
        df = df.drop(columns=['Email No.'])

    if 'Prediction' not in df.columns:
        raise ValueError("Der Datensatz muss eine 'Prediction'-Spalte mit 0 (Ham) und 1 (Spam) enthalten.")

    X = df.drop(columns=['Prediction'])
    y = df['Prediction']

    return X, y

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def format_email(example_email, feature_names):
    features = [0] * len(feature_names)
    for word in example_email.split():
        word = word.lower()
        if word in feature_names:
            index = feature_names.index(word)
            features[index] = 1
    return pd.DataFrame([features], columns=feature_names)

class EmailPhishingDetection:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.label_map = {
            0: 'legitimate_email',
            1: 'phishing_email',
            2: 'legitimate_url',
            3: 'phishing_url'
        }

    def predict(self, content: str):
        inputs = self.tokenizer(content, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()

        label_probs = {self.label_map[i]: round(prob, 2) for i, prob in enumerate(probs)}
        max_label = max(label_probs.items(), key=lambda x: x[1])

        return {
            "prediction": max_label[0],
            "confidence": round(max_label[1], 2),
            "all_probabilities": label_probs
        }

def main():
    train_dataset_path = 'data/phishing_train_processed.csv'
    train_data = pd.read_csv(train_dataset_path)
    X_train, y_train = preprocess_data(train_data)
    feature_names = X_train.columns.tolist()
    lr_model = train_logistic_regression(X_train, y_train)

    model_name = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
    llm_detector = EmailPhishingDetection(model_name)

    print("Geben Sie eine E-Mail ein (oder 'exit', um zu beenden):")
    while True:
        user_input = input("Eingabe: ").replace("\n", " ").strip()
        if not user_input:
            print("Leere Eingabe übersprungen.")
            continue
        if user_input.lower() == 'exit':
            print("Programm beendet.")
            break

        lr_features = format_email(user_input, feature_names)
        lr_probabilities = lr_model.predict_proba(lr_features)[0]
        lr_spam = round(lr_probabilities[1], 4)
        lr_ham = round(lr_probabilities[0], 4)

        llm_result = llm_detector.predict(user_input)

        print("\n--- Ergebnisse ---")
        print("Logistische Regression:")
        print(f"  Spam-Wahrscheinlichkeit: {lr_spam}")
        print(f"  Ham-Wahrscheinlichkeit: {lr_ham}")

        print("\nLLM (DistilBERT):")
        print(f"  Vorhersage: {llm_result['prediction']} (Vertrauen: {llm_result['confidence']})")
        print("  Alle Wahrscheinlichkeiten:")
        for label, prob in llm_result['all_probabilities'].items():
            print(f"    {label}: {prob}")
        print("\n")

if __name__ == "__main__":
    main()
