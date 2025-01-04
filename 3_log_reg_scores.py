import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def preprocess_data(df):
    """
    Bereitet die Daten vor, indem irrelevante Spalten entfernt und
    Features sowie Labels extrahiert werden.
    Erwartet, dass die Spalte 'Prediction' die Zielvariable enthält.
    """
    if 'Email No.' in df.columns:
        df = df.drop(columns=['Email No.'])

    if 'Prediction' not in df.columns:
        raise ValueError("Der Datensatz muss eine 'Prediction'-Spalte mit 0 (Ham) und 1 (Spam) enthalten.")

    # Features (X) und Labels (y) trennen
    X = df.drop(columns=['Prediction'])
    y = df['Prediction']

    return X, y

def train_logistic_regression(X_train, y_train):
    """
    Trainiert ein Modell der logistischen Regression mit den Trainingsdaten.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("\nLogistische Regression erfolgreich trainiert!")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluieren des Modells auf den Testdaten.
    Gibt Accuracy, Precision, Recall und F1-Score aus.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nModellbewertung:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

def calculate_ham_spam_percentage(df):
    """
    Berechnet und gibt den Anteil von Ham und Spam im Datensatz als Dezimalzahlen aus.
    """
    label_counts = df['Prediction'].value_counts()

    spam_count = label_counts.get(1, 0)
    ham_count = label_counts.get(0, 0)
    total_count = spam_count + ham_count

    if total_count == 0:
        print("Der Datensatz enthält keine Emails.")
        return

    ham_percentage = ham_count / total_count
    spam_percentage = spam_count / total_count

    print(f"\nAnzahl Ham-Emails: {ham_count} ({ham_percentage:.4f})")
    print(f"Anzahl Spam-Emails: {spam_count} ({spam_percentage:.4f})")

def display_dataset_structure(dataset, dataset_name):
    """
    Zeigt die ersten fünf Zeilen des Datensatzes sowie die Spaltennamen an.
    """
    print(f"\n{dataset_name} - Struktur des Datensatzes:")
    print(dataset.head())

def main():
    # Pfade zu den Datensätzen anpassen
    train_dataset_path = 'data/phishing_train_processed.csv'
    test_dataset_path = 'data/phishing_test_processed.csv'

    # Lade die Datensätze
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    # Anzeigen der Struktur der Datensätze
    display_dataset_structure(train_data, "Trainingsdatensatz")

    # Ham/Spam-Verteilung im Trainingsdatensatz
    print("Verteilung im Trainingsdatensatz:")
    calculate_ham_spam_percentage(train_data)

    # Ham/Spam-Verteilung im Testdatensatz
    print("\nVerteilung im Testdatensatz:")
    calculate_ham_spam_percentage(test_data)

    # Anzahl der Testdatensätze ausgeben
    print(f"\nAnzahl der Testdatensätze: {test_data.shape[0]}")

    # Daten vorbereiten
    X_train, y_train = preprocess_data(train_data)
    X_test, y_test = preprocess_data(test_data)

    # Logistische Regression trainieren und bewerten
    lr_model = train_logistic_regression(X_train, y_train)
    print("\nBewertung des Modells der logistischen Regression:")
    evaluate_model(lr_model, X_test, y_test)

if __name__ == "__main__":
    main()

