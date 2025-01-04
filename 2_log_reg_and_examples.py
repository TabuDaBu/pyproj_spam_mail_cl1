import pandas as pd
from sklearn.linear_model import LogisticRegression

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
    return model

def format_email_with_counts(example_email, feature_names):
    """
    Wandelt einen E-Mail-String in ein numerisches Feature-Format um,
    wobei die Häufigkeit jedes Wortes berücksichtigt wird.
    """
    features = [0] * len(feature_names)  # Initialisiere alle Features mit 0

    # Splitte die Wörter und zähle Vorkommen
    for word in example_email.split():
        word = word.lower()  # Kleinbuchstaben für Konsistenz
        if word in feature_names:
            index = feature_names.index(word)
            features[index] += 1  # Erhöhe den Zähler für jedes Vorkommen

    return pd.DataFrame([features], columns=feature_names)


def test_example_emails_with_score(model, example_emails, feature_names):
    """
    Testet mehrere Beispiel-E-Mails und gibt die Klassifikation sowie die Wahrscheinlichkeit für jede E-Mail aus.
    """
    for email in example_emails:
        example_features = format_email_with_counts(email, feature_names)
        probabilities = model.predict_proba(example_features)[0]
        spam_score = probabilities[1]  # Wahrscheinlichkeit für Spam
        ham_score = probabilities[0]  # Wahrscheinlichkeit für Ham

        print(f"\nE-Mail: {email}")
        print(f"Wahrscheinlichkeit - Spam: {spam_score:.4f}, Kein Spam: {ham_score:.4f}")

def main():
    # Pfade zu den Datensätzen anpassen
    train_dataset_path = 'data/phishing_train_processed.csv'
    #test_dataset_path = 'data/phishing_test_processed.csv'

    # Lade die Datensätze
    train_data = pd.read_csv(train_dataset_path)
    #test_data = pd.read_csv(test_dataset_path)

    # Daten vorbereiten
    X_train, y_train = preprocess_data(train_data)
    #X_test, y_test = preprocess_data(test_data)

    # Logistische Regression trainieren
    lr_model = train_logistic_regression(X_train, y_train)

    # Feature-Namen extrahieren
    feature_names = X_train.columns.tolist()

    # Beispiel-E-Mails definieren
    example_emails = [
        "hi dan, lunch at the new cafe? let me know.",

        "meeting rescheduled. please confirm your availability by replying to this email.",

        " hi steve, as discussed before, here the link to my youtube channel: https://www.youtube.com/@LinusTechTips see you, regards linus ",

        "https://www.youtube.com",
        
        "https://www.youtube.com/@LinusTechTips",

        """Olá, daniel.buttazoni@gamil.com, 
        Seu pedido foi bloqueado pela fiscalização alfandegária devido à falta de pagamento de taxas obrigatórias. 
        Para garantir o recebimento do seu pedido, por favor, efetue o pagamento o mais breve possível caso contrário seu pedido será cancelado sem reembolso. Clique no botão abaixo para realizar o pagamento: Efetuar Pagamento 
        <https://pedidocorreios.txbrasilcx.co.ua/taxa/taxacorreios951> Correios 2024 - Todos os direitos reservados. 
        Este e-mail foi gerado automaticamente. Por favor, não responda.""",

        """Hallo, daniel.buttazoni@gmail.com,
        Ihre Bestellung wurde von der Zollbehörde blockiert, da die erforderlichen Gebühren nicht bezahlt wurden.
        Um den Erhalt Ihrer Bestellung sicherzustellen, zahlen Sie bitte die Gebühren so schnell wie möglich, ansonsten wird Ihre Bestellung storniert und es erfolgt keine Rückerstattung. Klicken Sie auf die Schaltfläche unten, um die Zahlung vorzunehmen: Zahlung vornehmen
        https://pedidocorreios.txbrasilcx.co.ua/taxa/taxacorreios951 Correios 2024 - Alle Rechte vorbehalten.
        Diese E-Mail wurde automatisch generiert. Bitte antworten Sie nicht darauf.""",

        """Hello, daniel.buttazoni@gmail.com,
        Your order has been blocked by customs due to unpaid required fees.
        To ensure the receipt of your order, please pay the fees as soon as possible; otherwise, your order will be canceled, and no refund will be issued. Click on the button below to make the payment: Make Payment
        https://pedidocorreios.txbrasilcx.co.ua/taxa/taxacorreios951 Correios 2024 - All rights reserved.
        This email was generated automatically. Please do not reply to it.""",

        """Liebe Studierende,
        im Rahmen meiner Bachelorarbeit führe ich eine Studie zum Thema "Vergleich digitaler Werbekampagnen: Storytelling versus Produktwerbung" durch. Ich würde mich sehr über eure Teilnahme freuen.
        Die Durchführung der Studie ist anonym und erfolgt online. Die Dauer beträgt ungefähr 5 Minuten.
        Die Studie findet ihr unter folgenden Link:
        https://survey.aau.at/index.php/116195?lang=de
        Alle Daten, die in der Studie erhoben werden, werden vertraulich behandelt und ausschließlich für wissenschaftliche Zwecke verwendet.
        Bei Fragen/Unklarheiten sowie bei Interesse an den Ergebnissen könnt ihr euch jederzeit per E-Mail (milanmaj@edu.aau.at) an mich wenden.
        Ich freue mich sehr über eure Teilnahme!
        Liebe Grüße
        Milan Majstorovic """,

        """"Dear Students,
        As part of my bachelor's thesis, I am conducting a study on the topic 'Comparison of Digital Advertising Campaigns: Storytelling versus Product Advertising.' I would greatly appreciate your participation.
        The study is conducted anonymously and takes place online. It will take approximately 5 minutes to complete.
        You can find the study at the following link:
        https://survey.aau.at/index.php/116195?lang=de
        All data collected in the study will be treated confidentially and used exclusively for scientific purposes.
        If you have any questions or uncertainties, or if you are interested in the results, you can contact me at any time via email (milanmaj@edu.aau.at).
        I am very much looking forward to your participation!
        Best regards,
        Milan Majstorovic"""

    ]

    # Testen der Beispiel-E-Mails mit Wahrscheinlichkeiten
    test_example_emails_with_score(lr_model, example_emails, feature_names)

if __name__ == "__main__":
    main()
