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
        "Hi Team, please find the attached report from today's meeting. Let me know if you have any further questions.",
        "Meeting rescheduled to next Monday at 10 AM.",
        "Meting rescheduld to next Mondy at 10 AM., mandatory atendaynce, outherwirse your are fired",
        "Meting rescheduled to next Monday at 10 AM., mandatory attendance, otherwise your are fired",
        "Reminder: Your package delivery is scheduled for tomorrow. The tracking # is 1024151515, click here to get further information.",
        "Hi Steve, Lunch at the new cafe? Let me know.",
        "Hi you, Exclusive offer: Save 50% on all items. Limited time only!",
        "Exclusive offer: Save 50% on all items. Limited time only!",
        "Your invoice for the last purchase is attached.",
        "Don't miss out on this once-in-a-lifetime opportunity!",
        "We are excited to announce our new product line launching next week. Join us for an exclusive webinar where our team will showcase the latest innovations and features. Register now to secure your spot.",
        "Thank you for your continued support. Our records indicate your subscription will expire soon. Renew now to ensure uninterrupted access to premium services and special member discounts.",
        "Meeting rescheduled. Please confirm your availability by replying to this email.",
        "Congratulations! You've been selected for an exclusive deal. Click here to claim your $1000 gift card instantly. Hurry, offer expires in 24 hours! Don’t miss this once-in-a-lifetime opportunity. Validate your account now to access this reward. Terms and conditions apply. Act fast!",
        "Hi Team,Thank you for attending the meeting today. Attached is the summary document with action items for each department. Please review and let me know if there are any questions or updates needed. Let’s aim to complete all tasks by the end of the week. Best, John",
        """
        Correios, Olá, daniel.buttazoni@hotmail.com, 
        Seu pedido foi bloqueado pela fiscalização alfandegária devido à falta de pagamento de taxas obrigatórias. 
        Para garantir o recebimento do seu pedido, por favor, efetue o pagamento o mais breve possível caso contrário seu pedido será cancelado sem reembolso. Clique no botão abaixo para realizar o pagamento: Efetuar Pagamento 
        <https://pedidocorreios.txbrasilcx.co.ua/taxa/taxacorreios951> Correios 2024 - Todos os direitos reservados. 
        Este e-mail foi gerado automaticamente. Por favor, não responda.
        """,
        "<https://pedidocorreios.txbrasilcx.co.ua/taxa/taxacorreios951>",
        "https://www.youtube.com",
        "https://www.youtube.com/@LinusTechTips",
        """ hi steve, as discussed before, here the link to my youtube channel: https://www.youtube.com/@LinusTechTips see you, regards linus """,
        """Correios, Hallo, daniel.buttazoni@hotmail.com, 
        Ihre Bestellung wurde von der Zollbehörde blockiert, da die erforderlichen Gebühren nicht bezahlt wurden. 
        Um den Erhalt Ihrer Bestellung sicherzustellen, zahlen Sie bitte die Gebühren so schnell wie möglich, 
        anderenfalls wird Ihre Bestellung storniert und es erfolgt keine Rückerstattung. 
        Klicken Sie auf die Schaltfläche unten, um die Zahlung vorzunehmen: Zahlung vornehmen 
        <https://pedidocorreios.txbrasilcx.co.ua/taxa/taxacorreios951> Correios 2024 - Alle Rechte vorbehalten. 
        Diese E-Mail wurde automatisch generiert. Bitte antworten Sie nicht darauf.""",
        
        """
        Liebe Studierende,
        im Rahmen meiner Bachelorarbeit führe ich eine Studie zum Thema "Vergleich digitaler Werbekampagnen: Storytelling versus Produktwerbung" durch. Ich würde mich sehr über eure Teilnahme freuen.
        Die Durchführung der Studie ist anonym und erfolgt online. Die Dauer beträgt ungefähr 5 Minuten.
        Die Studie findet ihr unter folgenden Link:
        https://survey.aau.at/index.php/116195?lang=de
        Alle Daten, die in der Studie erhoben werden, werden vertraulich behandelt und ausschließlich für wissenschaftliche Zwecke verwendet.
        Bei Fragen/Unklarheiten sowie bei Interesse an den Ergebnissen könnt ihr euch jederzeit per E-Mail (milanmaj@edu.aau.at) an mich wenden.
        Ich freue mich sehr über eure Teilnahme!
        Liebe Grüße
        Milan Majstorovic
        """
    ]

    # Testen der Beispiel-E-Mails mit Wahrscheinlichkeiten
    test_example_emails_with_score(lr_model, example_emails, feature_names)

if __name__ == "__main__":
    main()
