from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class EmailPhishingDetection:
    def __init__(self, model_name: str, api_key: str = None):
        """
        Initialize the phishing detection model with a Hugging Face model.
        Args:
            model_name (str): Hugging Face model identifier.
            api_key (str, optional): Hugging Face API key if private model is used.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_key)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, token=api_key)
        self.label_map = {
            0: 'legitimate_email',
            1: 'phishing_email',
            2: 'legitimate_url',
            3: 'phishing_url'
        }

    def predict(self, contents: list[str]) -> list[dict]:
        """
        Predict whether contents are legitimate or phishing with probabilities.
        Args:
            contents (list[str]): List of email/URL texts to classify.
        Returns:
            list[dict]: List of results with classification details and probabilities.
        """
        results = []
        for content in contents:
            # Tokenize and preprocess input
            inputs = self.tokenizer(
                content,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()

            # Map predictions to labels
            label_probs = {self.label_map[i]: round(prob, 2) for i, prob in enumerate(probs)}
            max_label = max(label_probs.items(), key=lambda x: x[1])

            results.append({
                "content": content,
                "prediction": max_label[0],
                "confidence": round(max_label[1], 2),
                "all_probabilities": label_probs
            })
        return results


# Usage Example
if __name__ == "__main__":
    # Use the specified Hugging Face model and your API token
    model_name = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
    api_key = "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"  # Your Hugging Face API token

    phishing_detector = EmailPhishingDetection(model_name, api_key)

    test_contents = [
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

    predictions = phishing_detector.predict(test_contents)
    for result in predictions:
        print(f"Email-Content: {result['content']}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
        print("All probabilities:")
        for label, prob in result['all_probabilities'].items():
            print(f"  {label}: {prob:.2f}")
        print("\n")
