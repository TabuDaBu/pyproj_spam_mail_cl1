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
        Milan Majstorovic""",

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

    predictions = phishing_detector.predict(test_contents)
    for result in predictions:
        print(f"Email-Content: {result['content']}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
        print("All probabilities:")
        for label, prob in result['all_probabilities'].items():
            print(f"  {label}: {prob:.2f}")
        print("\n")
