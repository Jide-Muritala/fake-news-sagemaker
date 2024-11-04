from transformers import BertTokenizer, BertForSequenceClassification
import torch

def load_model(model_path="./fake_news_model"):
    # Load the tokenizer and model from the saved directory
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def preprocess_text(text, tokenizer):
    # Tokenize and convert to tensor
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    return inputs

def predict(text, tokenizer, model):
    # Preprocess the text and get the model's prediction
    inputs = preprocess_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()  # 0 for real, 1 for fake
    return "Fake News" if prediction == 1 else "Real News"

if __name__ == "__main__":
    model_path = "./fake_news_model"
    text = input("Enter news text to classify: ")
    
    # Load model and tokenizer
    tokenizer, model = load_model(model_path)
    
    # Predict and display result
    result = predict(text, tokenizer, model)
    print("Prediction:", result)
