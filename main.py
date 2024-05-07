from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

model_name = "course-review-analysis"
labels = {
    0: "Improvement Suggestions",
    1: "Questions",
    2: "Confusion",
    3: "Support Request",
    4: "Discussion",
    5: "Course Comparison",
    6: "Related Course Suggestions",
    7: "negative",
    8: "positive"
}

## model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def classify(text):
    encoded_text = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**encoded_text)
        logits = outputs.logits.squeeze(0) 

    ## predict proba -> logits
    # predicted_label = labels[torch.argmax(logits).item()]
    # predicted_prob = torch.nn.functional.softmax(logits, dim=0)[torch.argmax(logits).item()]

    return logits # predicted_label, predicted_prob

# Example usage
text = "I think it's extremely terrible!"
#predicted_label, predicted_prob = classify(text)

# print(f"Predicted label: {predicted_label} (probability: {predicted_prob:.4f})")
predicted_proba = torch.nn.functional.softmax(classify(text), dim=0)
print(predicted_proba)
print(predicted_proba[torch.argmax(predicted_proba).item()])

proba_df = pd.DataFrame(
            predicted_proba.numpy().reshape(1, -1), columns=labels.items()
)
proba_df_clean = proba_df.T.reset_index()
proba_df_clean.columns = ["id", "type", "proba"]
print(proba_df_clean)