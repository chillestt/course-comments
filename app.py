import streamlit as st
import pandas as pd
import numpy as np
import torch 

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import altair as alt

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

## ----------- helper functions ---------------

def classify(text):
    encoded_text = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**encoded_text)
        logits = outputs.logits.squeeze(0) 

    proba = torch.nn.functional.softmax(logits, dim=0)
    return proba


def main():
    st.title("Course comment sentiment analysis")
    st.subheader("Detect emotions in text")

    with st.form(key="my_form"):
        raw_text = st.text_area("Type here")
        submit_text = st.form_submit_button(label="submit")
    
    if submit_text:
        col1, col2 = st.columns(2)

        predicted_proba = classify(raw_text)

        with col1:
            st.success("Original text")
            st.write(raw_text)

            st.success("Prediction")
            predicted_label = labels[torch.argmax(predicted_proba).item()]
            st.write(f"{predicted_label}")

            torch.argmax(predicted_proba)
            st.write(f"Confidence: {predicted_proba[torch.argmax(predicted_proba).item()]}")
        
        with col2:
            st.success("Prediction probability")
            
            proba_df = pd.DataFrame(
                predicted_proba.numpy().reshape(1, -1), columns=labels.items()
            )
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["id", "type", "proba"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='type', y='proba', color='type')
            st.altair_chart(fig, use_container_width=True)
        
if __name__ == "__main__":
    main()