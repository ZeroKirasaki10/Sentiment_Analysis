import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
with open("sentiment_pipeline.pkl", "rb") as pipeline_file:
    loaded_pipeline = pickle.load(pipeline_file)

def clean_review(review):
    review = review.lower()  
    stop_words = set(stopwords.words('english')) 
    tokens = review.split()  
    filtered_words = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(filtered_words)

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🔎", layout="centered")

st.title("🔎 Sentiment Analysis Web App")
st.write("Enter a review, and I'll predict its sentiment!")

user_input = st.text_area("💬 Type your review below:")

if st.button("Analyze Sentiment"):
    if user_input:
        cleaned_review = clean_review(user_input)  # Apply cleaning
        sentiment_prob = loaded_pipeline.predict_proba([cleaned_review])  # Directly use pipeline
        
        pos_prob = sentiment_prob[0][1]  # Probability of positive
        neg_prob = sentiment_prob[0][0]  # Probability of negative

        threshold = 0.55  

        if abs(pos_prob - neg_prob) < (1 - threshold):  
            sentiment_label = "😐 Neutral"
        elif pos_prob > neg_prob:
            sentiment_label = "😃 Positive"
        else:
            sentiment_label = "😞 Negative"

        st.subheader(f"Prediction: {sentiment_label}")
        st.write(f"Confidence Scores → Positive: {pos_prob:.2f}, Negative: {neg_prob:.2f}")

    else:
        st.warning("⚠ Please enter a review before analyzing.")