import streamlit as st
import joblib
import re

# Load the saved model, vectorizer, and label encoder
model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

# Define text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  
    text = re.sub(r"@\w+|\#", "", text)  
    text = re.sub(r"[^\w\s]", "", text)  
    text = re.sub(r"\d+", "", text)  
    text = " ".join(text.split()) 
    return text

# Define emotion prediction function
def predict_emotion(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return encoder.inverse_transform(prediction)[0]

# Streamlit UI
st.title("Twitter Emotion Sentiment Analysis")
st.write("Enter a tweet below to predict its emotion.")
st.subheader("Example tweets to try:")

st.markdown("> 1. I feel scared and anxious about the future.")
st.markdown("> 2. I love spending time with my family.")
st.markdown("> 3. I feel so alone and lost right now.")
st.markdown("> 4. Life has its ups and downs, you just deal with it.")
st.markdown("> 5. I love using this new AI tool! It's amazing.")

# User input
user_input = st.text_area("Enter a tweet:", "")

if st.button("Predict Emotion"):
    if user_input:
        predicted_emotion = predict_emotion(user_input)
        st.success(f"Predicted Emotion: **{predicted_emotion}**")
    else:
        st.warning("Please enter a tweet to analyze.")
