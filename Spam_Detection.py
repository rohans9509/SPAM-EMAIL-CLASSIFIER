import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("spam.csv")

# Clean data
data = data[['v1','v2']]
data.columns = ['Category','Message']
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','spam'], ['Not Spam','Spam'])

# Split
X = data['Message']
y = data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Vectorize
cv = TfidfVectorizer(stop_words='english')
X_train_vec = cv.fit_transform(X_train)


# Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)

# Accuracy
X_test_vec = cv.transform(X_test)
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

lr_pred = lr_model.predict(X_test_vec)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Predict
def predict(message, model_choice):
    message_vec = cv.transform([message])

    if model_choice == "Naive Bayes":
        result = model.predict(message_vec)[0]
        confidence = max(model.predict_proba(message_vec)[0])
    else:
        result = lr_model.predict(message_vec)[0]
        confidence = max(lr_model.predict_proba(message_vec)[0])

    return result, confidence


def get_top_words(message):
    import re
    words = re.findall(r'\b\w+\b', message.lower())
    spam_keywords = ["free", "win", "winner", "lottery", "urgent", "claim", "prize", "cash"]

    found = [word for word in words if word in spam_keywords]
    return found

# ================= UI =================
st.title('🚀 AI Spam Message Detector')
st.caption("Detect spam messages using Machine Learning & NLP")
st.write(f"Naive Bayes Accuracy: {round(accuracy*100, 2)}%")
st.write(f"Logistic Regression Accuracy: {round(lr_accuracy*100, 2)}%")
st.markdown("---")


st.subheader("Enter your message below")
model_choice = st.selectbox("Choose Model", ["Naive Bayes", "Logistic Regression"])
input_msg = st.text_input('Enter message here:')


if st.button('Validate'):
    if input_msg.strip() != "":
        output, conf = predict(input_msg, model_choice)

        if output == "Spam":
            st.error("🚨 This is a Spam Message")
        else:
            st.success("✅ This is Not Spam")

        st.progress(conf)
        st.info(f"Confidence: {round(conf*100, 2)}%")
        keywords = get_top_words(input_msg)
        if len(keywords) > 0:
            st.write("⚠️ Spam Trigger Words:", keywords)
        
    else:
        st.warning("Please enter a message")
