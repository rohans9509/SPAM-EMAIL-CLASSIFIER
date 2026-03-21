# EMAIL-SPAM-CLASSIFIER by @Rohan Singh
# This is a project I made to classify spam emails

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load data
data = pd.read_csv(r"C:/Users/rohan/Downloads/archive/spam.csv", encoding='latin-1')

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
cv = CountVectorizer(stop_words='english')
X_train_vec = cv.fit_transform(X_train)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict
def predict(message):
    message_vec = cv.transform([message])
    return model.predict(message_vec)[0]

# ================= UI =================
st.title('📩 Spam Message Detector')

input_msg = st.text_input('Enter message here:')

if st.button('Validate'):
    if input_msg.strip() != "":
        output = predict(input_msg)

        if output == "Spam":
            st.error(f"🚨 {output}")
        else:
            st.success(f"✅ {output}")
    else:
        st.warning("Please enter a message")
