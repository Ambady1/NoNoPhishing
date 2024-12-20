import streamlit as st
from tensorflow.keras.models import load_model  # type: ignore[import]
from tensorflow.keras.utils import pad_sequences  # type: ignore[import]
import os
from gensim.models.doc2vec import Doc2Vec
from bs4 import BeautifulSoup
import numpy as np
import re
import joblib

# Load Models
urlnet_model = load_model("urlnet_model.h5")
htmlphish_model = load_model("htmlphish_model.h5")
dom2vec_model = Doc2Vec.load("dom2vec_model")
stacking_model = joblib.load("stacking_model.pkl")

# Preprocess URL Function


def preprocess_url(urls, max_length=200):
    ascii_sequences = [[ord(char) for char in re.sub(
        r'[^a-zA-Z0-9\-_]', '', url)] for url in urls]
    return pad_sequences(ascii_sequences, maxlen=max_length, padding='post', truncating='post')


# Preprocess HTML Function

def preprocess_html(html_content, max_length=500):
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text(separator=" ").strip()
    tokens = re.findall(r'\w+', text_content)
    tokenized = [hash(word) % 10000 for word in tokens]
    return pad_sequences([tokenized], maxlen=max_length, padding='post', truncating='post')


# Streamlit App
def predict_phishing(url, uploaded_file):
    """
    Streamlit-based prediction: Determines if a website is phishing or legitimate.
    """
    # Preprocess URL
    url_sequence = preprocess_url([url])
    pred_urlnet = urlnet_model.predict(url_sequence).flatten()[0]

    # Preprocess HTML content
    html_content = uploaded_file.getvalue().decode("utf-8")
    html_sequence = preprocess_html(html_content)
    pred_htmlphish = htmlphish_model.predict(html_sequence).flatten()[0]

    # Extract DOM Tree Structure
    soup = BeautifulSoup(html_content, "html.parser")
    dom_tree = " ".join(tag.name for tag in soup.find_all(True))
    dom_embedding = dom2vec_model.infer_vector(dom_tree.split())

    # Combine Predictions for Stacking Model
    pred_urlnet = np.array([pred_urlnet])
    pred_htmlphish = np.array([pred_htmlphish])
    dom_embedding = dom_embedding.reshape(1, -1)
    combined_input = np.column_stack(
        [pred_urlnet, pred_htmlphish, dom_embedding])
    final_prediction = stacking_model.predict(combined_input)
    print(final_prediction[0])

    # Output the result
    result = "Phishing" if final_prediction[0] == 1 else "Legitimate"
    if result == "Phishing":
        st.markdown(f"<h1 style='text-align: center; color: red;'>🚨 {result.upper()} 🚨</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: red;'>Dont click that link, Bad boys tryna fool you</h2>", unsafe_allow_html=True)

    else:
        st.markdown(f"<h1 style='text-align: center; color: green;'>✅ {result.upper()} ✅</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: green;'>Phewww!!! This one's safe</h2>", unsafe_allow_html=True)




# Streamlit App Main Code
st.title("NoNoPhishing🚫")
st.write("Enter a URL and upload the HTML file to determine if it's phishing or legitimate.")

# Step 1: Input a URL
url = st.text_input("Enter the URL:")

# Step 2: Upload an HTML file
uploaded_file = st.file_uploader("Upload the HTML file:", type=["html"])

# Step 3: Trigger Prediction
if st.button("Check for Phishing"):
    if url and uploaded_file:
        # result, pred_urlnet, pred_htmlphish, dom_tree = predict_phishing(url, uploaded_file)
        predict_phishing(url, uploaded_file)
    else:
        st.warning("Please provide both a URL and an HTML file.")
