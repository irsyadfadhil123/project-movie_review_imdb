import streamlit as st
import numpy as np
import re
import string
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests


# Preprocessing functions
def remove_emoticons(text):
    emoticon_pattern = re.compile("["  
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002702-\U000027B0"  
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoticon_pattern.sub(r'', text)

def preprocessing_text(text):
    # lowercase ngecilin huruf
    text = text.lower()
    # hapus kalimat yang bukan ASCII
    text = re.sub(r'[^\x00-\x7F]', '', text)
    # hapus format url
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'http\S+|www\S+|https\S+|[a-zA-Z0-9.-]+\.(com|id|net|org|co|info|tv|io|xyz)\S*', '', text, flags=re.MULTILINE)
    text = re.sub(r'http\S+|www\S+|https\S+|[a-zA-Z0-9.-]+\.(com|id|net|org|co|info|tv|io|xyz|twitter\.com|instagram\.com|facebook\.com)\s*\S*', '', text, flags=re.MULTILINE)
    # hapus format tag html
    text = re.sub(r'<.*?>', '', text)
    # hapus tanda baca dan nomor
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    # hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    # menghapus karakter berlebih (misalnya loooove harusnya love)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub("[^A-Za-z\s']"," ", text)  # Menghilangkan yang bukan huruf
    text = re.sub("@[A-Za-z0-9_]+"," ", text)  # Menghilangkan mention
    text = re.sub("#[A-Za-z0-9_]+"," ", text)  # Menghilangkan hashtag
    # text = re.sub("rt", " ", text, flags=re.IGNORECASE) #menghilangkan RT
    text = re.sub(r'(\W)\1+', r'\1', text)  # Menghapus tanda baca yang berulang
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus semua tanda baca
    # hapus enter (newline)
    text = re.sub(r'[\r\n]+', ' ', text)  # Mengganti newline dengan spasi
    return text



# Load tokenizer saat testing
with open('tokenizer_final.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Streamlit UI for user input
st.title("Movie IMDb comment Sentiment Analysis")

# Text input for user
input_text = st.text_area("Enter the commentar for sentiment analysis", "")

if st.button("Analyze Sentiment"):
    if input_text:
        # hapus emot
        text_emot_removed = remove_emoticons(input_text)
        
        # Preprocess the text
        processed_text = preprocessing_text(text_emot_removed)

        # tokenizer.fit_on_texts([processed_text])  # Fit tokenizer on the input text
        X_input_seq = tokenizer.texts_to_sequences([processed_text])
        max_len = 274
        X_input_padded = pad_sequences(X_input_seq, maxlen=max_len, padding='post')

        # API Request setup
        API_KEY = "Ve5Q_X3_XxvUt3XDDlDdB8Eq3YtHIrFfCvNFFsehXMSR" #ganti jangan lupa
        token_response = requests.post(
            'https://iam.cloud.ibm.com/identity/token',
            data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'}
        )
        mltoken = token_response.json()["access_token"]
        header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

        # Prepare payload for prediction
        input_data = X_input_padded.tolist()
        payload_scoring = {
            "input_data": [
                {
                    "fields": ["text"],
                    "values": input_data
                }
            ]
        }

        # API Request for prediction #ganti endpoint
        response_scoring = requests.post(
            'https://us-south.ml.cloud.ibm.com/ml/v4/deployments/streamlit_capstone_server/predictions?version=2021-05-01', # endpointnya diganti
            json=payload_scoring,
            headers=header
        )

        # Parse the JSON response
        response_json = response_scoring.json()

        try:
            # Ambil data dari response JSON
            predictions = response_json['predictions'][0]  # Ambil elemen pertama
            values = predictions['values'][0]  # Ambil nilai dari 'values'

            # Extract the data
            probabilities = values[0]  # Probabilitas untuk setiap kelas
            class_prediction = values[1]  # Kelas terprediksi
            full_probabilities = values[2]  # Probabilitas lengkap

            # Tampilkan hasil
            st.subheader("Sentiment Analysis Results")
            sentiment = "Positive" if class_prediction == 1 else "Negative"
            st.write(f"Predicted Sentiment: {sentiment}")
            st.write(f"Probabilities for each class (Negative, Positive): {probabilities}")
        except KeyError as e:
            st.error(f"KeyError: {e}. Check the response structure.")
            st.write("Full response:", response_json)
            st.stop()
