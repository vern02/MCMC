import os
import cv2
import joblib
import numpy as np
import streamlit as st
from streamlit_extras.let_it_rain import rain
from streamlit_option_menu import option_menu

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "NAVIGATIONS",
        ['Introduction', 'QR Phishing Detection', 'E-Text Phishing Detection'],
        menu_icon='browser-safari',
        icons=['house-fill', 'qr-code', 'envelope-fill'],
        default_index=0,
        styles={
            "nav-link": {"font-weight": "bold"},
            "nav-link-selected": {"font-weight": "bold"}
        }
    )

# Load models and vectorizers
tfidf_path = "https://github.com/vern02/MCMC/blob/main/TfIdf_Vectorizer.joblib"
logreg_path = "https://github.com/vern02/MCMC/blob/main/Logistics_Regression_Model.joblib"

if os.path.exists(tfidf_path):
    vectorizer = joblib.load(tfidf_path)
else:
    st.error("TF-IDF vectorizer file not found!")

if os.path.exists(logreg_path):
    best_model = joblib.load(logreg_path)
else:
    st.error("Logistics Regression model file not found!")

# Function to predict email text
def predict_email_text(text):
    text_vectorized = vectorizer.transform([text])
    prediction = best_model.predict(text_vectorized)
    return prediction

# Function to decode QR code
def decode_qr_code(image):
    qr_detector = cv2.QRCodeDetector()
    data, bbox, _ = qr_detector.detectAndDecode(image)
    return data if data else None

# Function to extract features from URL (placeholder)
def extract_url_features(url):
    return [len(url)]  # Placeholder for demonstration

# Section 1: Introduction
if selected == 'Introduction':
    st.title("PhishÜp")
    st.write("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    st.markdown("## ¡ Protect Yourself from Phishing Attacks 🎣 !")

    st.write("""
        Phishing is a fraudulent attempt to obtain sensitive information like usernames, passwords, and credit card details, often disguised as trustworthy entities.
        With phishing incidents on the rise, it’s crucial to recognize and avoid them. This app empowers you to detect phishing threats in both QR codes and email text.
    """)

    st.markdown("### Why Phishing Matters")
    st.write("""
        - Phishing is responsible for over **36% of data breaches** globally.
        - Attackers increasingly use clever tactics, making phishing harder to detect.
        - Emails and QR codes are common vectors for phishing attacks.
    """)

    st.write("Explore the detection tools in this app and **protect your digital identity** from phishing threats.")
    st.image("/Users/vernsin/Documents/Datathon/PhishingAlertBanner.png", caption="Don't take the bait!")

# Section 2: QR Code Phishing Detection
elif selected == 'QR Phishing Detection':
    st.title("QR Code Phishing Detection")

    input_method = st.selectbox("Choose Image Input Method", ("Please Select", "Use Camera", "Upload Image"))

    if input_method == "Use Camera":
        # Camera input for live capture
        camera_input = st.camera_input("Capture a QR Code image")

        if camera_input:
            # Decode the captured image from the camera
            img = cv2.imdecode(np.frombuffer(camera_input.getvalue(), np.uint8), cv2.IMREAD_COLOR)

            url = decode_qr_code(img)
            if url:
                st.write(f"Captured QR Code contains URL: {url}")
                features = extract_url_features(url)
                prediction = best_model.predict([features])[0]

                if prediction == 1:
                    st.error("⚠️ Phishing URL detected!")
                else:
                    st.success("✅ This URL is safe.")
            else:
                st.warning("No valid QR code detected in the captured image.")

    elif input_method == "Upload Image":
        uploaded_files = st.file_uploader("Upload QR Code images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                img = np.frombuffer(uploaded_file.read(), np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)

                url = decode_qr_code(img)
                if url:
                    st.write(f"QR Code in {uploaded_file.name} contains URL: {url}")
                    features = extract_url_features(url)
                    prediction = best_model.predict([features])[0]

                    if prediction == 1:
                        st.error(f"⚠️ Phishing URL detected in {uploaded_file.name}!")
                    else:
                        st.success(f"✅ {uploaded_file.name}: This URL is safe.")
                else:
                    st.warning(f"No valid QR code detected in {uploaded_file.name}.")

# Section 3: Email Text Phishing Detection
elif selected == 'E-Text Phishing Detection':
    st.title("Email Text Phishing Detection")
    with st.form("email_form"):
        user_input = st.text_area("Enter an email text to check if it's phishing:")
        submitted = st.form_submit_button("Predict")

        if submitted and user_input.strip():
            sentiment = predict_email_text(user_input)

            if sentiment == 1:
                st.warning("!! The given text is a phishing email !! 🎣")
                rain(emoji="🎣", font_size=20, falling_speed=2, animation_length="infinite")
            else:
                st.success("The given text is a safe email!")
                rain(emoji="✅", font_size=20, falling_speed=2, animation_length="infinite")
        elif submitted:
            st.warning("Please enter a text before predicting.")
