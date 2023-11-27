import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
import io
import openai
from dotenv import load_dotenv
import os


load_dotenv()

openai_client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))


private_key_bytes = os.getenv('GOOGLE_PRIVATE_KEY').replace('\\n', '\n').encode()

credentials_dict = {
    "type": os.getenv('GOOGLE_CREDENTIALS_TYPE'),
    "project_id": os.getenv('GOOGLE_PROJECT_ID'),
    "private_key_id": os.getenv('GOOGLE_PRIVATE_KEY_ID'),
    "private_key": private_key_bytes,
    "client_email": os.getenv('GOOGLE_CLIENT_EMAIL'),
    "client_id": os.getenv('GOOGLE_CLIENT_ID'),
    "auth_uri": os.getenv('GOOGLE_AUTH_URI'),
    "token_uri": os.getenv('GOOGLE_TOKEN_URI'),
    "auth_provider_x509_cert_url": os.getenv('GOOGLE_AUTH_PROVIDER_X509_CERT_URL'),
    "client_x509_cert_url": os.getenv('GOOGLE_CLIENT_X509_CERT_URL')
}

credentials = service_account.Credentials.from_service_account_info(credentials_dict)

client = vision.ImageAnnotatorClient(credentials=credentials)


def generate_response(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output a optimized JSON."},
            {"role": "user", "content": f"{prompt}"}
        ],
    )
    return response.choices[0].message.content


def detect_text(image_bytes):
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else 'No text detected'

def main():
    st.title("OCR with Google Cloud Vision and Streamlit + GPT-4.5 Turbo Integration")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        ocr_text = detect_text(image_bytes)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("Detected Text:")
        st.write(ocr_text)

        gpt_response = generate_response(ocr_text)
        st.write("GPT-4.5 Turbo Response:")
        st.write(gpt_response)

if __name__ == "__main__":
    main()
