import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
import io
import openai
from dotenv import load_dotenv
import os
import time


load_dotenv()


print(st.secrets["openai"]["OPENAI_API_KEY"])

openai_client = openai.Client(api_key=st.secrets["openai"]["OPENAI_API_KEY"])

credentials_dict = {
    "type": st.secrets["google_cloud"]["GOOGLE_CREDENTIALS_TYPE"],
    "project_id": st.secrets["google_cloud"]["GOOGLE_PROJECT_ID"],
    "private_key_id": st.secrets["google_cloud"]["GOOGLE_PRIVATE_KEY_ID"],
    "private_key": st.secrets["google_cloud"]["GOOGLE_PRIVATE_KEY"].replace('\\n', '\n'),
    "client_email": st.secrets["google_cloud"]["GOOGLE_CLIENT_EMAIL"],
    "client_id": st.secrets["google_cloud"]["GOOGLE_CLIENT_ID"],
    "auth_uri": st.secrets["google_cloud"]["GOOGLE_AUTH_URI"],
    "token_uri": st.secrets["google_cloud"]["GOOGLE_TOKEN_URI"],
    "auth_provider_x509_cert_url": st.secrets["google_cloud"]["GOOGLE_AUTH_PROVIDER_X509_CERT_URL"],
    "client_x509_cert_url": st.secrets["google_cloud"]["GOOGLE_CLIENT_X509_CERT_URL"]
}

credentials = service_account.Credentials.from_service_account_info(credentials_dict)

client = vision.ImageAnnotatorClient(credentials=credentials)


def generate_response(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output a optimized JSON for medical documents, if the image is not related to medical documents, return a json with a negative response, like 'valid': not valid."},
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
    st.title("Upload image for OCR and GPT-3.5 Turbo response")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        start_time = time.time()  # Start time measurement

        image_bytes = uploaded_file.getvalue()
        ocr_text = detect_text(image_bytes)

        ocr_time = time.time()  # Time after OCR is done

        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("Detected Text:")
        st.write(ocr_text)

        gpt_response = generate_response(ocr_text)

        gpt_time = time.time()  # Time after GPT response is done

        st.write("gpt-3.5-turbo-1106 Response:")
        st.write(gpt_response)

        # Calculate and print the time taken for OCR and GPT-3.5 Turbo response
        ocr_duration = ocr_time - start_time
        gpt_duration = gpt_time - ocr_time
        total_duration = gpt_time - start_time

        st.write(f"OCR Time: {ocr_duration:.2f} seconds")
        st.write(f"GPT-3.5 Turbo Response Time: {gpt_duration:.2f} seconds")
        st.write(f"Total Operation Time: {total_duration:.2f} seconds")

if __name__ == "__main__":
    main()
