import streamlit as st
import requests

# Set the URL of your FastAPI server
API_URL = "http://127.0.0.1:8000/upload_image/"

st.title("Image Uploader to Bee-Tector's api")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Image uploaded successfully!")
    
    uploaded_file.seek(0)
    # Prepare the file data to be sent
    # import ipdb; ipdb.set_trace()
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

    # Send a POST request to the FastAPI endpoint
    response = requests.post(API_URL, files= files)
    # response = requests.get("http://127.0.0.1:8000/").json()
    # print(response)

    # Check the response from the server
    if response.status_code == 200:
        st.success("File uploaded to FastAPI successfully!")
        st.json(response.json())
    else:
        st.error(f"Failed to upload file to FastAPI. Status code: {response.status_code}")
        st.write(response.text)