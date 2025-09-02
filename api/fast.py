from fastapi import FastAPI, UploadFile, File
import shutil
from fastapi.middleware.cors import CORSMiddleware

# import predict

app = FastAPI()

# app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
@app.post("/upload_image/")
def create_upload_file(img: UploadFile = File(...)):
    """
    This endpoint accepts an image file upload.
    It saves the uploaded image to a local file.
    """
    # file path to save the uploaded image
    file_path = f"uploaded_{img.filename}"
        
    with open(file_path, "wb") as test_img:
        # saving img to file test_img
        shutil.copyfileobj(image.file, test_img)
            
    return {"filename": img.filename, "message": "Image uploaded successfully!"}
    

@app.get("/")
def root():
    return {"message": "Welcome to the Bee-ector!"}
