from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import shutil
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from bee_tector.pipeline import run_pipeline

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
async def create_upload_file(files: UploadFile = File(...), data: str = Form(None)):
    # Check if file is an image
    # if not file.content_type.startswith("image/"):
    #     raise HTTPException(status_code=400, detail="File must be an image")

    """
    This endpoint accepts an image file upload.
    It saves the uploaded image to a local file.
    """
    # file path to save the uploaded image
    # print(file)
    file_path = f"uploaded_{files.filename}"
    # print(file_path)
    # import ipdb; ipdb.set_trace()
    with open(file_path, "wb") as test_img:
        # saving img to file test_img
        shutil.copyfileobj(files.file, test_img)
    print("test")
    # return {"filename": file.filename, "message": "Image uploaded successfully!"}
    prediction = run_pipeline(file_path, country_code = data) # dict
    return prediction



@app.get("/")
def root():
    return {"message": "Welcome to the Bee-ector!"}
