from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from api.predict import preprocess_image, load_best_model, predict

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
async def create_upload_file(file: UploadFile = File(...)):
    # Check if file is an image
    # if not file.content_type.startswith("image/"):
    #     raise HTTPException(status_code=400, detail="File must be an image")

    """
    This endpoint accepts an image file upload.
    It saves the uploaded image to a local file.
    """
    # file path to save the uploaded image
    # print(file)
    file_path = f"uploaded_{file.filename}"
    # print(file_path)
    # import ipdb; ipdb.set_trace()    
    with open(file_path, "wb") as test_img:
        # saving img to file test_img
        shutil.copyfileobj(file.file, test_img)
            
    # return {"filename": file.filename, "message": "Image uploaded successfully!"}
    prediction = get_predict(file_path) # dict
    return prediction

def get_predict(img_path):
    img_array = preprocess_image(img_path)
    model = load_best_model()
    return predict(model, img_array)

@app.get("/")
def root():
    return {"message": "Welcome to the Bee-ector!"}
