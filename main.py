from model import SimpleCNN
from fastapi import FastAPI, UploadFile, Depends
from Utils import load_model
from typing_extensions import Annotated
from PIL import Image
import io
from inference import inference

app = FastAPI()

# Define a global variable to store the model
model = None


def initialize_model() -> SimpleCNN:
    global model
    if model is None:
        model = load_model('artifact/simpleCNN_cat_dog_classifier.pth')
    return model


@app.post("/inference")
async def running_image_classification(file: UploadFile,
                             model: Annotated[SimpleCNN,
                                              Depends(initialize_model)]):
    # Read the file content
    contents = await file.read()

    # Convert to PIL Image
    image = Image.open(io.BytesIO(contents))
    inference_result = inference(model, image)
    print("file type", type(file.file))
    return {"inference result": inference_result}
