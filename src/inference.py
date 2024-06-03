import torch
from .model import SimpleCNN
from . import utils
from PIL import Image


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def inference(model: SimpleCNN, image: Image) -> str:
    # Define image transformation
    transform = utils.image_transformation()
    image_transformed = transform(image).unsqueeze(0).to(DEVICE)
    # Pass the image through the model
    with torch.no_grad():
        model.eval()
        model_output = model(image_transformed)
    # Interpret the model's output
    _, predicted = torch.max(model_output, 1)
    predicted_class = predicted.item()
    return "Cat" if predicted_class == 0 else "Dog"
