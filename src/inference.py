import torch
from .model import SimpleCNN
from . import utils
from PIL import Image
import torch.backends.cudnn as cudnn


def inference(model: SimpleCNN, image: Image) -> str:
    # Define image transformation
    cudnn.benchmark = True
    transform = utils.image_transformation()
    image_transformed = transform(image).unsqueeze(0).to(utils.DEVICE)
    # Pass the image through the model
    with torch.no_grad():
        model.eval()
        model_output = model(image_transformed)
    # Interpret the model's output
    _, predicted = torch.max(model_output, 1)
    predicted_class = predicted.item()
    return "Cat" if predicted_class == 0 else "Dog"
