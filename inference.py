import torch
from model import SimpleCNN
import Utils
from PIL import Image


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def inference(model: SimpleCNN, image: Image) -> str:
    # Define image transformation
    transform = Utils.image_transformation()
    image_transformed = transform(image).unsqueeze(0).to(DEVICE)
    # Pass the image through the model
    with torch.no_grad():
        model.eval()
        model_output = model(image_transformed)
    # Interpret the model's output
    _, predicted = torch.max(model_output, 1)
    predicted_class = predicted.item()
    return "Cat" if predicted_class == 0 else "Dog"

# Unit test


def main():
    model = Utils.load_model('artifact/simpleCNN_cat_dog_classifier.pth')
    test_cat1(model)
    test_dog1(model)


def test_dog1(model: SimpleCNN) -> None:
    image = Image.open(r'artifact/sample-data/dog1.jpeg')
    result = inference(model, image)
    assert ("Dog" == result)


def test_cat1(model: SimpleCNN) -> None:
    image = Image.open(r'artifact/sample-data/cat1.jpg')
    result = inference(model, image)
    assert ("Cat" == result)


if __name__ == "__main__":
    main()
