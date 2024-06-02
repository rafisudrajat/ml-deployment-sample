import torch
import model
import Utils
from PIL import Image


def load_model(trained_weight_path: str) -> model.SimpleCNN:
    model_instance = model.SimpleCNN()
    model_instance.load_state_dict(torch.load(trained_weight_path))
    return model_instance


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_instance = load_model("artifact/simpleCNN_cat_dog_classifier.pth")
    model_instance.to(DEVICE)
    image = Image.open(r'artifact/sample-data/cat1.jpg')
    # Define image transformation
    transform = Utils.image_transformation()
    image_transformed = transform(image).unsqueeze(0).to(DEVICE)
    # Pass the image through the model
    with torch.no_grad():
        model_instance.eval()
        model_output = model_instance(image_transformed)

    # Interpret the model's output
    _, predicted = torch.max(model_output, 1)
    predicted_class = predicted.item()
    print("Cat" if predicted_class == 0 else "Dog")


if __name__ == "__main__":
    main()
