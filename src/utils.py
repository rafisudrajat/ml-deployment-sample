import torch
from torchvision import transforms
from .model import SimpleCNN

# Define the transformation
DEFAULT_IMG_SIZE = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(trained_weight_path: str) -> SimpleCNN:
    model_instance = SimpleCNN()
    model_instance.load_state_dict(
        torch.load(
            trained_weight_path,
            map_location=DEVICE))
    model_instance.to(DEVICE)
    return model_instance


class MinMaxNormalization(object):
    def __call__(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        return (tensor - min_val) / (max_val - min_val)


def image_transformation(
        img_widht: int = DEFAULT_IMG_SIZE,
        img_height: int = DEFAULT_IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_widht, img_height)),  # Resize to 256x256
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert PIL image to tensor
        MinMaxNormalization()  # Apply min-max normalization
    ])
