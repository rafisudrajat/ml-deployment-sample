import torch
from torchvision import transforms

# Define the transformation
DEFAULT_IMG_SIZE = 256


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
