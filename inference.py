import torch
import model

def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_instance = model.SimpleCNN()
    model_instance.load_state_dict(torch.load("artifact/random_model_cat_dog.pth"))
    model_instance.to(DEVICE)

if __name__ == "__main__":
    main()