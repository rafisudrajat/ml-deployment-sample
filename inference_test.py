import Utils
from PIL import Image
from inference import inference
from model import SimpleCNN
import unittest


class TestInferenceMethod(unittest.TestCase):

    def setUp(self):
        self.model = Utils.load_model(
            'artifact/simpleCNN_cat_dog_classifier.pth')

    # When image input is a cat image, should return "Cat" as image label
    def test_input_cat_should_return_cat(self):
        image = Image.open(r'artifact/sample-data/cat1.jpg')

        result = inference(self.model, image)

        self.assertEqual("Cat", result)

    # When image input is a dog image, should return "Dog" as image label
    def test_input_dog_should_return_dog(self):
        image = Image.open(r'artifact/sample-data/dog1.jpeg')

        result = inference(self.model, image)

        self.assertEqual("Dog", result)


if __name__ == "__main__":
    unittest.main()
