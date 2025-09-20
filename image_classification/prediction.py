import torch
from torchvision.io import read_image
from torchvision.models import ResNet50_Weights, resnet50


class ResNetImagePredictor:
    def __init__(self):
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.eval()
        self.preprocess = self.weights.transforms()

    def predict(self, img_path):
        with torch.no_grad():
            img = read_image(img_path)
            batch = self.preprocess(img).unsqueeze(0)
            prediction = self.model(batch).squeeze(0).softmax(0)
            results = [
                f"{self.weights.meta['categories'][i]}: " + f"{100 * v:.1f}%"
                for v, i in zip(*torch.topk(prediction, 3))
            ]
            results = " | ".join(results)

            return results
