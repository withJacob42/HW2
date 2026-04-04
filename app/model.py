import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import json

class FoodClassifier:
    def __init__(self):
        # We use a lightweight pre-trained model (MobileNet V2)
        # It's trained on ImageNet, which contains several food categories.
        # In a real MLOps pipeline, this is where you'd load your fine-tuned weights.
        self.weights = models.MobileNet_V2_Weights.DEFAULT
        self.model = models.mobilenet_v2(weights=self.weights)
        self.model.eval()
        
        # Extract ImageNet classes
        self.categories = self.weights.meta["categories"]
        
        # Preprocessing pipelne for MobileNet V2
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_bytes: bytes):
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        # Move to appropriate device (CPU by default here for simple server)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        input_batch = input_batch.to(device)

        with torch.no_grad():
            output = self.model(input_batch)

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top prediction
        top_prob, top_catid = torch.topk(probabilities, 1)
        
        class_name = self.categories[top_catid[0].item()]
        confidence = top_prob[0].item()

        return {
            "class": class_name,
            "confidence": confidence
        }

# Instantiate once so it remains in memory across requests
classifier = FoodClassifier()
