import torch
import matplotlib.pyplot as plt
from PIL import Image
from model import KeypointCNN
from torchvision import transforms
import numpy as np

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_keypoints(image_path, model_path="saved_models/keypoint_model.pth"):
    device = torch.device("cpu")
    model = KeypointCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    image = Image.open(image_path).convert("L")
    transformed = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(transformed)
        keypoints = output.view(-1, 2).cpu().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=20)
    plt.title("Predicted Keypoints")
    plt.axis('off')
    plt.show()