import torch
from models.model import CustomResNet50, CustomEfficientNet
from utils.data_preprocessing import val_transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_height(image_path, model, transform):
    model.eval()
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    return output.item()

# Load your model
model = CustomResNet50()  # Or CustomEfficientNet()
model.load_state_dict(torch.load('models/best_model.pth'))
model.to(device)

# Use the model for prediction
new_image_path = 'path/to/new_image.jpg'
predicted_height = predict_height(new_image_path, model, val_transforms)
print(f"Predicted liquid height: {predicted_height:.2f}")
