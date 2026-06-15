import torch
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet
        std=[0.229, 0.224, 0.225]
    ),
])

img = Image.open('data/img/jura.jpg').convert('RGB')
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0).to(device)
with torch.no_grad():
    outputs = model(img_tensor)
probs = torch.nn.functional.softmax(outputs, dim=1)
top_prob, top_catid = torch.topk(probs, 1)

path = "data/img/imagenet_classes.txt"
classes = [line.strip() for line in open(path).read().splitlines()]
label = classes[top_catid.item()]
confidence = top_prob.item() * 100

print(f"{label} ({confidence:.2f}%)")