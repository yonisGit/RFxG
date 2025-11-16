import metrics
from metrics import compute_ccs,compute_cgc,compute_cgs,compute_pgs
import rfxg_xai
from rfxg_xai import get_by_class_saliency_iia
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

model = torchvision.models.resnet101(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_path = 'sports_car.jpg'
img = Image.open(image_path).convert('RGB')
original_size = img.size

input_tensor = preprocess(img)
img_array = input_tensor.permute(1, 2, 0).numpy()

with torch.no_grad():
    outputs = model(input_tensor.unsqueeze(0).to(device))
    _, predicted_idx = torch.max(outputs, 1)
    predicted_class = predicted_idx.item()

_, indices = torch.topk(outputs, 5)
percentages = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
print("Top 5 predictions:")
for idx in indices[0]:
    print(f"{idx.item()}: {percentages[idx].item():.2f}%")


class_a = 817  # convertible
class_b = 511  # sports car


models = ['densnet', 'convnext', 'resnet101']
layer_options = [12, 8]

model_name = models[2]
FEATURE_LAYER_NUMBER = layer_options[1]

PREV_LAYER = FEATURE_LAYER_NUMBER - 1
num_layers_options = [1]

USE_MASK = True

t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = get_by_class_saliency_dix(image_path=image_path,
                                              label=[class_a,class_b],
                                              operations=['iia'],
                                              model_name=model_name,
                                              layers=[FEATURE_LAYER_NUMBER],
                                              device=device,
                                              use_mask=True)
print(compute_ccs(model,img, blended_img_mask, class_a, class_b))