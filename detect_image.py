from PIL import Image
import torch
from models.detr import DETRdemo
from utils.detection_utils import detect, plot_results, transform

#Cargar modelo

model = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',map_location='cpu', check_hash=True)
model.load_state_dict(state_dict)
model.eval()

# Cargar imagen
im = Image.open("filesJupyter/uno/f6.jpg")

# Detección
scores, boxes = detect(im, model, transform)

# Visualización
plot_results(im, scores, boxes)
