from PIL import Image
import torch
from models.detr import DETRdemo
from utils.detection_utils import detect, plot_results, transform


model = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',map_location='cpu', check_hash=True)
model.load_state_dict(state_dict)
model.eval()

#change this url later
im = Image.open("filesJupyter/uno/f7.jpg")

scores, boxes = detect(im, model, transform)
print(boxes)

plot_results(im, scores, boxes)
