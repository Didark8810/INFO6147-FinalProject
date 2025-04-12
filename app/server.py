import io
import base64
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import matplotlib.cm as cm
import torch.nn.functional as F
import uvicorn

from models.detr import DETRdemo
from models.classifier import load_model_and_labels
from models.gradcam import generate_gradcam
from models.classifierCNNModel import FruitClassifierCNNModel
from utils.coco_classes import CLASSES
from utils.detection_utils import detect, plot_results_to_image, transform as tdTales

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "models_trained/efficientnet_finetuned.pth"
label_path = "models_trained/class_labels.txt"
model, class_names = load_model_and_labels(model_path, label_path)
model.to(device)
model.eval()

modelDetect = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    'https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
modelDetect.load_state_dict(state_dict)
modelDetect.to(device)
modelDetect.eval()

class_namesBasic = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
model_pathBasic = "models_trained/modelBasic.pth"
modelBasic = FruitClassifierCNNModel(num_classes=len(class_namesBasic))
modelBasic.load_state_dict(torch.load(model_pathBasic, map_location=device))
modelBasic.to(device)
modelBasic.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transformBasic = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

def pil_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = torch.argmax(probs).item()
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx].item()

    heatmap_np = generate_gradcam(model, input_tensor, target_layer="features", class_idx=pred_idx)

    heatmap_color = cm.jet(heatmap_np)[..., :3] * 255
    heatmap_color = heatmap_color.astype(np.uint8)
    overlay = Image.blend(Image.fromarray(np.array(image.resize((224, 224)))), Image.fromarray(heatmap_color), alpha=0.5)

    original_b64 = pil_to_base64(image.resize((224, 224)))
    heatmap_b64 = pil_to_base64(overlay)

    return JSONResponse({
        "class": pred_class,
        "confidence": round(confidence * 100, 2),
        "original_image": original_b64,
        "gradcam_image": heatmap_b64
    })

@app.post("/detect")
async def detect_banana(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    scores1, boxes1 = detect(image, modelDetect, tdTales)
    detection_img_buf = plot_results_to_image(image, scores1, boxes1)
    img_base64 = base64.b64encode(detection_img_buf.getvalue()).decode()
    original_b64 = pil_to_base64(image.resize((224, 224)))

    result = []
    for score, box in zip(scores1, boxes1):
        label = CLASSES[score.argmax()]
        result.append({
            "label": label,
            "score": float(score.max()),
            "box": [float(b) for b in box]
        })

    return JSONResponse({
        "detections": result,
        "detection_image": img_base64,
        "original_image": original_b64,
    })

@app.post("/predictBasic")
async def predictBasic(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transformBasic(image).unsqueeze(0).to(device)
    original_b64 = pil_to_base64(image.resize((224, 224)))

    with torch.no_grad():
        output = modelBasic(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        label = class_namesBasic[predicted.item()]
        confidence_score = confidence.item() 
    
    return JSONResponse({
        "label": label,
        "original_image": original_b64,
        "confidence": round(confidence_score * 100, 2),
    })

def start_server():
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000)
