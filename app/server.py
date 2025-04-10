# server.py

import io
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from models.gradcam import generate_gradcam
from classifier import load_model_and_labels
import numpy as np
import base64
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import uvicorn

app = FastAPI()

# Permite acceso desde cualquier origen (útil para pruebas móviles o frontend web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga del modelo entrenado
model_path = "models/efficientnet_finetuned.pth"
label_path = "models/class_labels.txt"
model, class_names = load_model_and_labels(model_path, label_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Transformaciones para las imágenes recibidas
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def pil_to_base64(pil_img):
    """Convierte una imagen PIL a base64 para mostrar en HTML."""
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

    # Generar gradCAM
    heatmap_np = generate_gradcam(model, input_tensor, target_layer="features", class_idx=pred_idx)

    # Superponer gradCAM sobre imagen original
    heatmap_img = Image.fromarray(np.uint8(heatmap_np * 255))
    image_np = np.array(image.resize((224, 224)))
    heatmap_color = cm.jet(heatmap_np)[..., :3] * 255
    heatmap_color = heatmap_color.astype(np.uint8)
    overlay = Image.blend(Image.fromarray(image_np), Image.fromarray(heatmap_color), alpha=0.5)

    # Convertimos a base64 para enviar al navegador
    original_b64 = pil_to_base64(image.resize((224, 224)))
    heatmap_b64 = pil_to_base64(overlay)

    return JSONResponse({
        "class": pred_class,
        "confidence": round(confidence * 100, 2),
        "original_image": original_b64,
        "gradcam_image": heatmap_b64
    })

# Función para arrancar el servidor desde main.py
def start_server():
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
