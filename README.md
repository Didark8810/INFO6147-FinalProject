# INFO6147-FinalProject


# 🍎 Fruit Classifier + GradCAM + DETR 🍌

Proyecto final para la materia **INFO6147**, que combina visión por computadora con técnicas de interpretación visual y detección de objetos. Esta aplicación permite:

- Clasificar frutas como **manzana, banana u naranja**, frescas o podridas.
- Visualizar la atención del modelo con **GradCAM**.
- Detectar objetos con un modelo basado en **DETR (DEtection TRansformer)**.
- Ver videos asociados a cada clase de fruta.

---

## 🧠 Modelo de Clasificación

El modelo de clasificación fue entrenado con PyTorch sobre un dataset organizado por carpetas con clases como:

```
train/
  freshapples/
  freshbanana/
  freshorange/
  rottenapples/
  rottenbanana/
  rottenoranges/
```

El modelo genera dos salidas principales:
- Predicción de clase.
- Imagen GradCAM para explicar la activación del modelo.

---

## 🎯 Detección con DETR

Se utilizó una versión simplificada del modelo **DETR de Facebook Research** para realizar detección de frutas, destacando la posición y distancia desde el centro de la imagen.

---

## 🖥️ Interfaz Web

El frontend fue desarrollado con **HTML + Bootstrap 5 + JavaScript** y permite:

- Subir una imagen para clasificación o detección.
- Visualizar resultados en tarjetas con estilo moderno.
- Reproducir videos asociados por clase.

### 🎬 Video Viewer

Al seleccionar una fruta del menú desplegable, se carga un video correspondiente desde `data/video-fruit.mp4` y se reproduce en bucle. Controles incluidos para reproducir/pausar el video.

---

## 📦 Estructura del Proyecto

```
.
├── backend/
│   ├── main.py              # FastAPI server con endpoints /predict y /detect
│   ├── model_classifier.pt  # Modelo entrenado
│   ├── gradcam.py           # Utilidad para generar mapas GradCAM
│   └── detr_model.py        # Implementación simplificada de DETR
│
├── data/
│   ├── video-apple.mp4
│   ├── video-banana.mp4
│   └── video-orange.mp4
│
├── frontend/
│   ├── index.html
│   ├── index.js
│   └── styles.css
│
├── README.md
└── requirements.txt
```

---

## 🚀 Cómo ejecutar el proyecto

### 1. Backend (FastAPI + PyTorch)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2. Frontend

Solo abre `frontend/index.html` en tu navegador.

> ⚠️ Asegúrate de que el backend esté corriendo en `http://localhost:8000`.

---

## 📷 Ejemplo de salida

> Puedes incluir imágenes como `docs/example_classification.png` y `docs/example_gradcam.png` si las agregas al repositorio.

---

## 📌 Requisitos

- Python 3.9+
- PyTorch
- FastAPI
- OpenCV
- Bootstrap 5

---

## 🔗 Enlace al repositorio

[📁 GitHub: INFO6147 Final Project](https://github.com/Didark8810/INFO6147-FinalProject)

---

## 🧑‍💻 Autor

**Didark8810** – Proyecto Final INFO6147 @ Universidad
