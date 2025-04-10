# gradcam.py

import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Inicializa GradCAM con un modelo y la capa objetivo a la que se desean aplicar hooks.
        
        Args:
            model: Modelo de clasificación (por ejemplo, EfficientNet) ya entrenado.
            target_layer: La capa (módulo) en la que se obtendrán las activaciones y gradientes.
                          Por ejemplo, model._conv_head o la capa que consideres adecuada.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        # Configuramos los hooks para capturar activaciones y gradientes
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        # Registra los hooks en la capa objetivo
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """ Elimina los hooks registrados (muy útil para liberar memoria). """
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def generate(self, input_tensor, target_class=None):
        """
        Genera el mapa GradCAM para una imagen de entrada.
        
        Args:
            input_tensor: Imagen preprocesada (Tensor) de tamaño adecuado (por ejemplo, 224x224) y sin batch dim.
            target_class (int, opcional): Clase objetivo para la cual se calcularán los gradientes.
                Si no se especifica, se usa la clase predicha.
                
        Returns:
            grad_cam_map (numpy.array): Mapa de activación GradCAM normalizado a escala [0, 1].
            predicted_class (int): Clase predicha por el modelo.
        """
        # Asegurarse de que el modelo esté en modo evaluación
        self.model.eval()
        # Añadir batch dimension
        input_tensor = input_tensor.unsqueeze(0)
        
        # Forward pass
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Determinar la clase objetivo
        if target_class is None:
            target_class = predicted_class
        
        # Backward pass: Se realiza la retropropagación para el target_class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)
        
        # Obtener gradientes y activaciones de la capa objetivo
        gradients = self.gradients  # Shape: [1, C, H, W]
        activations = self.activations  # Shape: [1, C, H, W]
        
        # Calcular pesos: promedio global de los gradientes en cada canal
        weights = torch.mean(gradients, dim=(2, 3))[0]  # Shape: [C]
        
        # Construir el mapa de GradCAM: suma ponderada de las activaciones
        grad_cam_map = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights):
            grad_cam_map += w * activations[0, i, :, :]
        
        # Aplicar la función ReLU
        grad_cam_map = F.relu(grad_cam_map)
        
        # Convertir a numpy array y redimensionar (asumimos el tamaño final 224x224)
        grad_cam_map = grad_cam_map.cpu().numpy()
        grad_cam_map = cv2.resize(grad_cam_map, (224, 224))
        
        # Normalizar el mapa para que esté entre 0 y 1
        grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)
        
        return grad_cam_map, predicted_class


# Ejemplo de uso cuando se corre este módulo de forma independiente
if __name__ == "__main__":
    import torch
    import torchvision.transforms as transforms
    from efficientnet_pytorch import EfficientNet
    from PIL import Image
    import matplotlib.pyplot as plt

    # Cargar modelo preentrenado y ajustar la capa final (por ejemplo, para 2 clases)
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_features = model._fc.in_features
    model._fc = torch.nn.Linear(num_features, 2)
    model.eval()
    
    # Seleccionar la capa objetivo para GradCAM
    target_layer = model._conv_head  # Ajusta según el modelo y la versión que utilices
    
    # Inicializar GradCAM
    gradcam = GradCAM(model, target_layer)
    
    # Cargar y preprocesar una imagen de ejemplo
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open("ruta/a/tu/imagen.jpg").convert("RGB")
    input_tensor = preprocess(img)
    
    # Generar el mapa GradCAM
    cam_map, pred_class = gradcam.generate(input_tensor)
    
    # Visualización: superponer el mapa sobre la imagen (convertir imagen a numpy)
    img_np = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Imagen Original")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("GradCAM")
    plt.imshow(cam_map, cmap='jet')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")
    plt.suptitle(f"Predicción: Clase {pred_class}")
    plt.show()
