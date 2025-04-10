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
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """ Elimina los hooks registrados para liberar recursos. """
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def generate(self, input_tensor, target_class=None):
        """
        Genera el mapa GradCAM para una imagen de entrada.

        Args:
            input_tensor: Imagen preprocesada (Tensor) con tamaño 3x224x224.
            target_class (int, opcional): Clase para la cual calcular gradientes. Si no se da, se usa la predicha.

        Returns:
            grad_cam_map (numpy.array): Mapa de activación GradCAM [0, 1].
            predicted_class (int): Clase predicha.
        """
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0)  # Añadir batch dimension
        
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        if target_class is None:
            target_class = predicted_class
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)
        
        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=(2, 3))[0]
        
        grad_cam_map = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights):
            grad_cam_map += w * activations[0, i, :, :]
        
        grad_cam_map = F.relu(grad_cam_map)
        grad_cam_map = grad_cam_map.cpu().numpy()
        grad_cam_map = cv2.resize(grad_cam_map, (224, 224))
        grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)
        
        return grad_cam_map, predicted_class

# --- NUEVO: función utilitaria para uso externo (como desde server.py) ---

def generate_gradcam(model, input_tensor, target_layer="features", class_idx=None):
    """
    Función externa para generar el mapa GradCAM fácilmente.

    Args:
        model: Modelo de clasificación.
        input_tensor: Imagen ya transformada (tensor 1x3x224x224).
        target_layer (str): Nombre de la capa interna, por defecto "features".
        class_idx (int, opcional): Índice de clase objetivo (si se quiere forzar).

    Returns:
        heatmap (np.array): Mapa de activación normalizado.
    """
    if isinstance(target_layer, str):
        target_layer = dict([*model.named_modules()])[target_layer]

    gradcam = GradCAM(model, target_layer)
    cam_map, _ = gradcam.generate(input_tensor.squeeze(0), target_class=class_idx)
    gradcam.remove_hooks()
    return cam_map
