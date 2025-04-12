import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
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
        self.forward_handle.remove()
        self.backward_handle.remove()
    
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0)
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

def generate_gradcam(model, input_tensor, target_layer="features", class_idx=None):
    if isinstance(target_layer, str):
        target_layer = dict([*model.named_modules()])[target_layer]

    gradcam = GradCAM(model, target_layer)
    cam_map, _ = gradcam.generate(input_tensor.squeeze(0), target_class=class_idx)
    gradcam.remove_hooks()
    return cam_map
