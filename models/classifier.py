# classifier.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights



class FruitClassifier:
    def __init__(self, num_classes=2, model_name='efficientnet-b0', pretrained=True, device=None):
        """
        Inicializa el modelo EfficientNet para clasificación de frutas.
        Args:
            num_classes (int): Número de clases (por ejemplo, 2 para "Fresh" y "Rotten").
            model_name (str): Nombre del modelo EfficientNet (ej. 'efficientnet-b0').
            pretrained (bool): Si usar pesos preentrenados en ImageNet.
            device: Dispositivo donde se ejecutará el modelo. Si None, se usará cuda si está disponible.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        self._modify_final_layer(num_classes)
        self.model.to(self.device)

    def _modify_final_layer(self, num_classes):
        """ Reemplaza la capa final para ajustar el número de clases deseado. """
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)


    def train_model(self, train_loader, val_loader=None, epochs=10, lr=1e-4):
        """
        Entrena el modelo usando el dataset de entrenamiento (y validación opcional).
        Args:
            train_loader: DataLoader del conjunto de entrenamiento.
            val_loader: DataLoader del conjunto de validación (opcional).
            epochs (int): Número de épocas a entrenar.
            lr (float): Learning rate para el optimizador.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = correct / total * 100

            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

            # Evaluación en el conjunto de validación si se proporciona
            if val_loader:
                self.evaluate_model(val_loader)
    
    def evaluate_model(self, data_loader):
        """
        Evalúa el rendimiento del modelo con un DataLoader (por ejemplo, el conjunto de validación).
        Args:
            data_loader: DataLoader con los datos a evaluar.
        """
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
        loss_avg = running_loss / total
        acc = correct / total * 100
        print(f"Validation | Loss: {loss_avg:.4f} | Accuracy: {acc:.2f}%")
        return loss_avg, acc

    def predict(self, image_tensor):
        """
        Realiza una predicción para una imagen preprocesada.
        Args:
            image_tensor: Imagen preprocesada (Tensor) de tamaño adecuado (ej., 224x224).
        Returns:
            Predicción (clase) y probabilidades.
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor.unsqueeze(0))
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        return predicted.item(), probabilities.squeeze().cpu().numpy()
    
    def save_model(self, path):
        """ Guarda los pesos del modelo en el archivo especificado. """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """ Carga los pesos del modelo desde el archivo especificado. """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
