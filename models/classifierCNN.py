# import torch
# import torch.nn as nn
# from sklearn.metrics import classification_report
# from models.classifierCNNModel import FruitClassifierCNNModel



# class FruitClassifierCNN:
#     def __init__(self, num_classes, device=None):
#         """
#         Inicializa el modelo de clasificación CNN para frutas.
        
#         Args:
#             num_classes (int): Número de clases (por ejemplo, 2 para "Fresh" y "Rotten").
#         """
#         self.model = FruitClassifierCNNModel(num_classes)
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
        

#     def train_model1(self):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         train_dataset, val_dataset, test_dataset, classes = get_data_loaders("data", batch_size=32)
#         num_classes = len(classes)

#         model = FruitClassifierCNN(num_classes).to(device)
#         train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=32)
#         test_loader = DataLoader(test_dataset, batch_size=32)

#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)

#         model, history = train_model(model, train_loader, val_loader, criterion, optimizer, device)
#         print("Entrenamiento completo.")
    
#     def train_model(self, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, patience=3):
#         best_model_wts = copy.deepcopy(self.model.state_dict())
#         best_val_loss = float('inf')
        
#         patience_counter = 0

#         for epoch in range(num_epochs):
#             self.model.train()
#             running_loss = 0.0
#             correct = 0
            
#             for inputs, labels in train_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item() * inputs.size(0)
#                 correct += (outputs.argmax(1) == labels).sum().item()
                
#             train_loss = running_loss / len(train_loader.dataset)
#             train_acc = correct / len(train_loader.dataset)
            
            
#             # Evaluación en validación
#             self.model.eval()
#             val_loss = 0.0
#             val_correct = 0
#             with torch.no_grad():
#                 for inputs, labels in val_loader:
#                     inputs, labels = inputs.to(device), labels.to(device)
#                     outputs = self.model(inputs)
#                     loss = criterion(outputs, labels)
#                     val_loss += loss.item() * inputs.size(0)
#                     val_correct += (outputs.argmax(1) == labels).sum().item()
#             val_loss /= len(val_loader.dataset)
#             val_acc = val_correct / len(val_loader.dataset)
            
            
#             print(f"Epoch {epoch+1}/{num_epochs}: Train loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
#             # Early Stopping
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_model_wts = copy.deepcopy(self.model.state_dict())
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     print("Early stopping activado")
#                     break

        
        

#     def evaluate_model(self, test_loader, classes):
#         # Evaluación en test
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.eval()
#         all_preds, all_labels = [], []
#         with torch.no_grad():
#             for images, labels in test_loader:
#                 images = images.to(device)
#                 outputs = self.model(images)
#                 all_preds += outputs.argmax(1).cpu().tolist()
#                 all_labels += labels.tolist()

#         print(classification_report(all_labels, all_preds, target_names=classes))

#     def predict(self, image):
#         """
#         Realiza una predicción sobre una imagen de fruta.
#         Args:
#             image (PIL Image): Imagen de entrada.
#         Returns:
#             int: Clase predicha (0 o 1).
#         """
#         self.model.eval()
#         image = image.to(self.device)
#         with torch.no_grad():
#             output = self.model(image.unsqueeze(0))
#             _, predicted = torch.max(output, 1)
#         return predicted.item()
#     def save_model(self, path):
#         torch.save(self.model.state_dict(), path)
#     def load_model(self, path):
#         self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
#         self.model.to(self.device)
#         self.model.eval()
    
        
        