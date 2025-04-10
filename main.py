from models.classifier import FruitClassifier
from utils.data_utils import get_data_loaders  
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Entrenar el modelo')
    parser.add_argument('--evaluate', action='store_true', help='Evaluar el modelo')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas de entrenamiento')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_path', type=str, default='models_trained/efficientnet_finetuned.pth', help='Ruta para guardar/cargar el modelo')
    parser.add_argument('--serve', action='store_true', help="Iniciar el servidor para predicción web.")

    args = parser.parse_args()

    # Configuración
    data_dir = "./data"
    batch_size = 32
    img_size = 224

    # Carga de datos
    loaders = get_data_loaders(data_dir, batch_size=batch_size, img_size=img_size)

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Obtener clases desde los datos
    class_names = train_loader.dataset.dataset.classes

    print(f"Clases encontradas: {class_names}")
    print(f"Total de clases: {len(class_names)}")
    print(f"Total de imágenes de entrenamiento: {len(train_loader.dataset)}")
    print(f"Total de imágenes de validación: {len(val_loader.dataset)}")

    # Instancia del clasificador
    classifier = FruitClassifier(num_classes=len(class_names))

    if args.train:
        print("🔧 Entrenando modelo...")
        classifier.train_model(train_loader, val_loader, epochs=args.epochs, lr=args.lr)
        classifier.save_model(args.model_path, class_names=class_names)  

    if args.serve:
        from app.server import start_server
        print("🚀 Iniciando servidor web...")
        start_server()

    elif args.evaluate:
        print("📊 Evaluando modelo...")
        classifier.load_model(args.model_path)
        classifier.evaluate_model(test_loader)


if __name__ == '__main__':
    main()

