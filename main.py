from models.classifier import FruitClassifier
from utils.data_utils import get_data_loaders  
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--trainBasic', action='store_true', help='Train the basic model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_path', type=str, default='models_trained/efficientnet_finetuned.pth', help='Path to save/load the model')
    parser.add_argument('--serve', action='store_true', help="Start the web server for prediction.")

    args = parser.parse_args()

    data_dir = "./data/train"
    batch_size = 32
    img_size = 224

    loaders = get_data_loaders(data_dir, batch_size=batch_size, img_size=img_size)

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    class_names = train_loader.dataset.dataset.classes

    print(f"Found classes: {class_names}")
    print(f"Total number of classes: {len(class_names)}")
    print(f"Total training images: {len(train_loader.dataset)}")
    print(f"Total validation images: {len(val_loader.dataset)}")

    classifier = FruitClassifier(num_classes=len(class_names))

    if args.train:
        print("Training model...")
        classifier.train_model(train_loader, val_loader, epochs=args.epochs, lr=args.lr)
        classifier.save_model(args.model_path, class_names=class_names)  
    
    elif args.trainBasic:
        print("Training basic model...")
        #classifier.train_model(train_loader, val_loader, epochs=args.epochs, lr=args.lr)
        #classifier.save_model(args.model_path, class_names=class_names)  

    elif args.serve:
        from app.server import start_server
        print("Starting web server...")
        start_server()

    elif args.evaluate:
        print("Evaluating model...")
        classifier.load_model(args.model_path)
        classifier.evaluate_model(test_loader)


if __name__ == '__main__':
    main()
