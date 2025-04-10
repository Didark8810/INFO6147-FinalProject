# data_utils.py

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_transforms(img_size=224):
    """
    Define las transformaciones para entrenamiento, validación y prueba.
    
    Args:
        img_size (int): Tamaño al que se redimensionarán las imágenes.
    
    Returns:
        dict: Diccionario con transformaciones para 'train' y 'val'.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }
    return data_transforms

def load_datasets(data_dir, transforms, valid_ratio=0.2, test_ratio=0.1):
    """
    Carga el dataset usando ImageFolder y lo divide en entrenamiento, validación y prueba.

    Se asume que 'data_dir' tiene la siguiente estructura:
        data_dir/
          class1/
            img1.jpg
            img2.jpg
            ...
          class2/
            imgA.jpg
            imgB.jpg
            ...
    
    Args:
        data_dir (str): Ruta a la carpeta con el dataset.
        transforms (dict): Transformaciones para 'train' y 'val'.
        valid_ratio (float): Proporción para el conjunto de validación (ej: 0.2).
        test_ratio (float): Proporción para el conjunto de prueba (ej: 0.1).

    Returns:
        dict: Diccionario con subdatasets de 'train', 'val' y 'test'.
    """
    # Cargamos el dataset completo (sin aplicar transformaciones de manera definitiva)
    full_dataset = datasets.ImageFolder(data_dir, transform=transforms['train'])
    dataset_size = len(full_dataset)

    # Calculamos tamaños para entrenamiento, validación y prueba
    test_size = int(test_ratio * dataset_size)
    valid_size = int(valid_ratio * dataset_size)
    train_size = dataset_size - valid_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])
    
    # Actualizamos las transformaciones para el conjunto de validación y prueba
    # Por ejemplo, reasignando la transformación 'val' para ellos.
    val_dataset.dataset.transform = transforms['val']
    test_dataset.dataset.transform = transforms['val']
    
    datasets_dict = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    return datasets_dict

def get_data_loaders(data_dir, batch_size=32, img_size=224, valid_ratio=0.2, test_ratio=0.1):
    """
    Crea los DataLoaders para entrenamiento, validación y prueba.

    Args:
        data_dir (str): Ruta a la carpeta del dataset.
        batch_size (int): Tamaño del batch.
        img_size (int): Tamaño de redimensionamiento de las imágenes.
        valid_ratio (float): Proporción para validación.
        test_ratio (float): Proporción para test.

    Returns:
        dict: Diccionario con DataLoaders para 'train', 'val' y 'test'.
    """
    transforms_dict = get_data_transforms(img_size)
    datasets_dict = load_datasets(data_dir, transforms_dict, valid_ratio, test_ratio)

    dataloaders = {
        phase: DataLoader(datasets_dict[phase], batch_size=batch_size, shuffle=(phase=='train'), num_workers=4)
        for phase in ['train', 'val', 'test']
    }
    return dataloaders

if __name__ == "__main__":
    # Ejemplo de uso:
    # Suponiendo que el dataset está organizado en:
    #   ./data/FruitImages/
    #       Fresh/
    #           imagen1.jpg
    #           ...
    #       Rotten/
    #           imagenA.jpg
    #           ...
    data_directory = "./data/FruitImages"
    batch_size = 32
    img_size = 224

    loaders = get_data_loaders(data_directory, batch_size=batch_size, img_size=img_size)
    
    # Mostramos la cantidad de imágenes en cada DataLoader
    print("Número de lotes:")
    for phase in loaders:
        print(f"{phase.capitalize()}: {len(loaders[phase])} batches")
