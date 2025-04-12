import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_transforms(img_size=224):
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
    full_dataset = datasets.ImageFolder(data_dir, transform=transforms['train'])
    dataset_size = len(full_dataset)
    test_size = int(test_ratio * dataset_size)
    valid_size = int(valid_ratio * dataset_size)
    train_size = dataset_size - valid_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])
    val_dataset.dataset.transform = transforms['val']
    test_dataset.dataset.transform = transforms['val']
    datasets_dict = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    return datasets_dict

def get_data_loaders(data_dir, batch_size=32, img_size=224, valid_ratio=0.2, test_ratio=0.1):
    transforms_dict = get_data_transforms(img_size)
    datasets_dict = load_datasets(data_dir, transforms_dict, valid_ratio, test_ratio)
    dataloaders = {
        phase: DataLoader(datasets_dict[phase], batch_size=batch_size, shuffle=(phase=='train'), num_workers=4)
        for phase in ['train', 'val', 'test']
    }
    return dataloaders

if __name__ == "__main__":
    data_directory = "./data/FruitImages"
    batch_size = 32
    img_size = 224
    loaders = get_data_loaders(data_directory, batch_size=batch_size, img_size=img_size)
    print("Número de lotes:")
    for phase in loaders:
        print(f"{phase.capitalize()}: {len(loaders[phase])} batches")
