from torchvision import datasets, transforms
from torch.utils.data import random_split

def get_data_loadersBasic(data_dir, batch_size=32, val_split=0.1):
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset_full = datasets.ImageFolder(root=f'{data_dir}/train', transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=test_transforms)

    val_size = int(val_split * len(train_dataset_full))
    train_size = len(train_dataset_full) - val_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])
    val_dataset.dataset.transform = test_transforms  # quitar augmentación para validación

    return train_dataset, val_dataset, test_dataset, train_dataset_full.classes
