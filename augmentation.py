import torch
import torchvision.transforms as transforms

image_transformer = {
    'train':
        transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(x)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    'valid':
        transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(x)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}
