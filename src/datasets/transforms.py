from __future__ import annotations


def build_cifar10_transforms(train: bool):
    from torchvision import transforms

    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    return transforms.ToTensor()


def build_train_transform():
    return build_cifar10_transforms(train=True)


def build_eval_transform():
    return build_cifar10_transforms(train=False)
