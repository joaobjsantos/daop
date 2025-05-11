import torchvision


class CIFAR10AlbumentationsCUDATest(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, download=False, pre_transform=None, da_transform=None, post_transform=None, device=None):
        super().__init__(root=root, train=train, download=download, transform=pre_transform)
        
        self.pre_transform = pre_transform
        self.da_transform = da_transform
        self.post_transform = post_transform
        self.device = "cpu" if device is None else device

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.pre_transform is not None:
            transformed = self.pre_transform(image=image, label=label)
            image = transformed["image"]

        image = image.to(self.device)

        if self.da_transform is not None:
            transformed = self.da_transform(image=image, label=label)
            image = transformed["image"]

        if self.post_transform is not None:
            transformed = self.post_transform(image=image, label=label)
            image = transformed["image"]

        return image.cpu(), label
    


class CIFAR10Albumentations(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, download=False, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label