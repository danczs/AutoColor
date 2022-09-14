from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import InterpolationMode

from torchvision.datasets import ImageFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
class MyImageFolder(ImageFolder):

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


def build_dataset(args):
    transform = build_transform(args)

    root = args.data_path
    dataset = MyImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    t = []
    size = args.input_size
    t.append(
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )

    #new add
    t.append(transforms.Grayscale(num_output_channels=3))

    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
