from typing import Any, Tuple
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

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

def build_dataset_clip(args):
    transform = build_transform( args)
    root = args.data_path
    dataset = MyImageFolder(root, transform=transform)

    return dataset

def build_transform(args):
    t = []
    t.append(
        transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
    return transforms.Compose(t)
