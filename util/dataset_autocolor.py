import numpy as np
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import InterpolationMode

from torchvision.datasets import ImageFolder
from typing import Any, Callable, Optional, Tuple
import torch


class AutoColorImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            transform_l_s1: Optional[Callable] = None,
            transform_h_s1: Optional[Callable] = None,
            transform_s2: Optional[Callable] = None,
            transform_gray: Optional[Callable] = None,
            mae_feature_path = None,
            clip_feature_path = None
    ):
        super().__init__(
            root,
            transform = transform
        )
        self.imgs = self.samples
        self.last_data = None

        self.transform_l_s1 = transform_l_s1
        self.transform_h_s1 = transform_h_s1
        self.transform_s2 = transform_s2
        self.transform_gray = transform_gray
        if clip_feature_path:
            self.clip_features = torch.from_numpy(np.load(clip_feature_path))
        else:
            self.clip_features = None

        if mae_feature_path:
            self.mae_feature_files = []
            with open(mae_feature_path, 'r') as file:
                for line in file:
                    self.mae_feature_files.append(line.strip())
        else:
            self.mae_feature_files = None

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
            img_l_interp = self.transform(sample)
            color_mask = self.transform_s2(img_l_interp)
        img_l_s1 = self.transform_l_s1(sample)
        img_h_s1 = self.transform_h_s1(sample)
        img_l_gray = self.transform_gray(img_l_s1)
        img_h_gray = self.transform_gray(img_h_s1)
        img_h = self.transform_s2(img_h_s1)
        img_l = self.transform_s2(img_l_s1)
        img_l_gray = self.transform_s2(img_l_gray)
        img_h_gray = self.transform_s2(img_h_gray)
        if self.mae_feature_files is not None:
            mae_feature = np.load(self.mae_feature_files[index])
            mae_feature = torch.from_numpy(mae_feature).float()
        else:
            mae_feature = []
        if self.clip_features is not None:
            clip_feature = self.clip_features[index]
        else:
            clip_feature = []

        if self.target_transform is not None:
            target = self.target_transform(target)

        return mae_feature, clip_feature, color_mask, img_l, img_l_gray, img_h, img_h_gray, target

def build_dataset(args):
    transform_l_s1 = transform_img_l_s1(args)
    transform_h_s1 = transform_img_h_s1(args)
    transform_s2 = transform_img_s2()
    gray_transform = transformer_gray()
    color_mask_transformer = ColorMask(img_size=args.input_size, p=args.colormask_prob, grids=[2,4,8,16])

    root = args.data_path
    dataset = AutoColorImageFolder(root, transform=color_mask_transformer,
                            transform_l_s1=transform_l_s1,
                            transform_h_s1=transform_h_s1,
                            transform_s2=transform_s2,
                            transform_gray=gray_transform,
                            mae_feature_path = args.mae_feature_path,
                            clip_feature_path = args.clip_feature_path)
    return dataset

def transform_img_h_s1(args):
    t = []
    t.append(
        transforms.Resize(args.input_size_supercolor, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size_supercolor))
    return transforms.Compose(t)

def transform_img_l_s1(args):
    t = []
    t.append(
        transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))
    return transforms.Compose(t)

def transform_img_s2():
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    t = []
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def transformer_gray():
    return transforms.Grayscale(num_output_channels=3)

class ColorMask(torch.nn.Module):
    def __init__(self, img_size=224, p=0.2, grids=[4,8,16]):
        super().__init__()
        self.p = p
        self.grids = grids
        self.img_size = img_size

    def forward(self,img):
        return self.random_color_mask(img,self.img_size)

    def random_color_mask(self, img, size=224):
        p = self.p
        if p < 1e-6:
            return np.zeros((size,size,3),dtype=np.float32)
        grids = self.grids
        all_size = [size // g for g in grids]
        output_img = 0.
        prev_mask = 1.0
        for i in range(len(all_size)):
            tmp_size = all_size[i]
            p_tmp = np.random.rand() * p

            p_size = (tmp_size, tmp_size)
            p_img = img.resize(p_size, resample=Image.Resampling.BICUBIC)

            p_mask = np.random.rand(tmp_size, tmp_size)
            p_mask = p_mask[:, :, None]
            p_mask = (p_mask < p_tmp).astype(np.float32)

            p_mask = p_mask.repeat(grids[i], axis=0).repeat(grids[i], axis=1)
            p_img = np.array(p_img).astype(np.float32)
            p_img = p_img.repeat(grids[i], axis=0).repeat(grids[i], axis=1)

            p_img = prev_mask * p_mask * p_img
            prev_mask = (1.0 - p_mask) * prev_mask
            output_img += p_img

        #code to show color mask example
        # image = output_img.astype(np.uint8)
        # pil_image = Image.fromarray(image)
        # pil_image.show()
        return output_img/255.

if __name__ == '__main__':
    class Test:
        def __init__(self):
            self.input_size=224
            self.input_size_supercolor = 448
            self.colormask_prob = 0.1
            self.mae_feature_path = None
            self.clip_feature_path = None
    args = Test()
    args.input_size = 224
    args.data_path = 'E://data//carton_subset//train'

    data_set = build_dataset(args)
    for i in range(10):
        mae_feature, clip_feature, color_mask, img_l, img_l_gray, img_h, img_h_gray, target = data_set.__getitem__(i)
        #print(sample.shape,target,path,clip_feature.shape,mae_feature.shape)
    #print(get_image_backend())

