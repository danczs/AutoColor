import argparse
import clip

#model, preprocess = clip.load("ViT-B/16", device=device)
from util.datasets_clip import build_dataset_clip
import torch
import numpy as np

BATCH_SIZE = 20
#
def get_args_parser():
    parser = argparse.ArgumentParser('extract clip features', add_help=False)
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--data_path', default='E://data//carton_subset//val', type=str,
                        help='dataset path')
    return parser


def main(args):
    dataset_test = build_dataset_clip(args=args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", download_root='./models', device=device)
    all_features = None
    all_path = []
    with torch.no_grad():
        for samples, targets, path in data_loader_test:
            samples = samples.to(device, non_blocking=True)
            image_features = model.encode_image(samples)
            image_features = image_features.detach().cpu().numpy()
            if all_features is None:
                all_features = image_features
            else:
                all_features = np.concatenate([all_features, image_features], axis=0)
            all_path += path
            if all_features.shape[0] % 100 ==0:
                print(all_features.shape[0])

    np.save('clip_features_subset.npy', all_features)

if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)