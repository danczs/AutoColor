import argparse
import numpy as np
import torch
from util.datasets import build_dataset
from mae import mae_vit_base_patch16_dec512d8b
import os
BATCH_SIZE = 20
#
def get_args_parser():
    parser = argparse.ArgumentParser('extract mae features', add_help=False)
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--data_path', default='', type=str,
                        help='path//to//dataset')
    parser.add_argument('--output_path', default='', type=str,
                        help='path//to//output//mae_feature//')
    return parser

def main(args):
    model_dir = './models/mae_visualize_vit_base.pth'
    dataset_test = build_dataset(args=args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    checkpoint = torch.load(model_dir, map_location='cpu')
    checkpoint_model = checkpoint['model']
    model = mae_vit_base_patch16_dec512d8b()
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.cuda()
    model.eval()

    mae_feature_names = []
    for samples, targets, path in data_loader_test:
        samples = samples.to(device, non_blocking=True)
        x, mask, ids_restore = model.forward_encoder(samples,mask_ratio=0.0)
        x = x.detach().cpu().numpy()
        mae_features = x
        for b in range(x.shape[0]):
            img_name = path[b].split('\\')[-1].split('.')[-2]
            feature_name = os.path.join(args.output_path, 'mae_feature_{}.npy'.format(img_name))
            mae_feature_names.append(feature_name)
            np.save(feature_name,mae_features[b])

        if len(mae_feature_names) % 100 ==0:
            print(len(mae_feature_names))

    with open('features/mae_feature_names_train.txt', 'w') as file:
        for i, p in enumerate(mae_feature_names):
            file.write(p + '\n')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)