'''
By danczs (https://github.com/danczs)
References:
    https://github.com/facebookresearch/mae
    https://github.com/openai/CLIP
    https://github.com/Lednik7/CLIP-ONNX
    https://github.com/rwightman/pytorch-image-models
'''

import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import util.misc as misc
from util.dataset_autocolor import build_dataset
import timm.optim.optim_factory as optim_factory

from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched

from color_decoder import mae_color_decoder_base
from super_color import SuperColor
import torch.nn.functional as F
#
def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=10, type=int)

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--input_size_supercolor', default=448, type=int,
                        help='images input size')
    parser.add_argument('--data_path', default='E://data//carton_subset//train', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='the output dir of models and logs')
    parser.add_argument('--eval', action='store_true', help='evaluete the model')

    parser.add_argument('--colormask_prob', type=float, default=0.1, metavar='PCT',
                        help='the hyper-paramter of generating a colormask')
    parser.add_argument('--mae_feature_path', default=None, type=str,
                        help='the mae feature path')
    parser.add_argument('--clip_feature_path', default=None, type=str,
                        help='the clip feature path')
    parser.add_argument('--mae_model_path', default=None, type=str,
                        help='the clip model')
    parser.add_argument('--clip_model_path', default=None, type=str,
                        help='the clip feature path')
    parser.add_argument('--colordecoder_model_path', default=None, type=str,
                        help='the initialization of color decoder weights. \
                        If not specified, it will use the pre-trained mae decoder weights.')
    parser.add_argument('--supercolor_model_path', default=None, type=str,
                        help='the initialization of the super color weights')
    parser.add_argument('--grad_state', default='010', type=str,
                        help=' whether or not to train mae encoder, mae color decoder and super color. \
                         e.g. 010 indicates only training the color decoder')
    parser.add_argument('--supercolor_only',action='store_true',
                        help='only train or eval the supercolor model')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--blr_cd', type=float, default=5e-3, metavar='LR',
                        help='base learning rate color decoder: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--blr_sc', type=float, default=1e-1, metavar='LR',
                        help='base learning rate of super color: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='hyper-parameter to balance L1 loss and L2 loss')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--seed', default=0, type=int)
    return parser

def main(args):
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    #dataset anda data loader
    dataset = build_dataset(args=args)
    if args.eval:
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last= not args.eval
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # mae model
    assert len(args.grad_state) == 3
    mae_eval, color_decoder_eval, super_color_eval = [i=='0' for i in args.grad_state]

    #mae encoder model
    if args.mae_feature_path is None:
        from mae_encoder import mae_vit_base_patch16_dec512d8b
        mae_model = mae_vit_base_patch16_dec512d8b()
        mae_weights = torch.load(args.mae_model_path, map_location='cpu')['model']
        msg = mae_model.load_state_dict(mae_weights, strict=False)
        print(msg)
        if mae_eval or args.eval:
            mae_model.eval()
        mae_model = mae_model.to(device)

    # build color decoder model
    if not args.supercolor_only:
        color_decoder = mae_color_decoder_base()
        if args.colordecoder_model_path is None:
            mae_weights = torch.load(args.mae_model_path, map_location='cpu')['model']
            del mae_weights['decoder_pos_embed']
            msg = color_decoder.load_state_dict(mae_weights, strict=False)
            print(msg)
        else:
            color_decoder_weight = torch.load(args.colordecoder_model_path, map_location='cpu')
            msg = color_decoder.load_state_dict(color_decoder_weight, strict=False)
            print(msg)
        if color_decoder_eval or args.eval:
            color_decoder.eval()
        color_decoder = color_decoder.to(device)

    #build supercolor model
    if not super_color_eval or args.eval:
        super_color = SuperColor(kernel_size=5, group=4)
        if args.supercolor_model_path:
            super_color_checkpoint = torch.load(args.supercolor_model_path, map_location='cpu')
            msg = super_color.load_state_dict(super_color_checkpoint, strict=False)
            print(msg)
            if args.eval:
                super_color.eval()
        super_color = super_color.to(device)

    if color_decoder_eval is False:
        lr_cd = args.blr_cd * args.batch_size / 256
        param_groups_cd = optim_factory.param_groups_weight_decay(color_decoder,weight_decay=args.weight_decay)
        optimizer_cd = torch.optim.AdamW(param_groups_cd, lr=lr_cd, betas=(0.9, 0.95))
    else:
        optimizer_cd = None
    if super_color_eval is False:
        lr_sc = args.blr_sc * args.batch_size / 256
        param_groups_sc = optim_factory.param_groups_weight_decay(super_color, weight_decay=args.weight_decay)
        optimizer_sc = torch.optim.AdamW(param_groups_sc, lr=lr_sc, betas=(0.9, 0.95))
    else:
        optimizer_sc = None

    loss_scaler = NativeScaler()
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(args.epochs):
        avg_loss =0
        for iter_step, (mae_feature, clip_feature, color_mask, img_l, img_l_gray, img_h, img_h_gray, target) in enumerate(data_loader):
            color_mask = color_mask.to(device, non_blocking=True)
            img_l = img_l.to(device, non_blocking=True)
            img_h = img_h.to(device, non_blocking=True)
            img_l_gray = img_l_gray.to(device, non_blocking=True)
            img_h_gray = img_h_gray.to(device, non_blocking=True)
            clip_feature = clip_feature.to(device, non_blocking=True)

            if iter_step % args.accum_iter == 0:
                if optimizer_cd is not None:
                    lr_sched.adjust_learning_rate(optimizer_cd, iter_step / len(data_loader) + epoch, lr_cd, args)
                if optimizer_sc is not None:
                    lr_sched.adjust_learning_rate(optimizer_sc, iter_step / len(data_loader) + epoch, lr_sc, args)

            if args.mae_feature_path is not None:
                mae_feature = mae_feature.to(device,non_blocking=True)
            else:
                with torch.cuda.amp.autocast():
                    mae_feature = mae_model(img_l_gray)

            with torch.cuda.amp.autocast():
                if not args.supercolor_only:
                    pred = color_decoder(mae_feature, clip_feature, color_mask=color_mask)
                    pred_upsampling = F.interpolate(pred + img_l_gray, size=(img_h.size()[2:]))
                    if not color_decoder_eval or args.eval:
                        loss_decoder = color_decoder.forward_loss(pred, img_l_gray, img_l,alpha=args.alpha)
                else:
                    pred_upsampling = F.interpolate(img_l, size=(img_h.size()[2:]))

                if not super_color_eval or args.eval:
                    color_mask_sc = F.interpolate(color_mask, size=(img_h.size()[2:]))
                    sc_pred = super_color(pred_upsampling.detach(), img_h_gray, color_mask_sc) #detach cd and sc
                    loss_sc = super_color.forward_loss(sc_pred, img_h_gray, img_h,alpha=args.alpha)

            if args.eval:
                if loss_decoder  is not None:
                    avg_loss += loss_decoder
                if loss_sc is not None:
                    avg_loss += loss_sc
                continue

            if optimizer_cd is not None:

                loss_decoder /= args.accum_iter
                loss_scaler(loss_decoder, optimizer_cd, parameters=color_decoder.parameters(),
                            update_grad=(iter_step + 1) % args.accum_iter == 0)
                if (iter_step + 1) % args.accum_iter == 0:
                    optimizer_cd.zero_grad()
                avg_loss += loss_decoder.detach().item()
                lr = optimizer_cd.param_groups[0]["lr"]
                if iter_step % 100 == 0:
                    print('epoch:{} iter:{} color deocder loss:{} lr:{}'.format(epoch, iter_step, loss_decoder, lr))

            if optimizer_sc is not None:

                loss_sc /= args.accum_iter
                loss_scaler(loss_sc, optimizer_sc, parameters=super_color.parameters(),
                            update_grad=(iter_step + 1) % args.accum_iter == 0)
                if (iter_step + 1) % args.accum_iter == 0:
                    optimizer_sc.zero_grad()
                avg_loss += loss_sc.detach().item()
                lr = optimizer_sc.param_groups[0]["lr"]
                if iter_step % 100 == 0:
                    print('epoch:{} iter:{} super color loss:{} lr:{}'.format(epoch, iter_step, loss_sc, lr))
            torch.cuda.synchronize()

        print('epoch:{} avg loss:{}'.format(epoch,avg_loss/len(data_loader)))
        if args.eval:
            break

        if optimizer_cd:
            torch.save(color_decoder.state_dict(),
                       os.path.join(args.output_dir,'colordecoder_alpha{}_lr{}_p{}.pth'.format(args.alpha,args.blr_cd,args.colormask_prob)))
        if optimizer_sc:
            torch.save(super_color.state_dict(),
                       os.path.join(args.output_dir,'supercolor_alpha{}_lr{}_p{}.pth'.format(args.alpha, args.blr_sc,args.colormask_prob)))

if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)