from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.append("..")

from mae_encoder import mae_vit_base_patch16_dec512d8b
from color_decoder import mae_color_decoder_base
from super_color import SuperColor
import clip

class AutoColorDeployment:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        mae_encoder_dir = './pytorch_models/mae_visualize_vit_base.pth'
        mae_encoder_model = mae_vit_base_patch16_dec512d8b()
        mae_cp = torch.load(mae_encoder_dir, map_location='cpu')
        msg = mae_encoder_model.load_state_dict(mae_cp['model'],strict=False)
        mae_encoder_model = mae_encoder_model.to(self.device)
        mae_encoder_model.eval()
        self.mae_encoder_model = mae_encoder_model
        print(msg)

        clip_model,_ = clip.load("ViT-B/16", download_root='./pytorch_models',device=self.device)
        clip_model.eval()
        self.clip_model = clip_model

        color_decoder_dir = './pytorch_models/color_decoder.pth'
        color_decoder_model = mae_color_decoder_base()
        color_decoder_cp = torch.load(color_decoder_dir,map_location='cpu')
        msg = color_decoder_model.load_state_dict(color_decoder_cp, strict=False)
        color_decoder_model = color_decoder_model.to(self.device)
        color_decoder_model.eval()
        self.color_decoder_model = color_decoder_model
        print(msg)

        super_color_dir = './pytorch_models/super_color.pth'
        super_color_model = SuperColor(kernel_size=5, group=4)
        super_color_checkpoint = torch.load(super_color_dir, map_location='cpu')
        msg = super_color_model.load_state_dict(super_color_checkpoint, strict=False)
        super_color_model = super_color_model.to(self.device)
        super_color_model.eval()
        self.super_color_model = super_color_model
        print(msg)

    def set_device(self, device):
        return

    def get_image_from_tensor(self, sample,clip_norm=False):
        sample = sample.cpu().detach().numpy()
        image = np.transpose(sample, (1, 2, 0))
        if clip_norm:
            mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)*255
            std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)*255
        else:
            mean = np.asarray([123.68, 116.28, 103.53], dtype=np.float32)
            std = np.asarray([58.395, 57.120, 57.375], dtype=np.float32)

        image = image * std + mean
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        return pil_image

    def pre_processing(self, img, clip_norm=False):
        img = np.array(img, dtype=np.float32)
        if clip_norm:
            mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)*255
            std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)*255
        else:
            mean = np.asarray([123.68, 116.28, 103.53], dtype=np.float32)
            std = np.asarray([58.395, 57.120, 57.375], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img[None,:,:,:])
        return img

    def autocolor_forward(self,input_images, image_info, input_text, color_mask):
        """
        input_images : List[PIL.Image]
            The input gray images with different resolutions.
            e.g. for a 1080 x 1080 input image, the input image list is [224x224, 448x448, 896x896, 1080x1080]

        image_info : PIL.Image 224x224
            The image info for image colorization, which will be fed into the clip image model

        input_text : str
            The text info for image colorization, which will be fed into the clip text model

        color_mask: numpy.array 224x224x3
            The color info for image colorization

        Returns
        -------
        the colored image
        """
        image_clip_feature_sum = 0
        if len(image_info) > 0:
            for img in image_info:
                clip_input = self.pre_processing(img,clip_norm=True)
                clip_input = clip_input.to(self.device, non_blocking=True)
                clip_feature = self.clip_model.encode_image(clip_input)
                image_clip_feature_sum += clip_feature

        for i in range(len(input_images)):
            img = self.pre_processing(input_images[i])
            input_images[i] = img.to(self.device, non_blocking=True)

        if input_text is not None:
            text = clip.tokenize([input_text]).to(self.device)
            text_features = self.clip_model.encode_text(text)#.repeat(BATCH_SIZE,1)
            if len(image_info) > 0:
                clip_feature = ( image_clip_feature_sum + text_features ) * 0.5
            else:
                clip_feature = text_features

        mae_feature = self.mae_encoder_model(input_images[0])
        color_mask = self.pre_processing(color_mask)
        color_mask =  color_mask.to(self.device, non_blocking=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self.color_decoder_model(mae_feature, clip_feature, color_mask=color_mask)

                for i in range(1,len(input_images)):
                    img_pred = pred + input_images[i-1]
                    pred_upsampling = F.interpolate(img_pred, size=(input_images[i].size()[2:]))

                    color_mask = F.interpolate(color_mask, size=(input_images[i].size()[2:]))
                    sc_pred = self.super_color_model(pred_upsampling,input_images[i],color_mask)

                    pred = sc_pred
                img_pred = pred + input_images[-1]
        output_img = self.get_image_from_tensor(img_pred[0])
        return output_img
