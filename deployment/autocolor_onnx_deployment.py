from PIL import Image
from typing import Any, Union, List
import onnxruntime
import numpy as np
from simple_tokenizer import SimpleTokenizer as _Tokenizer

class AutoColorDeployment:
    def __init__(self):
        self.device = 0 # 0 for cpu, 1 for gpu
        providers = ['CPUExecutionProvider']

        self.mae_encoder_file = './onnx_models/mae_encoder_vitb_onnx.onnx'
        self.clip_text_file = './onnx_models/clip_textual.onnx'
        self.clip_image_file = './onnx_models/clip_visual.onnx'
        self.color_decoder_file = './onnx_models/color_decoder_onnx.onnx'
        self.super_color_file = './onnx_models/super_color_onnx.onnx'
        self.mae_encoder_session = onnxruntime.InferenceSession(self.mae_encoder_file, providers=providers)
        self.clip_text_session = onnxruntime.InferenceSession(self.clip_text_file, providers=providers)
        self.color_decoder_session = onnxruntime.InferenceSession(self.color_decoder_file, providers=providers)
        self.clip_image_session = onnxruntime.InferenceSession(self.clip_image_file, providers=providers)
        self.super_color_session = onnxruntime.InferenceSession(self.super_color_file, providers=providers)
        self._tokenizer = _Tokenizer(bpe_path='./onnx_models/bpe_simple_vocab_16e6.txt.gz')

    # 0 for cpu, 1 for gpu
    def set_device(self, device):
        if device == self.device:
            return
        else:
            self.device = device
        providers = ['CPUExecutionProvider'] if device==0 else ['CUDAExecutionProvider']
        self.mae_encoder_session = onnxruntime.InferenceSession(self.mae_encoder_file, providers=providers)
        self.clip_text_session = onnxruntime.InferenceSession(self.clip_text_file, providers=providers)
        self.color_decoder_session = onnxruntime.InferenceSession(self.color_decoder_file, providers=providers)
        self.clip_image_session = onnxruntime.InferenceSession(self.clip_image_file, providers=providers)
        self.super_color_session = onnxruntime.InferenceSession(self.super_color_file, providers=providers)

    #interp sample with nearest pixels, which same with F.interpolate(x,nearest) in pytorch
    def numpy_nearest_interp(self, sample, size):

        b, c, w, h = sample.shape
        new_w, new_h = size
        if new_w % w == 0 and new_h % h == 0:
            return sample.repeat(new_w // w, axis=2).repeat(new_h // h, axis=3)
        x = np.arange(new_w) * w / new_w
        y = np.arange(new_h) * w / new_h
        x = x[:, None].repeat(new_h, axis=1).reshape(-1).astype(np.int)
        y = y[None, :].repeat(new_w, axis=0).reshape(-1).astype(np.int)
        interp_sample = sample[:, :, x, y].reshape(b, c, new_w, new_h)
        return interp_sample

    def get_image_from_numpy(self, sample,clip_norm=False):
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
        img = img[None,:,:,:]
        return img

    def clip_tokenize(self,texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) :
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self._tokenizer.encode(text) + [eot_token] for text in texts]
        # if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        #     result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        # else:
        result = np.zeros((len(all_tokens), context_length), dtype=np.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = tokens

        return result

    def autocolor_forward(self, input_images, image_info, input_text, color_mask):
        """
        input_images : List[PIL.Image]
            The input gray images with different resolutions.
            e.g. for a 1080 x 1080 input image, the input image list is [224x224, 448x448, 896x896, 1080x1080]

        image_info : PIL.Image 224x224
            The info for image colorization, which will be fed to the clip image model

        input_text : str
            The info for image colorization, which will be fed to the clip text model

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
                clip_feature = self.clip_image_session.run(None, {"input":clip_input})[0]
                image_clip_feature_sum += clip_feature

        for i in range(len(input_images)):
            img = self.pre_processing(input_images[i])
            input_images[i] = img

        if input_text is not None:
            text = self.clip_tokenize([input_text])
            text_features = self.clip_text_session.run(None,{"input":text})[0]
            if len(image_info) > 0:
                clip_feature = ( image_clip_feature_sum + text_features ) * 0.5
            else:
                clip_feature = text_features

        mae_feature = self.mae_encoder_session.run(None,{"input":input_images[0]})[0]
        color_mask = self.pre_processing(color_mask)

        #color decoder
        pred = self.color_decoder_session.run(None,{"input_mae":mae_feature, "input_clip":clip_feature, "input_color":color_mask})[0]

        #super color
        for i in range(1,len(input_images)):
            img_pred = pred + input_images[i-1]
            pred_upsampling = self.numpy_nearest_interp(img_pred,input_images[i].shape[2:])
            color_mask = self.numpy_nearest_interp(color_mask,input_images[i].shape[2:])
            sc_pred = self.super_color_session.run(None,{"input_interp":pred_upsampling, "input_gray":input_images[i], "input_color":color_mask})[0]
            pred = sc_pred
        img_pred = pred + input_images[-1]
        output_img = self.get_image_from_numpy(img_pred[0])
        return output_img

if __name__=='__main__':
    auto_color_deploy = AutoColorDeployment()



