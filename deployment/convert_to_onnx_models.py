import numpy as np
import torch
import sys
sys.path.append("..")
import onnx
import onnxruntime

def get_mae_onnx_model():
    from mae_encoder import mae_vit_base_patch16_dec512d8b
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mae_encoder_dir = './pytorch_models/mae_visualize_vit_base.pth'
    mae_encoder_model = mae_vit_base_patch16_dec512d8b()
    mae_cp = torch.load(mae_encoder_dir, map_location='cpu')
    msg = mae_encoder_model.load_state_dict(mae_cp['model'],strict=False)
    print(msg)
    mae_encoder_model = mae_encoder_model.to(device)
    mae_encoder_model.eval()

    export_onnx_file = './onnx_models/mae_encoder_vitb_onnx.onnx'
    x = torch.onnx.export(mae_encoder_model,
                          torch.randn(1,3,224,224,device=device),
                          export_onnx_file,
                          verbose=False,
                          input_names=['input'],
                          output_names=['output'],
                          opset_version=12,
                          do_constant_folding=True,
                          #dynamic_axes={'input':{0:"batch_size",2:"h"},"output":{0:"batch_size"},}
                          )
    print(x)
    net = onnx.load(export_onnx_file)
    onnx.checker.check_model(net)
    onnx.helper.printable_graph(net.graph)

    #test onnx model
    session = onnxruntime.InferenceSession(export_onnx_file, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    input = np.random.rand(1,3,224,224).astype('float32')
    out_r = session.run(None, {"input":input})
    print(out_r[0].shape)
    print(np.mean(out_r[0]))
    torch.set_printoptions(precision=12)
    input_tensor = torch.from_numpy(input).to(device)
    out = mae_encoder_model(input_tensor)
    print(out.size())
    print(torch.mean(out))

def get_clip_onnx_model():
    from clip_onnx import clip_onnx
    import clip
    device = "cpu"
    model, preprocess = clip.load("ViT-B/16", download_root='./pytorch_models',device=device)


    image = torch.randn(1,3,224,224,device=device)
    text = clip.tokenize(["a diagram"]).cpu().to(device)  # [3, 77]

    visual_path = "./onnx_models/clip_visual.onnx"
    textual_path = "./onnx_models/clip_textual.onnx"

    onnx_model = clip_onnx(model, visual_path=visual_path, textual_path=textual_path)
    onnx_model.convert2onnx(image, text, verbose=True)

    #image ecoder test
    torch.set_printoptions(precision=10)
    session = onnxruntime.InferenceSession(visual_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])

    input = np.random.rand(1,3,224,224).astype('float32')
    out_image = session.run(None, {"input":input})
    print(out_image[0].shape, np.mean(out_image[0]))

    input_tensor = torch.from_numpy(input).to(device)
    model = model.to(device)
    out_feature = model.encode_image(input_tensor)
    print(out_feature.size(), torch.mean(out_feature))

    #text encoder test
    session = onnxruntime.InferenceSession(textual_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    input = text.numpy()
    out_text = session.run(None, {"input": input})
    print(out_text[0].shape, np.mean(out_text[0]))

    input_tensor = text.to(device)
    out_feature = model.encode_text(input_tensor)
    print(out_feature.size(), torch.mean(out_feature))


def get_color_decoder_onnx_model():
    from color_decoder import mae_color_decoder_base
    export_onnx_file = './onnx_models/color_decoder_onnx.onnx'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    color_decoder_dir = './pytorch_models/color_decoder.pth'
    color_decoder_model = mae_color_decoder_base()
    color_decoder_cp = torch.load(color_decoder_dir, map_location='cpu')
    msg = color_decoder_model.load_state_dict(color_decoder_cp, strict=False)
    print(msg)
    color_decoder_model = color_decoder_model.to(device)
    color_decoder_model.eval()
    color_decoder_model = color_decoder_model
    x = torch.onnx.export(color_decoder_model,
                          (torch.randn(1,197,768,device=device),torch.randn(1,512,device=device),torch.randn(1,3,224,224,device=device)),
                          export_onnx_file,
                          verbose=False,
                          input_names=['input_mae','input_clip','input_color'],
                          output_names=['output'],
                          opset_version=12,
                          do_constant_folding=True,
                          #dynamic_axes={'input':{0:"batch_size"},"output":{0:"batch_size"},}
                          )
    net = onnx.load(export_onnx_file)
    onnx.checker.check_model(net)
    onnx.helper.printable_graph(net.graph)

    #test onnx model
    session = onnxruntime.InferenceSession(export_onnx_file, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    input_mae = np.random.rand(1,197,768).astype('float32')
    input_clip = np.random.rand(1, 512).astype('float32')
    input_color = np.random.rand(1, 3, 224, 224).astype('float32')
    out_r = session.run(None, {"input_mae":input_mae,"input_clip":input_clip,"input_color":input_color})
    print(out_r[0].shape)
    print(np.mean(out_r[0]))
    torch.set_printoptions(precision=10)
    input_mae_tensor = torch.from_numpy(input_mae).cuda()
    input_clip_tensor = torch.from_numpy(input_clip).cuda()
    input_color_tensor = torch.from_numpy(input_color).cuda()

    out = color_decoder_model( input_mae_tensor,input_clip_tensor,input_color_tensor)
    print(out.size())
    print(torch.mean(out))

def get_super_color_onnx_model():
    from super_color import SuperColor
    export_onnx_file = './onnx_models/super_color_onnx.onnx'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    super_color_dir = './pytorch_models/super_color.pth'
    super_color_model = SuperColor(kernel_size=5, group=4)
    super_color_checkpoint = torch.load(super_color_dir, map_location='cpu')
    msg = super_color_model.load_state_dict(super_color_checkpoint, strict=False)
    super_color_model = super_color_model.to(device)
    super_color_model.eval()
    super_color_model = super_color_model
    print(msg)

    x = torch.onnx.export(super_color_model,
                          (torch.randn(1, 3, 448, 448, device='cuda'),
                           torch.randn(1, 3, 448, 448, device='cuda'),
                           torch.randn(1, 3, 448, 448, device='cuda')),
                          export_onnx_file,
                          verbose=False,
                          input_names=['input_interp', 'input_gray','input_color'],
                          output_names=['output'],
                          opset_version=12,
                          do_constant_folding=True,
                          dynamic_axes={'input_interp':{2:"width",3:"height"},'input_gray':{2:"width",3:"height"},'input_color':{2:"width",3:"height"},"output":{2:"width",3:"height"},}
                          )
    print(x)
    net = onnx.load(export_onnx_file)
    onnx.checker.check_model(net)
    onnx.helper.printable_graph(net.graph)

    #test inputs with different resolutions
    session = onnxruntime.InferenceSession(export_onnx_file, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    # test 448x448 inputs
    input_interp = np.random.rand(1, 3,448, 448).astype('float32')
    input_gray = np.random.rand(1,3, 448, 448).astype('float32')
    input_color = np.random.rand(1,3, 448, 448).astype('float32')

    out_r = session.run(None, {"input_interp": input_interp, "input_gray": input_gray,"input_color":input_color})
    print(out_r[0].shape)
    print(np.mean(out_r[0]))
    torch.set_printoptions(precision=10)
    input_interp = torch.from_numpy(input_interp).cuda()
    input_gray = torch.from_numpy(input_gray).cuda()
    input_color = torch.from_numpy(input_color).cuda()


    out = super_color_model(input_interp, input_gray,input_color)
    print(out.size())
    print(torch.mean(out))

    #test 896x896 inputs
    input_interp = np.random.rand(1, 3, 896, 896).astype('float32')
    input_gray = np.random.rand(1, 3, 896, 896).astype('float32')
    input_color = np.random.rand(1, 3, 896, 896).astype('float32')

    out_r = session.run(None, {"input_interp": input_interp, "input_gray": input_gray,"input_color":input_color})
    print(out_r[0].shape)
    print(np.mean(out_r[0]))
    torch.set_printoptions(precision=10)
    input_interp = torch.from_numpy(input_interp).cuda()
    input_gray = torch.from_numpy(input_gray).cuda()
    input_color = torch.from_numpy(input_color).cuda()

    out = super_color_model(input_interp, input_gray,input_color)
    print(out.size())
    print(torch.mean(out))


if __name__=='__main__':
    #get_mae_onnx_model()
    get_color_decoder_onnx_model()
    #get_clip_onnx_model()
    #get_super_color_onnx_model()


