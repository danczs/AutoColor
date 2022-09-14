# AutoColor
![pytorch](https://img.shields.io/badge/pytorch-v1.9.0-green.svg?style=plastic)

## 简介
该项目用于漫画的自动上色。 项目主要基于 [clip](https://github.com/Lednik7/CLIP-ONNX), [MAE](https://github.com/facebookresearch/mae) 和 [timm](https://github.com/rwightman/pytorch-image-models).

## 使用说明
### 方法1、直接下载打包好的软件（适用win10系统)
下载地址: [https://github.com/danczs/AutoColor/releases/download/v0.1.0/AutoColor_win10.zip](https://github.com/danczs/AutoColor/releases/download/v0.1.0/AutoColor_win10.zip)

（通过onnxruntime部署，通过pyinstaller打包的python程序）
### 方法2、通过执行python程序使用
a. clone 或下载项目代码到本地（需要pytorch、onnxruntime等环境）:
```bash
git clone https://github.com/danczs/AutoColor.git
```
b. 安装相关包:
```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```
c. 分别下载 [pytorch模型和onnx模型](https://github.com/danczs/AutoColor/releases/download/v0.1.0/models_for_deployment.zip) 到文件夹 ```./deployment/pytorch_models```和```./deployment/onnx_models```

d. 启动GUI界面程序
```bash
cd deployment
python auto_color_gui.py
```
其中可以通过 ```auto_color_gui.py``` line14中的代码选择使用ONNX模型还是pytorch模型
```bash
# using pytorch or onnx deployment
from autocolor_pytorch_deployment import AutoColorDeployment # pytorch deployment
#from autocolor_onnx_deployment import AutoColorDeployment # onnx deployment
```
## 模型训练
适用于模型的修改与重新训练
### 数据准备
需要下载Danbooru数据集的 [kaggle子集](https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations/code)，约40G。
数据集目录：
```bash
/path/to/archive/
  danbooru-images/
    0000
      1.jpg
      2.jpg
      ...
    0001
    ...
    0149
  danbooru-metadata/
  moeimouto-faces/
```
我们只需使用danbooru-images中的前10个文件夹（0000-0009，共约2万张图片）进行训练，将该10个文件夹单独拷贝到一个新的文件夹（e.g.,archive_subset10)

### 模型准备
下载相应的 [预训练模型](https://github.com/danczs/AutoColor/releases/download/v0.1.0/models_for_training.zip) 到```./models```文件夹，以便后续的特征提取和模型训练

### 特征准备
a. 提取clip特征并保存
```bash
python get_clip_features.py --data_path = /path/to/archive_subset10
```
提取的 clip 特征会被保存到 ```./features/clip_features_subset.npy```

b. 提取mae特征并保存
```bash
python get_mae_features.py --data_path = /path/to/archive_subset10 --output_path=/path/to/mae_features
```
mae特征会被保存到```--output_path```文件夹，每张图片特征被保存为一个.npy文件，一共约12G。
这些特征文件的索引被保存在```./feature/mae_feature_names.txt```

### 训练模型
#### 训练 color deocder模型
```bash
python auto_color_main.py --grad_state = '010'
                          --output_dir = /path/to/output
                          --mae_model_path=./models/mae_visualize_vit_base.pth \
                          --mae_feature_path=./features/mae_feature_names.txt \
                          --clip_feature_path=./features/features/clip_features_subset.npy \
                          --colordecoder_model_path=./models/color_decoder_pretrained.pth
```
其中```--colordecoder_model_path```可以不设置，不设置时使用```mae_visualize_vit_base.pth```的decoder进行初始化。
这里的```./models/color_decoder_pretrained.pth```是在数据集0000-0049文件（约10万张图片）上的预训练模型，会略微提升最终性能。
输出模型保存在```--output_dir```。
#### 训练 super color 模型
super color 模型可以单独进行训练（较快），也可以结合训练好的 color decoder 一起训练，二者整体效果差别不大。

a. 单独训练

其输入输出为： 低分辨率彩图 + 高分辨率灰度图 --> 高分辨率彩图
```bash
python auto_color_main.py --grad_state='001'
                          --mae_feature_path=./features/mae_feature_names.txt \
                          --clip_feature_path=./features/features/clip_features_subset.npy \
                          --supercolor_only
```
b. 基于训练好的color decoder的输出进行训练

其输入输出为： color_decoder输出的低分辨率彩图 + 高分辨率灰度图 --> 高分辨率彩图
```bash
python auto_color_main.py --grad_state='001'
                          --batch_size=32
                          --mae_feature_path=./features/mae_feature_names.txt \
                          --clip_feature_path=./features/features/clip_features_subset.npy \
                          --colordecoder_model_path=/path/to/trained_colordeocder_model.pth
```
## 模型部署
### pytorch部署
a. 将训练好的color decoder模型和super color模型拷贝到 ```deployment/pytorch_models```中
    分别命名为```color_decoder.pth``` 和```super_color.pth```

b. 确保```auto_color_gui.py```中line14使用的是pytorch deployment模块
```bash
# using pytorch or onnx deployment
from autocolor_pytorch_deployment import AutoColorDeployment # pytorch deployment
#from autocolor_onnx_deployment import AutoColorDeployment # onnx deployment
```
c. 启动界面
```bash
cd deployment
python auto_color_gui.py
```

### onnx部署
a. 使用```conert_to_onnx_models.py```将 pytorch 模型转换为 onnx 模型，
需转换的```color_decoder.pth```模型和```super_color.pth```模型需放在```deployment\pytorch_models ```文件夹下：
```bash
python conert_to_onnx_models.py
```
b. 确保```auto_color_gui.py```中line14使用的是onnx deployment模块
```bash
# using pytorch or onnx deployment
#from autocolor_pytorch_deployment import AutoColorDeployment # pytorch deployment
from autocolor_onnx_deployment import AutoColorDeployment # onnx deployment
```
c. 启动界面
```bash
cd deployment
python auto_color_gui.py
```
### 基于onnx部署进行打包
```bash
pip install pyinstaller
cd deployment
python -m PyInstaller -F -w -i feather_icon.ico auto_color_gui.py --add-data "\\path\\to\\.conda\\envs\\your_env_name\\Lib\\site-packages\\onnxruntime\\capi\\*.dll;onnxruntime\\capi"
```
将```onnx_models,example_white.jpg, feather_icon.ico```拷贝到```deployment\dist ```文件夹下:
```bash
deployment\dist\
    onnx_models
        color_decoder_onnx.onnx
        super_color_onnx.onnx
        ...
    auto_color_gui.exe
    example_white.jpg
    feather_icon.ico
```
运行```auto_color_gui.exe```

## citing
```bash
@misc{chen2022autocolor,
  author = {Chen, Zhengsu},
  title = {Auto Color},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/danczs/AutoColor}}
}
```