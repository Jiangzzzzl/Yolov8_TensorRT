# Yolov8_TensorRT

## 一、环境准备

所需：

Jetson Xavier NX

Jetpack 5.1.x

python 3.8

cuda11.4



### 1.torch安装

若使用上述版本，应安装 PyTorch v2.1.0

https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

```
pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

### 2.其他

```
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libavdevice-dev libavfilter-dev libavresample-dev
```

### 3.torchvision安装

```
git clone --branch v0.16.0 https://github.com/pytorch/vision.git
```

进入torchvision文件夹：

```
export CUDA_HOME=/usr/local/cuda-11.4
export TORCH_CUDA_ARCH_LIST="7.2"  # Jetson Xavier NX 的架构号是7.2
```

```
 python3 setup.py install
```

若出现：

```
FORCE_CUDA: False
```

则：

```
FORCE_CUDA=1 python3 setup.py install
```



### 3.验证环境

```
nvcc -V        # 确保 CUDA 11.4
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```



## 二、安装YOLOv8依赖

### 1.更新apt和pip

```
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip -y
pip3 install --upgrade pip
```

### 2.安装依赖库

```
pip3 install numpy opencv-python
pip3 install matplotlib tqdm
pip3 install scipy PyYAML seaborn
pip3 install pandas ultralytics
```

 

## 三、TensorRT 加速

### 1.安装 TensorRT 导出支持包

```
pip install onnx onnxruntime onnxsim
pip install nvidia-pyindex
pip install nvidia-tensorrt --extra-index-url https://pypi.nvidia.com
```

### 2.YOLOv8 导出 TensorRT 模型

```
yolo export model=yolov8n.pt format=engine device=0
```





## 四、推理

安装依赖：

```
pip3 install pycuda
```

运行 rt_test.py


