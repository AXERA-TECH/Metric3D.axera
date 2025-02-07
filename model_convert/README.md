# 模型转换

## 创建虚拟环境

```
conda create -n metric3dv2 python=3.11 -y
conda activate metric3dv2
```

## 安装依赖

```
pip install -r requirements_v2.txt
```

## 导出模型（PyTorch -> ONNX）

导出 small 模型
```
python export_onnx.py metric3d_vit_small
```

导出成功后会生成 `metric3d_vit_small.onnx`.  

## 转换模型（ONNX -> Axera）

使用模型转换工具 `Pulsar2` 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 `.axmodel`，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

### 下载量化数据集
```
bash download_dataset.sh
```
这个模型的输入是单张图片，比较简单，这里我们直接下载打包好的图片数据  

### 模型转换

#### 修改配置文件
 
检查`config.json` 中 `calibration_dataset` 字段，将该字段配置的路径改为上一步下载的量化数据集存放路径  

#### Pulsar2 build

参考命令如下：


```
pulsar2 build --input metric3d_vit_small.onnx --config config_u16.json --output_dir build-output-u16 --output_name metric3d_vit_small_u16.axmodel  --target_hardware AX650 --compiler.check 0
```
