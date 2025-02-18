# Metric3D.axera
Metric3D v2 DEMO on Axera

- 目前支持  Python 语言，C++代码在开发中  
- 预编译模型下载[models](https://github.com/AXERA-TECH/Metric3D.axera/releases/download/v1.0.0/models.tar.gz)，如需自行转换请参考[模型转换](/model_convert/README.md)

## 支持平台

- [x] AX650N
- [ ] AX630C

## 模型转换

[模型转换](./model_convert/README.md)

## 上板部署

- AX650N 的设备已预装 Ubuntu22.04
- 以 root 权限登陆 AX650N 的板卡设备
- 链接互联网，确保 AX650N 的设备能正常执行 `apt install`, `pip install` 等指令
- 已验证设备：AX650N DEMO Board、爱芯派Pro(AX650N)

### Python API 运行

#### Requirements

```
mkdir /opt/site-packages
cd python
pip3 install -r requirements.txt --prefix=/opt/site-packages
``` 

#### 添加环境变量

将以下两行添加到 `/root/.bashrc`(实际添加的路径需要自行检查)后，重新连接终端或者执行 `source ~/.bashrc`

```
export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages  
export PATH=$PATH:/opt/site-packages/local/bin
``` 

#### 运行

##### 基于 ONNX Runtime 运行  
可在开发板或PC运行 

在开发板或PC上，运行以下命令  
```  
cd python
python3 infer_onnx.py --img examples/demo01.jpg --model ../models/metric3d_vit_small.onnx
```
输出结果
![output](asserts/output-onnx.png)

##### 基于AXEngine运行  
在开发板上运行命令

```
cd python  
python3 infer.py --img examples/demo01.jpg --model ../models/metric3d_vit_small_u16.axmodel 
```  
输出结果
![output](asserts/output-ax.png)


运行参数说明:  
| 参数名称 | 说明  |
| --- | --- | 
| --img | 输入图片路径 | 
| --model | 模型路径 | 

### Latency

#### AX650N

| model | latency(ms) |
|---|---|
|metric3d_vit_small_u16|416.7|



## 技术讨论

- Github issues
- QQ 群: 139953715
