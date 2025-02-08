"""
Export the torch hub model to ONNX format. Normalization is done in the model.
"""

import torch
import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import onnx
from onnx import helper

class Metric3DExportModel(torch.nn.Module):
    """
    The model for exporting to ONNX format. Add custom preprocessing and postprocessing here.
    """

    def __init__(self, meta_arch):
        super().__init__()
        self.meta_arch = meta_arch
        self.register_buffer(
            "rgb_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).cuda()
        )
        self.register_buffer(
            "rgb_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).cuda()
        )
        self.input_size = (616, 1064)

    def normalize_image(self, image):
        image = image - self.rgb_mean
        image = image / self.rgb_std
        return image

    def forward(self, image):
        # image = self.normalize_image(image)
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.meta_arch.inference(
                {"input": image}
            )
        return pred_depth


def update_vit_sampling(model):
    """
    For ViT models running on some TensorRT version, we need to change the interpolation method from bicubic to bilinear.
    """
    import torch.nn as nn
    import math

    def interpolate_pos_encoding_bilinear(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(sx, sy),
            mode="bilinear",  # Change from bicubic to bilinear
            antialias=self.interpolate_antialias,
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    model.depth_model.encoder.interpolate_pos_encoding = (
        interpolate_pos_encoding_bilinear.__get__(
            model.depth_model.encoder, model.depth_model.encoder.__class__
        )
    )
    return model




def modify_onnx_output(input_model_path, output_model_path, 
                      new_output_node_name="/depth_model/decoder/Slice_6",
                      original_output_name="pred_depth"):
    # 加载原始模型
    model = onnx.load(input_model_path)
    
    # 运行形状推断以获取中间节点的信息
    model = infer_shapes(model)
    
    # 查找目标节点
    target_node = None
    for node in model.graph.node:
        if node.name == new_output_node_name:
            target_node = node
            break
    
    if target_node is None:
        raise ValueError(f"未找到名为 {new_output_node_name} 的节点")
    
    # 获取目标节点的输出名称（假设取第一个输出）
    new_output_name = target_node.output[0]
    
    # 查找对应的ValueInfoProto
    new_output_value_info = None
    for value_info in model.graph.value_info:
        if value_info.name == new_output_name:
            new_output_value_info = value_info
            break
    
    # 如果找不到则创建默认的ValueInfo
    if new_output_value_info is None:
        new_output_value_info = helper.make_tensor_value_info(
            name=new_output_name,
            elem_type=onnx.TensorProto.FLOAT,  # 根据实际情况修改数据类型
            shape=None  # 可以保持未知形状
        )
    
    # 移除原始输出
    for i in reversed(range(len(model.graph.output))):
        if model.graph.output[i].name == original_output_name:
            del model.graph.output[i]
    
    # 添加新的输出
    model.graph.output.append(new_output_value_info)
    
    # 删除目标节点之后的所有节点
    # 首先收集需要保留的节点
    nodes_to_keep = []
    for node in model.graph.node:
        if node.name == new_output_node_name:
            nodes_to_keep.append(node)
            break
        nodes_to_keep.append(node)
    
    # 清空图中的所有节点
    del model.graph.node[:]
    
    # 将需要保留的节点重新添加到图中
    model.graph.node.extend(nodes_to_keep)
    
    # 保存修改后的模型
    onnx.save(model, output_model_path)
    print(f"模型已保存至 {output_model_path}，新的输出节点为：{new_output_name}")



def main(model_name="metric3d_vit_small", modify_upsample=False):
    model = torch.hub.load("yvanyin/metric3d", model_name, pretrain=True)
    model.cuda().eval()

    if modify_upsample:
        model = update_vit_sampling(model)

    B = 1
    if "vit" in model_name:
        dummy_image = torch.randn([B, 3, 616, 1064]).cuda()
    else:
        dummy_image = torch.randn([B, 3, 544, 1216]).cuda()

    export_model = Metric3DExportModel(model)
    export_model.eval()
    export_model.cuda()

    onnx_output = f"{model_name}.onnx"
    dummy_input = (dummy_image,)
    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_output,
        input_names=["image"],
        output_names=["pred_depth"],
        opset_version=11,
    )

    onnx_model = onnx.load(onnx_output)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_output)
    print("onnx simpilfy successed, and model saved in {}".format(onnx_output))


    modify_onnx_output(onnx_output, onnx_output, 
                      new_output_node_name="/depth_model/decoder/Slice_6",
                      original_output_name="pred_depth")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
