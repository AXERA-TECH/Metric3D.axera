import argparse
import cv2
import numpy as np
import onnxruntime as ort


MIN_VAL=0.1
MAX_VAL=200
REG_SCALE=100

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img",
        type=str,
        required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model.",
    )
    
    return parser.parse_args()

def relu(x):
    return np.maximum(0, x)

def clamp(x):
    min_val = MIN_VAL
    max_val = MAX_VAL
    y = relu(x - min_val) + min_val
    y = max_val - relu(max_val - y)
    return y

def infer(img: str, model: str, viz: bool = False):
    img_raw = cv2.imread(img)
    image = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB) 
    orig_h, orig_w = image.shape[:2]
    image = cv2.resize(image, (1064,616) )
    
    mean = np.array([123.675, 116.28, 103.53],dtype=np.float32).reshape(1,1,3)
    std = np.array([58.395, 57.12, 57.375],dtype=np.float32).reshape(1,1,3)

    image = (image-mean)/std 
    image = image.transpose(2,0,1)
    image = image[None]

    session = ort.InferenceSession(
        model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    depth = session.run(None, {"image": image})[0]
    depth = clamp(depth*REG_SCALE +MAX_VAL )

    depth = cv2.resize(depth[0, 0], (orig_w, orig_h))
    depth = 1.0/depth
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    combined_result = cv2.hconcat([img_raw,  depth_color])
            
    cv2.imwrite("output-onnx.png", combined_result)


    
    return depth


if __name__ == "__main__":
    args = parse_args()
    infer(**vars(args))
