import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from pathlib import Path
import os
import sys
import time
import math
import traceback

# ---------------- 配置 ----------------
ENGINE_FILE_PATH = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\unet_dynamic_batch_fp16(batchsize=16).engine"
INPUT_TENSOR_NAME = "input"
OUTPUT_TENSOR_NAME = "output"

INPUT_FOLDER = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\img_5\test"
OUTPUT_FOLDER = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\img_5\test_output"

MAX_BATCH_SIZE = 16
C, H, W = 3, 512, 512
STRIDE = 256

# ---------- Host/Device 内存封装 ----------
class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

# ---------- allocate_buffers ----------
def allocate_buffers(engine, batch_size):
    buffers_by_index = []
    binding_ptrs = []
    stream = cuda.Stream()

    for idx in range(engine.num_bindings):
        shape = list(engine.get_binding_shape(idx))
        if len(shape) > 0 and shape[0] == -1:
            shape[0] = batch_size
        elem_count = int(trt.volume(shape))
        dtype = trt.nptype(engine.get_binding_dtype(idx))
        host_mem = cuda.pagelocked_empty(elem_count, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        buffers_by_index.append(HostDeviceMem(host_mem, device_mem))
        binding_ptrs.append(int(device_mem))

    return buffers_by_index, binding_ptrs, stream

# ---------- do_inference ----------
def do_inference(context, buffers_by_index, binding_ptrs, stream):
    engine = context.engine
    for idx in range(engine.num_bindings):
        if engine.binding_is_input(idx):
            cuda.memcpy_htod_async(buffers_by_index[idx].device, buffers_by_index[idx].host, stream)
    context.execute_async_v2(bindings=binding_ptrs, stream_handle=stream.handle)
    for idx in range(engine.num_bindings):
        if not engine.binding_is_input(idx):
            cuda.memcpy_dtoh_async(buffers_by_index[idx].host, buffers_by_index[idx].device, stream)
    stream.synchronize()

    outputs = {}
    for idx in range(engine.num_bindings):
        if not engine.binding_is_input(idx):
            name = engine.get_binding_name(idx)
            try:
                out_shape = tuple(context.get_binding_shape(idx))
            except Exception:
                out_shape = tuple(engine.get_binding_shape(idx))
            outputs[name] = np.array(buffers_by_index[idx].host, copy=False).reshape(out_shape)
    return outputs

# ---------- 图像预处理 ----------
def preprocess_patch(img):
    img = img.astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    return img

# ---------- 构建 patch 坐标 ----------
def generate_patch_coords(h, w, patch_h, patch_w, stride):
    coords = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = y
            x1 = x
            y2 = min(y + patch_h, h)
            x2 = min(x + patch_w, w)
            coords.append((y1, y2, x1, x2))
    return coords

# ---------- 主推理函数 ----------
def trt_overlap_inference(input_dir, output_dir):
    image_files = sorted([p for p in Path(input_dir).glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not image_files:
        print("No images found.")
        return

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(ENGINE_FILE_PATH, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    input_idx = engine.get_binding_index(INPUT_TENSOR_NAME)
    output_idx = engine.get_binding_index(OUTPUT_TENSOR_NAME)

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        orig_h, orig_w = img.shape[:2]
        print(f"Processing {img_path.name} ({orig_w}x{orig_h})")

        # padding 到 stride 对齐
        pad_h = math.ceil((orig_h - H) / STRIDE) * STRIDE + H
        pad_w = math.ceil((orig_w - W) / STRIDE) * STRIDE + W
        pad_img = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        pad_img[0:orig_h, 0:orig_w, :] = img

        coords = generate_patch_coords(pad_h, pad_w, H, W, STRIDE)

        # 准备输出概率图
        num_classes = engine.get_binding_shape(output_idx)[1] if len(engine.get_binding_shape(output_idx)) > 1 else 1
        prob_map = np.zeros((num_classes, pad_h, pad_w), dtype=np.float32)
        count_map = np.zeros((num_classes, pad_h, pad_w), dtype=np.float32)

        # 批量推理
        for i in range(0, len(coords), MAX_BATCH_SIZE):
            batch_coords = coords[i:i + MAX_BATCH_SIZE]
            batch_size = len(batch_coords)
            batch_data = np.zeros((batch_size, C, H, W), dtype=np.float32)
            for j, (y1, y2, x1, x2) in enumerate(batch_coords):
                patch = pad_img[y1:y2, x1:x2, :]
                patch = cv2.resize(patch, (W, H))
                batch_data[j] = preprocess_patch(patch)
            # 设置动态 shape
            context.set_binding_shape(input_idx, (batch_size, C, H, W))
            buffers_by_index, binding_ptrs, stream = allocate_buffers(engine, batch_size)
            np.copyto(buffers_by_index[input_idx].host, batch_data.ravel())
            outputs = do_inference(context, buffers_by_index, binding_ptrs, stream)
            out_batch = outputs[OUTPUT_TENSOR_NAME]

            # 概率融合
            for j, (y1, y2, x1, x2) in enumerate(batch_coords):
                out_patch = out_batch[j]
                out_patch = cv2.resize(out_patch.transpose(1,2,0), (x2 - x1, y2 - y1)).transpose(2,0,1)
                prob_map[:, y1:y2, x1:x2] += out_patch
                count_map[:, y1:y2, x1:x2] += 1

            del buffers_by_index, binding_ptrs, stream

        # 平均概率
        prob_map /= np.maximum(count_map, 1e-6)
        pred_mask = np.argmax(prob_map, axis=0).astype(np.uint8)
        pred_mask = pred_mask[0:orig_h, 0:orig_w]

        # 保存
        output_file = Path(output_dir) / f"{img_path.stem}.png"
        cv2.imwrite(str(output_file), (pred_mask * 255 // pred_mask.max()).astype(np.uint8))
        print(f"Saved {output_file.name}")

# ---------- 主入口 ----------
if __name__ == "__main__":
    trt_overlap_inference(INPUT_FOLDER, OUTPUT_FOLDER)
