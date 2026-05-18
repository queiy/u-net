# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
#
# # --- 配置参数 ---
# ENGINE_FILE_PATH = "unet_dynamic_batch_fp16.engine"
# INPUT_TENSOR_NAME = "input"  # 您的 ONNX 模型输入名称
# OUTPUT_TENSOR_NAME = "output"  # 您的 ONNX 模型输出名称
#
#
# # ----------------------------------------------------------
# # 核心类：用于管理 TensorRT 引擎的内存和执行上下文
# # ----------------------------------------------------------
# class HostDeviceMem:
#     """封装主机(CPU)和设备(GPU)内存"""
#
#     def __init__(self, host_mem, device_mem):
#         self.host = host_mem
#         self.device = device_mem
#
#     def __str__(self):
#         return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
#
#
# def allocate_buffers(engine, batch_size):
#     """
#     为动态 Batch size 分配输入/输出缓冲区。
#     TensorRT 动态形状需要特殊的内存分配和流管理。
#     """
#     buffers = []
#     bindings = {}
#
#     # 创建 CUDA 流
#     stream = cuda.Stream()
#
#     # 遍历引擎中的所有绑定 (输入/输出)
#     for binding in engine:
#         # 获取 TensorRT 要求的输入/输出维度 (动态形状时，第0维将是 -1)
#         size = trt.volume(engine.get_binding_shape(binding))
#
#         # 确保动态 Batch Size 被替换为实际的 batch_size
#         if engine.is_shape_binding(binding):
#             # 形状输入不需要分配内存
#             continue
#
#         # 动态形状的核心处理：计算实际的缓冲区大小
#         size *= batch_size  # 缓冲区总大小 = Batch Size * C * H * W
#
#         # 获取数据类型 (您的模型应该是 FP32 或 FP16)
#         dtype = trt.nptype(engine.get_binding_dtype(binding))
#
#         # 分配主机和设备内存
#         host_mem = cuda.pagelocked_empty(size, dtype)
#         device_mem = cuda.mem_alloc(host_mem.nbytes)
#
#         # 将缓冲区添加到列表和字典
#         buffers.append(HostDeviceMem(host_mem, device_mem))
#         bindings[binding] = device_mem  # 绑定 GPU 内存地址
#
#     return buffers, bindings, stream
#
#
# def do_inference(context, buffers, bindings, stream, batch_size):
#     """
#     执行 TensorRT 推理。
#     """
#     # 1. 设置输入绑定 (Host -> Device)
#     for host_buffer, device_buffer in buffers:
#         # 将数据从 Host 传输到 Device
#         cuda.memcpy_htod_async(device_buffer.device, host_buffer.host, stream)
#
#     # 2. 执行推理 (异步)
#     # 输入绑定的顺序与引擎中的顺序一致
#     binding_ptrs = [b.device for b in buffers]
#
#     # context.execute_v2 适用于动态和静态形状
#     context.execute_async_v2(bindings=binding_ptrs, stream_handle=stream.handle)
#
#     # 3. 获取输出绑定 (Device -> Host)
#     for host_buffer, device_buffer in buffers:
#         # 将结果从 Device 传输回 Host
#         cuda.memcpy_dtoh_async(host_buffer.host, device_buffer.device, stream)
#
#     # 4. 同步 CUDA 流
#     stream.synchronize()
#
#     # 5. 提取输出
#     output_shape = context.get_binding_shape(1)  # 假设输出是第1个绑定
#     output_size = trt.volume(output_shape)
#
#     # 假设输出缓冲区是 buffers 中的最后一个
#     output_buffer = buffers[-1].host
#
#     # 重塑输出张量
#     return output_buffer[:output_size].reshape(output_shape)
#
#
# # ----------------------------------------------------------
# # 主执行函数
# # ----------------------------------------------------------
# def trt_inference_example(batch_size_to_test):
#     # 1. 创建 Logger 和 Runtime
#     TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#     runtime = trt.Runtime(TRT_LOGGER)
#
#     # 2. 加载序列化引擎
#     print(f"Loading engine from {ENGINE_FILE_PATH}...")
#     with open(ENGINE_FILE_PATH, "rb") as f:
#         serialized_engine = f.read()
#     engine = runtime.deserialize_cuda_engine(serialized_engine)
#
#     # 3. 创建执行上下文 (Execution Context)
#     context = engine.create_execution_context()
#
#     # --- 关键的动态形状设置 ---
#     # 设置输入张量的实际 Batch/尺寸
#     # 形状为 (B, C, H, W) = (batch_size_to_test, 3, 512, 512)
#     new_input_shape = (batch_size_to_test, 3, 512, 512)
#
#     # 检查请求的 Batch Size 是否在您构建引擎时设置的动态范围内 (1-8)
#     if not context.set_binding_shape(engine.get_binding_index(INPUT_TENSOR_NAME), new_input_shape):
#         print(f"Error: Requested batch size {batch_size_to_test} is outside the allowed range (1-8).")
#         return
#
#     print(f"Successfully set input shape to: {context.get_binding_shape(engine.get_binding_index(INPUT_TENSOR_NAME))}")
#
#     # 4. 分配缓冲区 (根据设置的动态形状)
#     buffers, bindings, stream = allocate_buffers(engine, batch_size_to_test)
#
#     # 5. 准备输入数据 (示例数据)
#     # 假设输入数据的 dtype 与引擎输入 dtype 一致 (通常是 np.float16 或 np.float32)
#     input_dtype = trt.nptype(engine.get_binding_dtype(INPUT_TENSOR_NAME))
#
#     # 创建随机输入数据，并将其放入输入缓冲区
#     input_index = engine.get_binding_index(INPUT_TENSOR_NAME)
#     input_host_mem = buffers[input_index].host
#
#     # 填充输入缓冲区（这里用随机数据模拟输入）
#     np.random.rand(*new_input_shape).astype(input_dtype).tofile(input_host_mem)
#
#     # 6. 执行推理
#     print(f"Running inference with Batch Size {batch_size_to_test}...")
#     output_array = do_inference(context, buffers, bindings, stream, batch_size_to_test)
#
#     # 7. 清理和输出结果
#     print(f"Inference finished. Output shape: {output_array.shape}")
#     # print("Sample Output values:", output_array[0, :5, 200, 200]) # 打印第一个样本的局部值
#
#     # 释放 PyCUDA 内存
#     del buffers, bindings, stream, context, engine, runtime
#
#
# # --- 调用示例 ---
# # 尝试使用 Batch Size 4 进行推理 (您在构建时设置为 Optimal)
# trt_inference_example(batch_size_to_test=4)
#
# # 尝试使用 Batch Size 1 进行推理 (您在构建时设置为 Minimum)
# # trt_inference_example(batch_size_to_test=1)
#
# # 尝试使用 Batch Size 8 进行推理 (您在构建时设置为 Maximum)
# # trt_inference_example(batch_size_to_test=8)

# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
# import cv2
# from pathlib import Path
# import os
#
# # --- 配置参数 ---
# ENGINE_FILE_PATH = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\unet_dynamic_batch_fp16.engine"
# INPUT_TENSOR_NAME = "input"  # 您的 ONNX 模型输入名称
# OUTPUT_TENSOR_NAME = "output"  # 您的 ONNX 模型输出名称
#
# # =================================================================
# # !!! 核心配置: 定义输入和输出文件夹 !!!
# # =================================================================
# # 请替换为您的实际路径
# INPUT_FOLDER = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\img"  # <-- 替换为您的输入图片文件夹
# OUTPUT_FOLDER = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\output_masks"  # <-- 替换为您的输出结果文件夹
#
# MAX_BATCH_SIZE = 4  # 您的引擎最大 Batch Size
#
#
# # ----------------------------------------------------------
# # 核心类：用于管理 TensorRT 引擎的内存和执行上下文
# # ----------------------------------------------------------
# class HostDeviceMem:
#     """封装主机(CPU)和设备(GPU)内存"""
#
#     def __init__(self, host_mem, device_mem):
#         self.host = host_mem
#         self.device = device_mem
#
#
# # ----------------------------------------------------------
# # !!! 修复后的 allocate_buffers 函数 (解决 TypeError) !!!
# # ----------------------------------------------------------
# def allocate_buffers(engine, batch_size):
#     """
#     为动态 Batch size 分配输入/输出缓冲区。
#     修复了 TypeError: is_shape_binding() 必须传入整数索引。
#     """
#     buffers = []
#     bindings = {}
#     stream = cuda.Stream()
#
#     # 获取所有绑定名称 (TensorRT 8.x+)
#     binding_names = engine.get_tensor_names()
#
#     # 遍历所有绑定名称的索引
#     for binding_index in range(len(binding_names)):
#
#         binding_name = binding_names[binding_index]
#
#         # 1. 使用新的 API 获取形状
#         size = trt.volume(engine.get_tensor_shape(binding_name))
#
#         # 2. 检查是否为形状绑定 (使用索引 INT)
#         if engine.is_shape_binding(binding_index):
#             continue
#
#         # 3. 检查并计算总大小
#         # 如果维度0是动态的 (即 Batch size)，我们需要用当前的 batch_size 重新计算总大小
#         if engine.get_tensor_dimension_type(binding_name, 0) == trt.TensorDimensionType.RUNTIME:
#             size = size * batch_size
#
#             # 4. 获取数据类型 (使用新的 API)
#         dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
#
#         # 分配主机和设备内存
#         host_mem = cuda.pagelocked_empty(size, dtype)
#         device_mem = cuda.mem_alloc(host_mem.nbytes)
#
#         # 将缓冲区添加到列表和字典
#         buffers.append(HostDeviceMem(host_mem, device_mem))
#         bindings[binding_name] = device_mem  # 绑定 GPU 内存地址
#
#     return buffers, bindings, stream
#
#
# # ----------------------------------------------------------
# # 修复后的 do_inference 函数 (解决 binding 顺序问题)
# # ----------------------------------------------------------
# def do_inference(context, buffers, bindings, stream, batch_size):
#     """
#     执行 TensorRT 推理，确保 binding 顺序正确。
#     """
#     # 1. 设置输入绑定 (Host -> Device)
#     for host_buffer, device_buffer in buffers:
#         cuda.memcpy_htod_async(device_buffer.device, host_buffer.host, stream)
#
#     # 2. 构建 binding_ptrs 列表 (必须按 TensorRT 引擎内部的顺序排列)
#     binding_names = context.engine.get_tensor_names()
#     binding_ptrs = []
#
#     # 根据引擎的 tensor 顺序，从 bindings 字典中查找并放入 device pointer
#     for name in binding_names:
#         if name in bindings:
#             # bindings 字典中只包含需要分配内存的 Tensor (输入和输出)
#             binding_ptrs.append(bindings[name])
#         else:
#             # 处理形状绑定（如果引擎有形状输入，但我们这里没有，为兼容性保留）
#             pass
#
#             # 执行推理 (异步)
#     context.execute_async_v2(bindings=binding_ptrs, stream_handle=stream.handle)
#
#     # 3. 获取输出绑定 (Device -> Host)
#     for host_buffer, device_buffer in buffers:
#         cuda.memcpy_dtoh_async(host_buffer.host, device_buffer.device, stream)
#
#     # 4. 同步 CUDA 流
#     stream.synchronize()
#
#     # 5. 提取输出
#     # 使用新的 API get_tensor_shape 和 OUTPUT_TENSOR_NAME
#     output_shape = context.get_tensor_shape(OUTPUT_TENSOR_NAME)
#     output_size = trt.volume(output_shape)
#
#     # 假设输出缓冲区是 buffers 中的最后一个 (通常是正确的)
#     output_buffer = buffers[-1].host
#
#     # 重塑输出张量
#     return output_buffer[:output_size].reshape(output_shape)
#
#
# # ----------------------------------------------------------
# # 图像预处理函数 (保持不变)
# # ----------------------------------------------------------
# def preprocess_image(image_path, target_h, target_w):
#     """加载、缩放和归一化单张图片，使其符合模型输入要求"""
#     img = cv2.imread(str(image_path))
#     if img is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")
#     img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
#     img = img.astype(np.float32) / 255.0
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.transpose((2, 0, 1))
#     return img
#
#
# # ----------------------------------------------------------
# # 文件夹推理主函数
# # ----------------------------------------------------------
# def trt_inference_folder(input_dir, output_dir):
#     # 查找所有图片文件 (假设是 .jpg)
#     image_files = sorted([p for p in Path(input_dir).glob("*.jpg") if p.is_file()])
#     if not image_files:
#         print(f"Error: No .jpg files found in {input_dir}. Check file extension.")
#         return
#
#     # 创建输出文件夹 (自动创建)
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#
#     # 1. 创建 Logger 和 Runtime
#     TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#     runtime = trt.Runtime(TRT_LOGGER)
#
#     # 2. 加载序列化引擎
#     print(f"Loading engine from {ENGINE_FILE_PATH}...")
#     with open(ENGINE_FILE_PATH, "rb") as f:
#         serialized_engine = f.read()
#     engine = runtime.deserialize_cuda_engine(serialized_engine)
#
#     # 获取模型所需的输入尺寸 C, H, W
#     C, H, W = 3, 512, 512
#
#     # 3. 创建执行上下文
#     context = engine.create_execution_context()
#
#     # 使用 get_tensor_idx 替换 get_binding_index
#     # input_index = engine.get_tensor_idx(INPUT_TENSOR_NAME)
#     input_index = engine.get_binding_index(INPUT_TENSOR_NAME)
#
#     print(f"Total images found: {len(image_files)}. Processing in batches of max {MAX_BATCH_SIZE}...")
#
#     # ----------------------------------------------------------------------------------
#     # 核心循环: 遍历所有图片并分批处理
#     # ----------------------------------------------------------------------------------
#     for i in range(0, len(image_files), MAX_BATCH_SIZE):
#
#         current_batch_files = image_files[i:i + MAX_BATCH_SIZE]
#         batch_size = len(current_batch_files)
#
#         print(f"--- Processing Batch {i // MAX_BATCH_SIZE + 1} (Size: {batch_size}) ---")
#
#         # 动态设置 Batch Size
#         new_input_shape = (batch_size, C, H, W)
#
#         # 每次循环都必须重新设置 shape (使用 tensor 索引)
#         if not context.set_binding_shape(input_index, new_input_shape):
#             print(f"Error: Batch size {batch_size} outside allowed range.")
#             break
#
#         # 4. 重新分配缓冲区 (Batch Size 变化时需要重新分配)
#         buffers, bindings, stream = allocate_buffers(engine, batch_size)
#
#         # 5. 准备输入数据
#         # 使用 get_tensor_dtype 替换 get_binding_dtype
#         input_dtype = trt.nptype(engine.get_tensor_dtype(INPUT_TENSOR_NAME))
#
#         # 假设 input 是第一个 buffer (索引 0)
#         input_host_mem = buffers[0].host
#
#         # 预分配一个 NumPy 数组来保存所有 Batch 的数据
#         batch_data = np.empty(new_input_shape, dtype=input_dtype)
#
#         for j, path in enumerate(current_batch_files):
#             # 加载和预处理单张图片
#             processed_img = preprocess_image(path, H, W)
#             batch_data[j] = processed_img
#
#         # 将整个 Batch 的数据扁平化后复制到 TensorRT 的 host 缓冲区
#         batch_data.ravel().tofile(input_host_mem)
#
#         # 6. 执行推理
#         output_array = do_inference(context, buffers, bindings, stream, batch_size)
#
#         # 7. 清理并保存结果
#         # 假设输出形状是 (B, NUM_CLASSES, H, W)
#         for j in range(batch_size):
#             single_output = output_array[j]
#
#             # 转换为类别掩码 (例如：取最大值通道)
#             predicted_mask = np.argmax(single_output, axis=0).astype(np.uint8)
#
#             # 保存掩码文件
#             output_filename = Path(output_dir) / f"{current_batch_files[j].stem}_mask.png"
#
#             # 将掩码缩放回 0-255 范围以便保存 (如果您的类别数 > 1)
#             # 这是一个简单的缩放，如果您的类别标签从 0 开始且是连续的，这将是合理的。
#             scale_factor = 255 // np.max(predicted_mask) if np.max(predicted_mask) > 0 else 1
#             scaled_mask = predicted_mask * scale_factor
#
#             cv2.imwrite(str(output_filename), scaled_mask)
#             print(f"Saved: {output_filename.name}")
#
#         # 释放 PyCUDA 内存 (在每次循环结束时释放)
#         del buffers, bindings, stream
#
#     # 释放引擎和上下文
#     del context, engine, runtime
#     print("\nAll images processed. Done!")
#
#
# # --- 调用示例 ---
# # 运行文件夹推理模式
# trt_inference_folder(INPUT_FOLDER, OUTPUT_FOLDER)


import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from pathlib import Path
import os
import sys
import traceback
import time

# ---------------- 配置 ----------------
ENGINE_FILE_PATH = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\unet_dynamic_batch_fp16(batchsize=16).engine"
INPUT_TENSOR_NAME = "input"   # 请根据 engine 确认名字
OUTPUT_TENSOR_NAME = "output" # 请根据 engine 确认名字

INPUT_FOLDER = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\img_new"
OUTPUT_FOLDER = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\img_new_1"

MAX_BATCH_SIZE = 16 # 引擎允许的最大 batch（构建 engine 时定义）
# 假设模型输入是 (C, H, W)
C, H, W = 3, 512, 512

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
        # binding 名字与方向
        name = engine.get_binding_name(idx)
        is_input = engine.binding_is_input(idx)

        # 获取绑定形状（注意：对于动态维度，context.set_binding_shape 之后 context.get_binding_shape(idx) 才会返回具体值）
        # 这里使用 engine.get_binding_shape(idx) 仅用于参考元素个数；通常我们希望在分配前已经调用了 context.set_binding_shape
        shape = list(engine.get_binding_shape(idx))
        # 当形状里含 -1（意味着动态）时，将会在 context.set_binding_shape 后再从 context.get_binding_shape 读取实际 shape
        # 为避免错误，若第一维是 -1（动态 batch），我们先把它设为 batch_size 以计算元素总数
        if len(shape) > 0 and shape[0] == -1:
            shape[0] = batch_size

        elem_count = int(trt.volume(shape))
        dtype = trt.nptype(engine.get_binding_dtype(idx))

        # 分配 pagelocked host buffer（1D 扁平）和 device buffer
        host_mem = cuda.pagelocked_empty(elem_count, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        buffers_by_index.append(HostDeviceMem(host_mem, device_mem))
        # execute_async_v2 接受 device 指针 list；这里使用整数地址
        binding_ptrs.append(int(device_mem))

    return buffers_by_index, binding_ptrs, stream

# ---------- do_inference ----------
def do_inference(context, buffers_by_index, binding_ptrs, stream):

    engine = context.engine

    # H2D (把所有 input binding 的 host -> device)
    for idx in range(engine.num_bindings):
        if engine.binding_is_input(idx):
            host_mem = buffers_by_index[idx].host
            device_mem = buffers_by_index[idx].device
            cuda.memcpy_htod_async(device_mem, host_mem, stream)

    # 执行
    context.execute_async_v2(bindings=binding_ptrs, stream_handle=stream.handle)

    # D2H
    for idx in range(engine.num_bindings):
        if not engine.binding_is_input(idx):
            host_mem = buffers_by_index[idx].host
            device_mem = buffers_by_index[idx].device
            cuda.memcpy_dtoh_async(host_mem, device_mem, stream)

    stream.synchronize()

    # 收集输出并 reshape
    outputs = {}
    for idx in range(engine.num_bindings):
        if not engine.binding_is_input(idx):
            name = engine.get_binding_name(idx)
            # context.get_binding_shape(idx) 在 set_binding_shape 后返回实际 shape（包含 batch）
            try:
                out_shape = tuple(context.get_binding_shape(idx))
            except Exception:
                # 兜底：若获取失败则用 engine 的绑定 shape（可能含 -1）
                out_shape = tuple(engine.get_binding_shape(idx))

            flat = buffers_by_index[idx].host  # 一维扁平
            # 可能存在类型不匹配或者 size 问题，尝试 reshape
            try:
                outputs[name] = np.array(flat, copy=False).reshape(out_shape)
            except Exception:
                # reshape 失败时把扁平数组返回以便调试
                outputs[name] = np.array(flat, copy=True)

    return outputs

# ---------- 图像预处理 ----------
def preprocess_image(image_path, target_h, target_w):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))  # C,H,W
    return img


# ---------- 主推理函数 (添加计时逻辑) ----------
def trt_inference_folder(input_dir, output_dir):
    image_files = sorted(
        [p for p in Path(input_dir).glob("*") if p.is_file() and p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]])
    total_images = len(image_files)

    if not image_files:
        print(f"Error: no images found in {input_dir}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    # ------------------ 初始化和加载 ------------------
    print(f"Loading engine from {ENGINE_FILE_PATH} ...")
    with open(ENGINE_FILE_PATH, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("Failed to deserialize engine. Check the engine file and TRT version compatibility.")

    # 检查版本和绑定信息
    print(f"TensorRT version: {trt.__version__}")
    print(f"Engine has {engine.num_bindings} bindings.")

    try:
        input_idx = engine.get_binding_index(INPUT_TENSOR_NAME)
        output_idx = engine.get_binding_index(OUTPUT_TENSOR_NAME)
    except Exception as e:
        print(f"Failed to get binding index for input/output name: {e}")
        print("Available binding names and indices:")
        for idx in range(engine.num_bindings):
            print(
                f"  idx={idx}, name={engine.get_binding_name(idx)}, is_input={engine.binding_is_input(idx)}, shape={engine.get_binding_shape(idx)}")
        return

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create execution context.")

    print(f"Total images found: {total_images}. Processing in batches of max {MAX_BATCH_SIZE}...")

    # ---------- 计时变量初始化 ----------
    total_inference_time = 0.0

    # ------------------ 推理循环 ------------------
    try:
        for i in range(0, total_images, MAX_BATCH_SIZE):
            current_batch_files = image_files[i:i + MAX_BATCH_SIZE]
            batch_size = len(current_batch_files)
            print(f"--- Processing batch {i // MAX_BATCH_SIZE + 1} (size {batch_size}) ---")

            # 1. 设置动态输入 shape
            new_input_shape = (batch_size, C, H, W)
            if not context.set_binding_shape(input_idx, new_input_shape):
                print(
                    f"Error: context.set_binding_shape returned False for batch size {batch_size}. The requested shape may be out of range.")
                return

            # 2. 重新分配 buffers
            buffers_by_index, binding_ptrs, stream = allocate_buffers(engine, batch_size)

            # 3. 准备 batch 数据
            input_dtype = trt.nptype(engine.get_binding_dtype(input_idx))
            batch_data = np.empty(new_input_shape, dtype=input_dtype)
            for j, path in enumerate(current_batch_files):
                try:
                    processed = preprocess_image(path, H, W)
                except Exception as ex:
                    print(f"Failed to preprocess {path}: {ex}")
                    raise
                if processed.dtype != batch_data.dtype:
                    processed = processed.astype(batch_data.dtype)
                batch_data[j] = processed

            # 4. copy 到 host buffer
            host_buf = buffers_by_index[input_idx].host
            flat_batch = batch_data.ravel()
            if host_buf.size != flat_batch.size:
                print(f"Warning: host buffer size ({host_buf.size}) != flattened input size ({flat_batch.size}).")
                minlen = min(host_buf.size, flat_batch.size)
                host_buf[:minlen] = flat_batch[:minlen]
            else:
                np.copyto(host_buf, flat_batch)

            # 5. 执行推理并计时
            start_time = time.time()
            outputs = do_inference(context, buffers_by_index, binding_ptrs, stream)
            end_time = time.time()

            batch_time = end_time - start_time
            total_inference_time += batch_time

            avg_time_per_image = batch_time / batch_size
            batch_fps = batch_size / batch_time
            print("------------------------------------------------------------------------------")
            print(
                f"[Batch {i // MAX_BATCH_SIZE + 1}] "
                f"BatchSize={batch_size} | "
                f"BatchTime={batch_time:.4f}s | "
                f"Avg={avg_time_per_image * 1000:.2f} ms/image | "
                f"Throughput={batch_fps:.2f} FPS"
            )
            print("------------------------------------------------------------------------------")
            print(f"Batch time: {batch_time:.4f} seconds ({1000 * batch_time / batch_size:.2f} ms/image)")

            # 6. 获取输出数组
            output_array = outputs.get(OUTPUT_TENSOR_NAME)
            if output_array is None:
                raise RuntimeError(f"Expected output tensor name '{OUTPUT_TENSOR_NAME}' not found in outputs.")

            # 7. 处理并保存每张图
            for j in range(batch_size):
                single_out = output_array[j]

                if single_out.ndim == 3:
                    predicted_mask = np.argmax(single_out, axis=0).astype(np.uint8)
                elif single_out.ndim == 2:
                    predicted_mask = single_out.astype(np.uint8)
                else:
                    try:
                        predicted_mask = np.array(single_out).reshape((H, W)).astype(np.uint8)
                    except Exception:
                        predicted_mask = np.array(single_out).squeeze().astype(np.uint8)

                maxv = predicted_mask.max()
                scale_factor = 255 // maxv if maxv > 0 else 1
                scaled_mask = (predicted_mask * scale_factor).astype(np.uint8)

                output_filename = Path(output_dir) / f"{current_batch_files[j].stem}.png"
                cv2.imwrite(str(output_filename), scaled_mask)
                print(f"Saved: {output_filename.name}")

            # 8. 清理
            del buffers_by_index, binding_ptrs, stream

    except Exception as e:
        print("An exception occurred during inference:")
        traceback.print_exc()
    finally:
        # 9. 最终计时结果输出
        if total_images > 0 and total_inference_time > 0:
            avg_time_per_image = total_inference_time / total_images

            print("\n--- 推理性能总结 ---")
            print(f"总处理图像数: {total_images}")
            # print(f"总推理时间 (不含 I/O): {total_inference_time:.4f} 秒")
            print(f"平均每张图像推理时间: {avg_time_per_image:.4f} s ({1000 * avg_time_per_image:.2f} ms)")
            print(f"平均吞吐率: {total_images / total_inference_time:.2f} FPS")

        try:
            del context, engine, runtime
        except Exception:
            pass

    print("\nAll images processed. Done!")
# ---------- 主入口 ----------
if __name__ == "__main__":
    print("Starting TensorRT folder inference...")
    # 检查 engine 文件是否存在
    if not Path(ENGINE_FILE_PATH).exists():
        print(f"Engine file not found: {ENGINE_FILE_PATH}")
        sys.exit(1)
    trt_inference_folder(INPUT_FOLDER, OUTPUT_FOLDER)
