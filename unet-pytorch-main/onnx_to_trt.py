# import tensorrt as trt
#
#
# def build_engine_from_onnx(onnx_file_path, engine_file_path,
#                            fp16=True,
#                            dynamic=False,
#                            min_shape=(1, 3, 512, 512),
#                            opt_shape=(1, 3, 512, 512),
#                            max_shape=(1, 3, 1024, 1024)):
#     logger = trt.Logger(trt.Logger.INFO)
#     builder = trt.Builder(logger)
#     network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # 支持显式 batch
#     network = builder.create_network(network_flags)
#     parser = trt.OnnxParser(network, logger)
#
#     print(f"Loading ONNX file: {onnx_file_path}")
#     with open(onnx_file_path, "rb") as f:
#         if not parser.parse(f.read()):
#             print("Failed to parse the ONNX file.")
#             for i in range(parser.num_errors):
#                 print(parser.get_error(i))
#             return None
#
#     config = builder.create_builder_config()
#     config.max_workspace_size = 4 * 1024 * 1024 * 1024  # 4GB
#
#     # --------------------
#     # FP16 支持
#     # --------------------
#     if fp16 and builder.platform_has_fast_fp16:
#         print("Using FP16 mode...")
#         config.set_flag(trt.BuilderFlag.FP16)
#
#     # --------------------
#     # 动态尺寸支持
#     # --------------------
#     if dynamic:
#         print("Using dynamic shapes...")
#         profile = builder.create_optimization_profile()
#         input_tensor = network.get_input(0).name
#         profile.set_shape(input_tensor, min_shape, opt_shape, max_shape)
#         config.add_optimization_profile(profile)
#
#     print("Building TensorRT engine, may take a few minutes...")
#     engine = builder.build_engine(network, config)
#
#     if engine is None:
#         print("Engine build failed!")
#         return None
#
#     # 保存 engine
#     print(f"Saving engine to {engine_file_path}")
#     with open(engine_file_path, "wb") as f:
#         f.write(engine.serialize())
#
#     print("Done!")
#     return engine
#
#
# # --------------------
# # 调用示例
# # --------------------
# onnx_path = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\unet_exported.onnx"
# engine_path = "unet_fp16.engine"
#
# build_engine_from_onnx(
#     onnx_file_path=onnx_path,
#     engine_file_path=engine_path,
#     fp16=True,
#     dynamic=False  # 如果要用动态 batch/尺寸改为 True
# )


import tensorrt as trt
import os

# ----------------------------------------------------------
# 配置参数
# ----------------------------------------------------------
ONNX_PATH = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\unet_exported.onnx"
ENGINE_PATH = "unet_dynamic_batch_fp32.engine"
MAX_WORKSPACE_SIZE_GB = 4  # 工作空间大小 (4GB)

# 动态 Batch Size 范围 (H和W固定为512)
BATCH_MIN = 2
BATCH_OPT = 16  # TensorRT 将以最优 Batch Size 进行深度优化
BATCH_MAX = 16

MIN_SHAPE = (BATCH_MIN, 3, 512, 512)
OPT_SHAPE = (BATCH_OPT, 3, 512, 512)
MAX_SHAPE = (BATCH_MAX, 3, 512, 512)


def build_engine_from_onnx(onnx_file_path, engine_file_path):
    # 1. 初始化 Logger, Builder, Network
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    # 启用显式 Batch size，这是动态形状所必需的
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"Loading ONNX file: {onnx_file_path}")

    # 2. 解析 ONNX 文件
    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX file not found at {onnx_file_path}")
        return None

    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse the ONNX file.")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    # 3. 配置 Builder Config
    config = builder.create_builder_config()
    # 使用新方法设置工作空间大小
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, MAX_WORKSPACE_SIZE_GB * 1024 * 1024 * 1024)

    # # 4. 配置 FP16 模式
    # if builder.platform_has_fast_fp16:
    #     print("Using FP16 mode...")
    #     config.set_flag(trt.BuilderFlag.FP16)
    # 4. 配置 FP32 模式
    print("Using FP32 mode...")


    # 5. 配置动态 Batch Size (核心修改)
    print(f"Using dynamic batch shapes: Min={BATCH_MIN}, Opt={BATCH_OPT}, Max={BATCH_MAX}")

    profile = builder.create_optimization_profile()
    # 假设输入张量名称是 'input' (PyTorch 默认导出名称)
    input_tensor_name = network.get_input(0).name

    # 设置 Batch Size 动态范围，H和W保持固定512
    profile.set_shape(
        input_tensor_name,
        min=MIN_SHAPE,
        opt=OPT_SHAPE,
        max=MAX_SHAPE
    )
    config.add_optimization_profile(profile)

    # 6. 构建 Engine
    print("Building TensorRT engine, may take a few minutes...")
    # 使用 build_serialized_network 替代 build_engine
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Engine build failed!")
        return None

    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized_engine)

    # 7. 保存 engine
    print(f"Saving engine to {engine_file_path}")
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

    print("Done!")
    return engine


# --------------------
# 运行构建函数
# --------------------
build_engine_from_onnx(
    onnx_file_path=ONNX_PATH,
    engine_file_path=ENGINE_PATH
)