import onnx

ONNX_MODEL_PATH = r"F:\unet\denseaspp+cbam\unet-pytorch-main\unet-pytorch-main\unet_exported.onnx" # <--- 替换为您的 ONNX 模型路径

# 加载 ONNX 模型
model = onnx.load(ONNX_MODEL_PATH)

# --- 获取输入名称 ---
print("--- 模型输入名称 ---")
for input_data in model.graph.input:
    print(f"名称: {input_data.name} | 形状: {[dim.dim_value for dim in input_data.type.tensor_type.shape.dim]}")

# --- 获取输出名称 ---
print("\n--- 模型输出名称 ---")
for output_data in model.graph.output:
    print(f"名称: {output_data.name} | 形状: {[dim.dim_value for dim in output_data.type.tensor_type.shape.dim]}")