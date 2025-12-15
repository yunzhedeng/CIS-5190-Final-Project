# -*- coding: utf-8 -*-
import torch
import os

# --- 配置 ---
# 这里换成你的 pt 文件路径，如果就在当前目录下则不用改
pt_file_path = 'model.pt'
# ------------

print("="*50)
print(f"正在开始检查文件: {pt_file_path}")
print("="*50)

if not os.path.exists(pt_file_path):
    print(f"[错误]: 在当前目录下找不到文件: {pt_file_path}")
    print("请确保你是在包含 model.pt 的目录下运行此脚本。")
    exit()

try:
    # 1. 加载文件 (强制映射到 CPU，避免缺少 GPU 报错)
    print("尝试加载文件 (map_location='cpu')...")
    checkpoint = torch.load(pt_file_path, map_location=torch.device('cpu'))
    print("[成功]: 文件加载完毕。\n")

    # 2. 分析顶层结构
    print(f"数据顶层类型: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        top_keys = list(checkpoint.keys())
        print(f"顶层包含的键 (Keys): {top_keys}\n")

        # 3. 寻找真正的模型权重字典
        state_dict_to_inspect = None
        
        # 情况 A: 嵌套结构 (常见的保存方式，包含 epoch, optimizer 等)
        if 'model_state_dict' in checkpoint:
            print("-> [结构识别]: 这是一个【嵌套字典】。权重保存在 'model_state_dict' 键下。")
            state_dict_to_inspect = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            print("-> [结构识别]: 这是一个【嵌套字典】。权重保存在 'state_dict' 键下。")
            state_dict_to_inspect = checkpoint['state_dict']
        # 情况 B: 扁平结构 (直接就是权重字典)
        elif any(k.startswith('roberta.') or k.startswith('module.') or k.startswith('classifier.') for k in checkpoint.keys()):
             print("-> [结构识别]: 这是一个【扁平权重字典】。文件直接包含了模型参数。")
             state_dict_to_inspect = checkpoint
        else:
            print("-> [结构识别]: 无法自动识别权重所在的键值。请检查上面的顶层键列表。")

        # 4. 打印关键信息
        if state_dict_to_inspect and isinstance(state_dict_to_inspect, dict):
            print("-" * 30)
            print(f"权重字典中共包含 {len(state_dict_to_inspect)} 个参数张量。")
            
            # 检查 module 前缀（这是最可能导致问题的元凶）
            example_keys = list(state_dict_to_inspect.keys())[:5]
            has_module_prefix = any(k.startswith('module.') for k in example_keys)

            print("\n【关键信息诊断】:")
            if has_module_prefix:
                 print("🔴 发现问题: 键名中包含 'module.' 前缀！")
                 print("   (原因: 模型通常是在多 GPU (DataParallel) 环境下训练保存的。)")
                 print("   (后果: eval 脚本如果是单卡运行，会无法匹配这些参数。)")
            elif 'model_state_dict' in checkpoint and isinstance(checkpoint, dict) and len(top_keys) > 1:
                 print("🟠 发现潜在问题: 这是一个嵌套字典。")
                 print("   (如果 eval 脚本没有编写解包逻辑，会加载失败。)")
            else:
                 print("🟢 结构看起来相对正常 (扁平且无 module 前缀)。")

            print("\n前 5 个参数键名示例 (Sample Keys):")
            for i, key in enumerate(example_keys):
                print(f"  {i+1}. {key}")
            print("-" * 30)

    else:
        print("警告: Checkpoint 不是字典类型，这很不寻常。")

except Exception as e:
    print(f"\n[严重错误]: 加载或分析文件时发生异常:\n{e}")

print("\n检查完成。")