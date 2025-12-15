import torch
from collections import OrderedDict

# 1. 备份原文件 (安全第一)
import shutil
shutil.copy('model.pt', 'model_backup.pt')
print("已将原 model.pt 备份为 model_backup.pt")

# 2. 加载权重
print("正在加载 model.pt ...")
state_dict = torch.load('model.pt', map_location='cpu')

# 3. 创建带 'module.' 前缀的新字典
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = "module." + k  # 强行加前缀
    new_state_dict[new_key] = v

print(f"转换完成！原键数量: {len(state_dict)}, 新键数量: {len(new_state_dict)}")
print(f"示例新键名: {list(new_state_dict.keys())[0]}")

# 4. 覆盖保存
torch.save(new_state_dict, 'model.pt')
print("已覆盖保存新的 model.pt (带 module. 前缀)")