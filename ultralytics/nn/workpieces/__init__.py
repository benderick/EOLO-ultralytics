"""
    将各个功能包的模块收集到列表中。
"""

# helper function: 将包内的具有style的模块收集到列表中
def collect_to_list(m, style='all'):
    lst = []
    # 检查模块是否有 __all__ 属性
    if hasattr(m, '__all__'):
        # 遍历 __all__ 中定义的名称
        for name in m.__all__:
            try:
                # 获取名称对应的对象并添加到列表中
                it = getattr(m, name)
                if style == 'all':
                    lst.append(it)
                elif hasattr(it, 'style') and it.style == style:
                    lst.append(it)
                else:
                    continue
            except AttributeError:
                # 如果 __all__ 中的名称在模块中实际不存在，则跳过
                print(f"警告：{m}.__all__ 中定义的 '{name}' 在 {m} 模块中未找到。")
    else:
        print(f"警告：{m} 模块未定义 __all__，将忽略此模块。")
    return lst

# 检测头
# import head
from .head import *

# 注意力
# import attention
from .attention import *

from ._2025 import *

from .convolution import *

m_list = [attention, convolution,_2025]

# style_list = ["i","ij", "ijk", "nij", "l"]
"""
i: 模块形参只有单个通道参数。输入通道与输出通道一致。实参不包括通道参数。
ij: 模块形参包含输入和输出通道。实参包括输出通道。
ijk: 模块形参包含输入和输出通道以及中间通道。实参包括中间通道和输出通道。
nij: 模块形参包含输入和输出通道。实参包括输出通道。n表示重复次数，用于C3k2，C2f等。
l: 模块形参包含输入和输出通道，输入通道是列表（汇集多层输入）。实参包括输出通道。形参若不给出输出通道，会使用输入通道列表的末项。
"""

detection_head_list = collect_to_list(head)
ijk_list = []
ij_list = []
i_list = []
nij_list = []
l_list = []
other_list = []

for m in m_list:
    if hasattr(m, '__all__'):
        # 遍历 __all__ 中定义的名称
        for name in m.__all__:
            try:
                # 获取名称对应的对象并添加到列表中
                it = getattr(m, name)
                if hasattr(it, 'style'):
                    if it.style == "ijk":
                        ijk_list.append(it)
                    elif it.style == "ij":
                        ij_list.append(it)
                    elif it.style == "i":
                        i_list.append(it)
                    elif it.style == "nij":
                        nij_list.append(it)
                    elif it.style == "l":
                        l_list.append(it)
                    else:
                        other_list.append(it)
                else:
                    continue
            except AttributeError:
                # 如果 __all__ 中的名称在模块中实际不存在，则跳过
                print(f"警告：{m}.__all__ 中定义的 '{name}' 在 {m} 模块中未找到。")
    else:
        print(f"警告：{m} 模块未定义 __all__，将忽略此模块。")

# print("收集到的模块：")
# print("检测头模块：", detection_head_list)
# print("ijk模块：", ijk_list)
# print("ij模块：", ij_list)
# print("i模块：", i_list)
# print("nij模块：", nij_list)
# print("l模块：", l_list)
# print("其他模块：", other_list)
