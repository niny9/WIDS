# By zhangting 2025/04/15

# connectome_analysis.py
import numpy as np

def connect_analysis(fmri, normalized=1):
    # 使用 .iloc 按位置选取偶数列
    connect_contra = fmri.iloc[:, 0::2]   # 等价于MATLAB的1:2:end[1,7](@ref)

    if normalized == 1:
        isconnect_contra = (connect_contra > np.finfo(float).eps).astype(float)
        count_connect_contra = isconnect_contra.sum(axis=1)
    else:
        count_connect_contra = connect_contra.sum(axis=1)

    # 同侧连接处理（偶数列+1）
    connect_within = fmri.iloc[:, 1::2]  # 等价于MATLAB的2:2:end[1,7](@ref)
    if normalized == 1:
        isconnect_within = (connect_within > np.finfo(float).eps).astype(float)
        count_connect_within = isconnect_within.sum(axis=1)
    else:
        count_connect_within = connect_within.sum(axis=1)

    return count_connect_contra, count_connect_within