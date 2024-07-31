from hashlib import md5, sha1, sha224, sha256, sha384
import math
import numpy as np
import pandas as pd
import os



def hash2int(type: str, value: str, size: int):
    """type是哈希函数类型，value是要哈希的字符串，size是哈希表大小"""
    hash_exp = None  # 构建对象
    if type == "md5":
        hash_exp = md5()
    elif type == "sha1":
        hash_exp = sha1()
    elif type == "sha224":
        hash_exp = sha224()
    elif type == "sha256":
        hash_exp = sha256()
    elif type == "sha384":
        hash_exp = sha384()
    
    bytes_value = value.encode()  # 编码value
    hash_exp.update(bytes_value)  # 将编码添加到对象
    encoded_value = hash_exp.hexdigest()  # 加密，返回十六进制字节流值
    encoded_value = encoded_value[-math.ceil(len(bin(size)[2:]) / 4):]  # 截取到需要值
    
    result = int(encoded_value, 16) % size  # 转换为哈希表索引
    return result

def hash2bin(value: str, size: int):
    """value是要哈希的字符串，size是目标字符串长度"""
    hash_exp = md5()
    bytes_value = value.encode()  # 编码value
    hash_exp.update(bytes_value)  # 将编码添加到对象
    encoded_value = hash_exp.hexdigest()  # 加密，返回十六进制字节流值
    encoded_value = encoded_value[-math.ceil(size / 4):]  # 截取到需要值
    binary_str = bin(int(encoded_value, 16))[2:]  # 转为二进制

    # hash_int = abs(hash(value))
    # binary_str = bin(hash_int)[2:]

    if len(binary_str) > size:
        binary_str = binary_str[-size:]  # 截取右侧
    elif len(binary_str) < size:
        binary_str = binary_str.zfill(size)  # 0补齐
    return binary_str

def str_pre0(value: str):
    """统计字符串中第一个1之前的0的个数"""
    for i, v in enumerate(value):  
        if v == '1':  
            return i
    return 0

def str_rshift(value: str, shift_bits: int, total_bits: int):
    """value逻辑右移shift_bits"""
    new_value = "0" * shift_bits + value
    return new_value[:total_bits]

def check_convert(value, bits: int):
    """value是要判断的整数或小数或字符串，bits是要满足的位数"""
    if isinstance(value, int):  # 对整数截取
        if value == 0:
            return 0
        elif len(bin(value)[2:]) <= bits:
            return value
        else:
            mask = (1 << bits) - 1
            return value & mask
    elif isinstance(value, float):  # 对浮点数检查
        value_32 = np.float32(value)
        flag = not(np.isinf(value_32) or np.isnan(value_32))
        return flag
    else:  # 对字符串截取
        flag = len(value) <= bits
        return value if flag else value[:bits]

def align(left_path, right_path, save_path):
    """将所有结果与左侧结果对齐，非匹配行填充0"""
    df_left = pd.read_csv(left_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(48)])
    df_right = pd.read_csv(right_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(48)])

    df_merged = pd.merge(df_left, df_right, on='src', how='left', suffixes=('_x', ''))  # 以左侧的src列连接，并为左侧中的Epoch列添加_x后缀
    df_merged = df_merged[['src'] + [f'Epoch{i}' for i in range(48)]]  # 取出左侧的src和右侧的Epoh列
    fill_cols = df_merged.columns.drop('src')  # 找到非匹配行
    df_merged[fill_cols] = df_merged[fill_cols].fillna(0)  # 填充为0

    df_merged.to_csv(save_path, header=False, index=False)

def topk(path, k_percent, save_path):
    """根据所有epoch值的和，找到前k_percent%的行并存储"""
    df = pd.read_csv(path, header=None, names=['src'] + [f'Epoch{i}' for i in range(48)])
    df['sum'] = df.iloc[:, 1:].astype(int).sum(axis=1)  # 对第一列后的元素求和，计算sum
    df_sorted = df.sort_values(by='sum', ascending=False)  # 按照sum降序

    top_k_df = df_sorted.head(int(len(df_sorted) * k_percent))  # 取前k_percent行
    top_k_df = top_k_df.drop('sum', axis=1)  # 删除sum列
    top_k_df.to_csv(save_path, index=False)

def suffix(path, suf):
    """为文件添加后缀"""
    base, ext = os.path.splitext(os.path.basename(path))
    new_base = base + suf  # 添加后缀
    dir = os.path.dirname(path)
    new_path = os.path.normpath(os.path.join(dir, new_base + ext))  # 组合目录、文件名、类型
 
    return new_path



if __name__ == "__main__":
    save_dir = "./7.23/test/"
    gt_path = "./7.23/test/spread_groundtruth.csv"
    sim_path = "./7.23/test/spread_simulation.csv"
    align(gt_path, sim_path, suffix(sim_path, "_aligned"))  # 模拟结果与ground truth对齐，方便查看
    print("sim_aligned is saved in", suffix(sim_path, "_aligned"))


