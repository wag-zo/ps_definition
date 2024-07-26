from hashlib import md5, sha1, sha224, sha256, sha384, sha3_512
import numpy as np
import pandas as pd



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
    encoded_value = hash_exp.hexdigest()  # 加密，返回十六进制字符串值
    
    result = int(encoded_value, 16) % size  # 取余到哈希表大小
    return result

def hash2str(value: str, size: int):
    """value是要哈希的字符串，size是目标字符串长度"""
    # hash_bytes = sha256(value.encode('utf-8')).digest()  # 哈希
    # binary_str = bin(int.from_bytes(hash_bytes, 'big'))[2:]  # 去除前缀'-0b'
    hash_value = bin(abs(hash(value)))  # 哈希并转换为二进制字符串
    binary_str = hash_value[2:]  # 去除前缀'-0b'
    if len(binary_str) < size:
        binary_str = binary_str.zfill(size)  # 0补齐
    elif len(binary_str) > size:
        binary_str = binary_str[:size]  # 截取左侧
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



if __name__ == "__main__":
    # r = hash2int("md5", "1234564656", 128)
    r = hash2str("serfgwrgs>fwefdfd", 32)
    print(r)

    print(str_rshift(r, 6, 32))

