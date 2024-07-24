from hashlib import md5, sha1, sha224, sha256, sha384



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
    
    if hash_exp != None:
        bytes_value = value.encode()  # 编码value
        hash_exp.update(bytes_value)  # 将编码添加到对象
        encoded_value = hash_exp.hexdigest()  # 加密，返回十六进制字符串值
    else:
        encoded_value = None
    
    result = int(encoded_value, 16) % size  # 取余到哈希表大小
    return result

def hash2str(value: str, size: int):
    """value是要哈希的字符串，size是目标字符串长度"""
    hash_value = bin(hash(value))  # 哈希并转换为二进制字符串
    for i, char in enumerate(hash_value):
        if char == 'b':
            binary_str = hash_value[i+1:]  # 去除前缀'0b'
            break
    if len(binary_str) < size:
        binary_str = binary_str.zfill(size)  # 0补齐
    elif len(binary_str) > size:
        binary_str = binary_str[:size]  # 截取左侧
    return binary_str

def int_inbits(value: int, bits: int):
    """value是要判断的整数或小数，bits是要满足的位数"""
    if value == 0:
        return 0
    else:
        if len(bin(value)[2:]) > bits:
            mask = (1 << bits) - 1
            return value & mask
        else:
            return value



if __name__ == "__main__":
    # r = hash2int("md5", "1234564656", 128)
    r = hash2str("serfgwrgs>fwefdfd", 8)
    print(r)

