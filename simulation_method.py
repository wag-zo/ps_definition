import pandas as pd
import math
import sys
import numpy as np
from utils import hash2int, hash2bin, str_pre0, str_rshift, check_convert
hash_algorithms = ["md5", "sha1", "sha224", "sha256", "sha384"]

class P_Sketch():
    def __init__(self, d, w, tau, c_bits, ts_bits) -> None:
        self.d = d
        self.w = w
        self.tau = tau
        self.c_bits = c_bits
        self.ts_bits = ts_bits
        self.hash_functions = hash_algorithms[:d]
        self.sketch_init()
    
    def sketch_init(self):
        """初始化d行w列sektch"""
        self.sketch = []
        for _ in range(self.d):
            self.sketch.append([[0, 0] for _ in range(self.w)])  # 每个bucket两个字段，分别是ts和lpf

    def new_packet(self, ID, ts, now_start, start):
        """一个新数据包抵达，判断和更新"""
        flag = False
        Pm = []
        for i in range(self.d):
            pos = hash2int(self.hash_functions[i], ID, self.w)
            if self.sketch[i][pos][0] < now_start:  # 判断是否属于当前epoch
                flag = True  # 当前包是某个element的第一个包
                empty_time = (ts - start) / epoch_len if self.sketch[i][pos][0] < start \
                    else (ts - self.sketch[i][pos][0]) / epoch_len  # 区分桶第一次更新和其他更新的情况
                new_lpf = self.sketch[i][pos][1] * math.exp(-self.tau * empty_time) + 1
                if not check_convert(new_lpf, self.c_bits) or not check_convert(ts, self.ts_bits):  # 位数检查
                    print("Pm or ts in P-Sketch overflow")
                    sys.exit()
                else: 
                    self.sketch[i][pos][1] = new_lpf  # 更新lpf
                    self.sketch[i][pos][0] = ts  # 更新ts    
            Pm.append(self.sketch[i][pos][1])
        return flag, min(Pm)
    
    def get_size(self):
        """返回c字段和ts字段所需的最大位数，和整个sketch的大小"""
        return self.c_bits, self.ts_bits, self.d * self.w * (self.ts_bits + self.c_bits) / 1024 / 1024



class HLL_Counter():
    def __init__(self, b, alpha, mi_bits) -> None:
        self.s = 2 ** b
        self.b = b
        self.alpha = alpha
        self.mi_bits = mi_bits
        self.register = [0] * self.s  # register初始化为0
    
    def update(self, h1, h2_1):
        """h1是编码字符串的前b位，h2_1是编码字符串后L-b位根据Pm右移后的结果, 其第一个1的位置"""
        idx = int(h1, 2)
        self.register[idx] = max(check_convert(h2_1, self.mi_bits), self.register[idx])  # 取最大值
    
    def estimate(self):
        """取出寄存器中存储的值"""
        sum_r, v = 0, 0
        for mi in self.register:
            if mi == 0:
                v += 1
            sum_r += 2 ** (-mi)
        result = self.alpha * (self.s ** 2) * (sum_r ** (-1))
        if result < (5 / 2) * self.s and v != 0:
            result = self.s * math.log(self.s / v)
        elif result > (2 ** 32 / 30):
            result = -2 ** 32 * math.log(1 - (result / (2 ** 32)))
        return result
    
    def get_size(self):
        """返回单个寄存器所需要的最大位数，整个HLL Counter的大小"""
        return self.mi_bits, self.mi_bits * self.s



class S_Bucket:  
    def __init__(self, h, x, l):  
        self.h = h
        self.x = x    
        self.l = l



class S_Sketch():
    def __init__(self, d, w, b, L, alpha, mi_bits, x_bits, l_bits) -> None:
        self.d = d
        self.w = w
        self.b = b
        self.L = L
        self.alpha = alpha
        self.mi_bits = mi_bits
        self.x_bits = x_bits
        self.l_bits = l_bits
        self.hash_functions = hash_algorithms[:d]
        self.spread = {}
        self.sketch_init()
    
    def sketch_init(self):
        """初始化d行w列sektch"""
        self.sketch =  []
        for _ in range(self.d):
            self.sketch.append([S_Bucket(h=HLL_Counter(self.b, self.alpha, self.mi_bits), x="", l=0) \
                                for _ in range(self.w)])  # 每个bucket三个字段，分别是h，x和l

    def new_packet(self, ID, Pm):
        """一个新数据包抵达，判断和更新"""
        hm = hash2bin(ID, self.L)  # 将element ID编码成长度为L的01字符串
        h1m = hm[:self.b]  # H1(m)取前b位
        h2m = str_rshift(hm[self.b:], math.ceil(math.log2(Pm)), self.L - self.b)  # H2'(m)取后L-b位，并逻辑右移log2(Pm)位
        h2_pre0s = str_pre0(h2m)  # 统计H2'(m)第一个1之前的0的个数
        src = ID.split(">")[0]  # 取出src

        for i in range(self.d):
            pos = hash2int(self.hash_functions[i], src, self.w)
            self.sketch[i][pos].h.update(h1m, h2_pre0s + 1)  # 更新h
            if self.sketch[i][pos].l < h2_pre0s:
                self.sketch[i][pos].l = check_convert(h2_pre0s, self.l_bits)  # 更新l
                self.sketch[i][pos].x = check_convert(src, self.x_bits)  # 更新x
    
    def report_per_epoch(self, epoch_num, now_epoch):
        """每个epoch结束后被调用，将遍历所有桶，取出不重复的src并计算对应的HLL统计值"""
        for i in range(self.d):
            for bucket in self.sketch[i]:
                src = bucket.x  # 对每个bucket存储的src
                if src == "":  # 桶为空时跳过
                    continue
                if src not in self.spread.keys():  # 添加第一次出现的src
                    self.spread[src] = [0] * epoch_num  # 初始化
                pos_list = [[j, hash2int(self.hash_functions[j], src, self.w)] for j in range(self.d)]  # 哈希的位置[行, 列]
                hll_list = [self.sketch[x][y].h.estimate() for x, y in pos_list]  # 取出寄存器中的值
                self.spread[src][now_epoch] = min(hll_list)
        print(f"Epoch{now_epoch} is done")
        self.sketch_init()  # 清空sketch
    
    def save(self, save_dir):
        """存储所有epoch结束后的结果"""
        df = pd.DataFrame.from_dict(self.spread, orient='index')
        df.to_csv(f"{save_dir}spread_simulation.csv", header=False)

    def get_size(self):
        """返回所有HLL Counter中单个寄存器的最大位数，整个HLL Counter的最大大小，x字段和l字段的最大位数，以及整个sketch的最大大小"""
        _, h_bits = self.sketch[0][0].h.get_size()
        return self.mi_bits, h_bits, self.x_bits, self.l_bits, self.d * self.w * (h_bits + self.x_bits + self.l_bits) / 1024 / 1024



if __name__ == "__main__":
    d1, d2 = 3, 4  # P-Sketch和S-Sketch的行数
    mem1, mem2 = 1 * 1024, 0.25 * 1024  # P-Sketch和S-Sketch的内存容量 / KB
    c_bits, ts_bits, mi_bits, x_bits, l_bits = 32, 32, 5, 16, 16  # float, float, int, char, int
    b = 4  # 编码字符串切分H1(m)和H2(m)的位置
    L = 32  # 编码字符串长度
    w1, w2 = math.floor(mem1 * 1024 * 8 / (c_bits + ts_bits) / d1),\
             math.floor(mem2 * 1024 * 8 / (2 ** b * mi_bits + x_bits + l_bits) / d2) # 根据内存容量确定P-Sketch和S-Sketch的列数
    alpha = 0.673  # HLL Counter估计值的补偿因子，根据s=2**b计算
    tau = 0.1  # fb: 0.1 MAWI: 0.05
    
    epoch_len = 300  # fb: 300 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1475305136  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1475319422  # fb: 1475319422 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    print("epoch_num = ", epoch_num)
    csv_file_path = "./7.12/data/ca_1.csv"  # fb: "./7.12/data/ca_1.csv" MAWI: "./7.12/data/202304112345_packets.csv"
    save_dir = "./8.22/FB/"  # fb: ./8.22/FB/ MAWI: ./8.22/MAWI/

    p_sketch = P_Sketch(d1, w1, tau, c_bits, ts_bits)
    s_sketch = S_Sketch(d2, w2, b, L, alpha, mi_bits, x_bits, l_bits)

    chunk_size = 20**5  # 可以根据内存调整这个值
    now_start = start_time
    now_epoch = 0

    eles_per_epoch = [set() for _ in range(epoch_num)]  # 每个epoch出现的所有element
    srcs_per_epoch = [set() for _ in range(epoch_num)]  # 每个epoch出现的所有src
    packets_psketch_processed = [0 for _ in range(epoch_num)]  # 每个epoch p_sketch处理的包的数量
    packets_ssketch_processed = [0 for _ in range(epoch_num)]  # 每个epoch s_sketch处理的包的数量 = element数量

    for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size, on_bad_lines='skip'):
        for _, row in chunk.iterrows():
            ID = f"{row[1]}>{row[2]}"
            ts = row[0]
            if ts - epoch_len >= now_start:  # 当前数据包属于下一epoch
                s_sketch.report_per_epoch(epoch_num, now_epoch)  # s_sketch收集结果，对当前epoch进行一次report
                now_start += epoch_len  # 更新为下一epoch的开始时间和标号
                now_epoch += 1

            eles_per_epoch[now_epoch].add(ID)
            srcs_per_epoch[now_epoch].add(row[1])
            flag, Pm = p_sketch.new_packet(ID, ts, now_start, start_time)
            packets_psketch_processed[now_epoch] +=1
            if flag:
                s_sketch.new_packet(ID, Pm)
                packets_ssketch_processed[now_epoch] +=1

    s_sketch.report_per_epoch(epoch_num, now_epoch)  # s_sketch收集结果，对最后一个epoch进行一次report
    s_sketch.save(save_dir)  # 存储结果

    file = open(f"{save_dir}output.txt", "w")
    sys.stdout = file  # 输出直接存入txt文件

    print("eles_per_epoch = ", [len(e)for e in eles_per_epoch], "\nsrcs_per_epoch = ", [len(s) for s in srcs_per_epoch])
    print("packets_psketch_processed = ", packets_psketch_processed, "\npackets_ssketch_processed = ", packets_ssketch_processed)
    c_bits, ts_bits, p_sketch_mbs = p_sketch.get_size()  # 统计P-Sketch大小
    print(f"d = {d1}, w = {w1}, c_bits = {c_bits}, ts_bits = {ts_bits}, p_sketch_mbs = {p_sketch_mbs}")
    r_bits, h_bits, x_bits, l_bits, s_sketch_mbs = s_sketch.get_size()  # 统计S-Sketch大小
    print(f"d = {d2}, w = {w2}, r_bits = {r_bits}, h_bits = {h_bits}, "
          f"x_bits = {x_bits}, l_bits = {l_bits}, s_sketch_mbs = {s_sketch_mbs}")
    
    print("All done")
    file.close()


