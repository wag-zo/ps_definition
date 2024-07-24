import pandas as pd
import math
from utils import hash2int, hash2str, int_inbits
hash_algorithms = ["md5", "sha1", "sha224", "sha256", "sha384"]

class P_Sketch():
    def __init__(self, d, w, tau, epoch_length) -> None:
        self.d = d
        self.w = w
        self.tau = tau
        self.l = epoch_length
        self.hash_functions = hash_algorithms[:d]
        self.sketch = []
        self.sketch_init()
    
    def sketch_init(self):
        """初始化d行w列sektch"""
        for _ in range(self.d):
            self.sketch.append([[0, 0] for _ in range(self.w)])  # 每个bucket两个字段，分别是ts和lpf
    
    def new_packet(self, ID, ts, now_epoch_start):
        """一个新数据包抵达，判断和更新"""
        flag = False
        Pm = []
        for i in range(self.d):
            pos = hash2int(self.hash_functions[i], ID, self.w)
            if self.sketch[i][pos][0] < now_epoch_start:
                empty_epochs = math.ceil((now_epoch_start - self.sketch[i][pos][0]) / self.l)  # 计算经过了多少个epoch
                Pm.append(self.sketch[i][pos][1] * math.exp(-self.tau * empty_epochs) + 1)
                self.sketch[i][pos][1] = Pm[-1]  # 更新lpf
                self.sketch[i][pos][0] = ts  # 更新ts
                flag = True
        
        min_Pm = min(Pm) if len(Pm) != 0 else 0  # 若当前数据包需要发送到S-Sketch，则统计对应Pm
        return flag, min_Pm
    
    def get_size(self):
        """统计需要的大小，返回c字段和ts字段所需的最大位数，和整个sketch的大小"""
        c_bits, ts_bits = 0, 0
        for array in self.sketch:
            for bucket in array:
                new_ts_bits, new_c_bits = len(bin(bucket[0])[2:]), len(bin(bucket[1])[2:])
                ts_bits = new_ts_bits if new_ts_bits > ts_bits else ts_bits
                c_bits = new_c_bits if new_c_bits > c_bits else c_bits
        return c_bits, ts_bits, self.d * self.w * (ts_bits + c_bits) / 1000



class HLL_Counter():
    def __init__(self, b, alpha) -> None:
        self.s = 2 ** b
        self.b = b
        self.alpha = alpha
        self.register = [0] * self.s  # regidter初始化为0
    
    def update(self, h1, h2_1):
        """h1是编码字符串的前b位，h2_0是编码字符串后L-b位根据Pm右移后的结果, 其第一个1的位置"""
        idx = int(h1, 2) % self.s  # ???---------H1(m)可能超出s-------???
        self.register[idx] = h2_1 if h2_1 > self.register[idx] else self.register[idx]  # 取最大值
    
    def find(self):
        """取出寄存器中存储的值"""
        sum_r = 0
        for mi in self.register:
            sum_r += 2 ** (-mi)
        result = self.alpha * (self.s ** 2) / sum_r if sum_r != 0 else 0
        return result
    
    def get_size(self):
        """统计需要的大小，返回单个寄存器所需要的最大位数，整个HLL Counter的大小"""
        max_bits = 0
        for r in self.register:
            bits = len(bin(r)[2:]) if r != 0 else 0  # 计算单个寄存器需要的位数
            if bits > max_bits:
                max_bits = bits  # 统计最大值
        return max_bits, max_bits * self.s



class S_Bucket:  
    def __init__(self, h, x, l):  
        self.h = h
        self.x = x    
        self.l = l



class S_Sketch():
    def __init__(self, d, w, b, L, alpha) -> None:
        self.d = d
        self.w = w
        self.b = b
        self.L = L
        self.alpha = alpha
        self.hash_functions = hash_algorithms[:d]
        self.sketch = []
        self.sketch_init()
    
    def sketch_init(self):
        """初始化d行w列sektch"""
        for _ in range(self.d):
            self.sketch.append([S_Bucket(h=HLL_Counter(self.b, self.alpha), x="", l=0) for _ in range(self.w)])  # 每个bucket三个字段，分别是h，x和l
    
    def new_packet(self, ID, Pm):
        """一个新数据包抵达，判断和更新"""
        hm = hash2str(ID, self.L)  # 将element ID编码成长度为L的01字符串
        h1m = hm[:self.b]  # H1(m)取前b位
        h2m = '0' * math.ceil(math.log2(Pm)) + hm[self.L - self.b:]  # H2'(m)取后L-b位，并在左边添log2(Pm)个零
        h2_0 = 0  # 返回第一个1之前的0的个数（即索引）
        for i, char in enumerate(h2m):  
            if char == '1':  
                h2_0 = i
        
        flowID = ID.split(">")[0]  # 取出src
        for i in range(self.d):
            pos = hash2int(self.hash_functions[i], flowID, self.w)
            self.sketch[i][pos].h.update(h1m, h2_0 + 1)  # 更新c
            if self.sketch[i][pos].l < h2_0:
                self.sketch[i][pos].l = h2_0  # 更新l
                self.sketch[i][pos].x = flowID  # 更新x
    
    def report(self, save_dir, epoch):
        """每个epoch结束后被调用，将遍历所有桶，取出不重复的flowID并计算对应的HLL统计值"""
        S_dict = {}
        for i in range(self.d):
            for bucket in self.sketch[i]:
                flowID = bucket.x  # 对每个bucket存储的flowID
                if flowID != "" and flowID not in S_dict.keys():
                    pos_list = [[j, hash2int(self.hash_functions[j], flowID, self.w)] for j in range(self.d)]  # 哈希的位置[行, 列]
                    # print(pos_list)
                    hll_list = []
                    for d, w in pos_list:
                        if self.sketch[d][w].x == flowID:  # 判断当前bucket存储的是否是要找的flowID
                            hll_list.append(self.sketch[d][w].h.find())  # 取出寄存器中的值
                    # print(hll_list)
                    S_dict[flowID] = min(hll_list)
        
        # print(S_dict)
        df = pd.DataFrame.from_dict(S_dict, orient='index')
        df.to_csv(f"{save_dir}ps_after{epoch}.csv", header=False)
    
    def get_size(self):
        """统计需要的大小，返回所有HLL Counter中单个寄存器的最大位数，整个HLL Counter的最大大小，x字段和l字段的最大位数，以及整个sketch的最大大小"""
        r_bits, h_bits, x_bits, l_bits = 0, 0, 0, 0
        for array in self.sketch:
            for bucket in array:
                new_r_bits, new_h_bits = bucket.h.get_size()
                r_bits = new_r_bits if new_r_bits > r_bits else r_bits
                h_bits = new_h_bits if new_h_bits > h_bits else h_bits
                new_x_bits, new_l_bits = len(bin(bucket.x)[2:]), len(bin(bucket.l)[2:])
                x_bits = new_x_bits if new_x_bits > x_bits else x_bits
                l_bits = new_l_bits if new_l_bits > l_bits else l_bits
        return r_bits, h_bits, x_bits, l_bits, self.d * self.w * (h_bits + x_bits + l_bits) / 1000



if __name__ == "__main__":
    d1, d2 = 4, 4  # P-Sketch和S-Sketch的行数
    w1, w2 = 512, 512  # P-Sketch和S-Sketch的列数
    b = 4  # 编码字符串切分H1(m)和H2(m)的位置
    L = 32  # 编码字符串长度
    alpha = 0.673  # HLL Counter估计值的补偿因子，根据s=2**b计算
    tau = 0.1  # MAWI中取0.05，Facebook中取0.1

    epoch_len = 300  # 1个epoch的时间范围
    start_time = 1475305136
    end_time = 1475319422
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    print("epoch_num = ", epoch_num)
    csv_file_path = "./7.12/data/ca_1.csv"
    save_dir = "./7.23/ca_1/"

    p_sketch = P_Sketch(d1, w1, tau, epoch_len)
    s_sketch = S_Sketch(d2, w2, b, L, alpha)

    chunk_size = 20**5  # 可以根据内存调整这个值
    now_start = start_time
    for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size, on_bad_lines='skip'):
        for _, row in chunk.iterrows():
            ID = f"{row[1]}>{row[2]}"
            ts = row[0]
            # print(ts, ID)
            if ts - epoch_len > now_start:  # 当前数据包属于下一epoch
                epoch = int((now_start - start_time) / epoch_len)
                s_sketch.report(save_dir, epoch)  # s_sketch收集结果，进行一次report
                now_start += epoch_len  # 更新为下一epoch的开始时间
                print(f"Epoch{epoch} is done")
            flag, Pm = p_sketch.new_packet(ID, ts, now_start)
            if flag:
                s_sketch.new_packet(ID, Pm)
    
    epoch = int((now_start - start_time) / epoch_len)  # 最后一个epoch
    s_sketch.report(save_dir, epoch)  # s_sketch收集结果，进行一次report
    now_start += epoch_len  # 更新为下一epoch的开始时间
    print(f"Epoch{epoch} is done")

    c_bits, ts_bits, p_sketch_kbs = p_sketch.get_size()
    print(f"c_bits = {c_bits}, ts_bits = {ts_bits}, p_sketch_kbs = {p_sketch_kbs}")
    r_bits, h_bits, x_bits, l_bits, s_sketch_kbs = s_sketch.get_size()
    print(f"r_bits = {r_bits}, h_bits = {h_bits}, x_bits = {x_bits}, l_bits = {l_bits}, s_sketch_kbs = s_sketch_bits")
    
    print("All done")


