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
        self.test_init()
    
    def sketch_init(self):
        """初始化d行w列sektch"""
        self.sketch = []
        for _ in range(self.d):
            self.sketch.append([[0, 0] for _ in range(self.w)])  # 每个bucket两个字段，分别是ts和lpf

    def test_init(self):
        """验证正确性"""
        self.b2eles, self.b2ele, self.num_b2eles = [], [], []  # 每个epoch中，每个桶对应哪些ID，每个桶对应的当前ID，每个桶对应的ID个数
        for _ in range(self.d):
            self.b2eles.append([set() for _ in range(self.w)])
            self.b2ele.append(["" for _ in range(self.w)])
        self.num_collision = [[0, 0] for _ in range(48)]  # 每轮出现了多少个element，多少个element被覆盖

    def new_packet(self, ID, ts, start, now_start, epoch_len):
        """一个新数据包抵达，判断和更新"""
        flag = False
        Pm = []
        for i in range(self.d):
            pos = hash2int(self.hash_functions[i], ID, self.w)
            if self.sketch[i][pos][0] < now_start:
                flag = True  # 当前包是某个element的第一个包
                # --------------test---------------------
                self.b2eles[i][pos].add(ID)
                self.b2ele[i][pos] = ID
                # --------------test---------------------
                if self.sketch[i][pos][0] < start:
                    empty_epochs = math.ceil((now_start - start) / epoch_len)
                else:
                    empty_epochs = math.ceil((now_start - self.sketch[i][pos][0]) / epoch_len)  # 计算经过了多少个epoch
                new_lpf = self.sketch[i][pos][1] * math.exp(-self.tau * empty_epochs) + 1
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
        return self.c_bits, self.ts_bits, self.d * self.w * (self.ts_bits + self.c_bits) / 1000000
    
    def test_save(self, save_dir):
        """存储桶对应的element个数"""
        df_numeles = pd.DataFrame(np.array(self.num_b2eles))
        df_numeles.to_csv(f"{save_dir}bucket_num_b2eles.csv", header=False)

        df_coll = pd.DataFrame(np.array(self.num_collision))
        df_coll.to_csv(f"{save_dir}epoch_collision_eles.csv", header=False)

    def collision(self, now_epoch):
        """记录每轮多少个element被覆盖"""
        eles_now_epoch = set()  # 统计当前epoch出现了哪些element
        for array in self.b2eles:
            num_array = []
            for bucket in array:
                num_array.append(len(bucket))
                for e in bucket:
                    eles_now_epoch.add(e)
            self.num_b2eles.append(num_array)
        self.num_collision[now_epoch][0] = len(eles_now_epoch)

        for e in eles_now_epoch:
            flag = True  # 初始化为发生冲突
            for i in range(self.d):
                pos = hash2int(self.hash_functions[i], e, self.w)
                if self.b2ele[i][pos] == e:
                    flag = False
                    break
            self.num_collision[now_epoch][1] += 1 if flag else 0



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
        self.test_init()
    
    def sketch_init(self):
        """初始化d行w列sektch"""
        self.sketch =  []
        for _ in range(self.d):
            self.sketch.append([S_Bucket(h=HLL_Counter(self.b, self.alpha, self.mi_bits), x="", l=0) \
                                for _ in range(self.w)])  # 每个bucket三个字段，分别是h，x和l
    
    def test_init(self):
        self.src2es, self.src2ps = {}, {}  # 每一轮每个src对应哪些element，每一轮每个src取出的ps真实值
        self.src2es_all, self.ele2ps_all = {}, {}  # 所有轮每个src对应的element，所有轮每个element对应的pm值
        self.b2srcs, self.b2src = [], []  # 每一轮中，每个桶对应哪些src，每个桶对应的当前src
        self.b2es_hll = []  # 使用hll存储每个桶对应的element个数
        self.src2e_hll = {}  # 存储从hll中取出的element个数
        self.src_r2es = []  # 特定src对应寄存器中存储哪些element
        for _ in range(self.d):
            self.b2srcs.append([set() for _ in range(self.w)])
            self.b2src.append(["" for _ in range(self.w)])
            self.b2es_hll.append([HLL_Counter(self.b, self.alpha, self.mi_bits) for _ in range(self.w)])
            self.src_r2es.append([set() for _ in range(2 ** self.b)])
        self.num_collision = [[0, 0] for _ in range(48)]  # 每轮出现了多少个element，多少个element被覆盖
        self.srcs_covered = {}  # 哪些src在当前epoch中所有element都没能更新
        self.src_r2ep = [[] for _ in range(48)]  # 特定src对应的第0行的寄存器，每轮存了多少element，估计的值是多少
        self.epoch_r2es, self.epoch_r2ps = [], []  # 特定src在特定epoch中，每新到来一个element对应的[真实值，每个寄存器对应的element个数，估计值，每个寄存器存储的值]
        self.epoch_r2e, self.epoch_r2p = 0, 0  # 特定src在特定epoch中，截至当前element到来后的真实值

    def new_packet(self, ID, Pm, epoch_num, now_epoch):
        """一个新数据包抵达，判断和更新"""
        hm = hash2bin(ID, self.L)  # 将element ID编码成长度为L的01字符串
        h1m = hm[:self.b]  # H1(m)取前b位
        h2m = str_rshift(hm[self.b:], math.ceil(math.log2(Pm)), self.L - self.b)  # H2'(m)取后L-b位，并逻辑右移log2(Pm)位
        h2_pre0s = str_pre0(h2m)  # 统计H2'(m)第一个1之前的0的个数
        src = ID.split(">")[0]  # 取出src
        # --------------test---------------------
        if src not in self.src2es.keys():  # 对第一次出现的src，初始化
            self.src2es[src] = [set() for _ in range(epoch_num)]
            self.src2es_all[src] = set()
            # self.srcs_covered[src] = [0 for _ in range(epoch_num)]
        self.src2es[src][now_epoch].add(ID)  # 将element和Pm添加进对应src的列表
        self.src2es_all[src].add(ID)
        # self.srcs_covered[src][now_epoch] += h2_pre0s
        if ID not in self.ele2ps_all.keys():  # 对第一次出现的element，初始化
            self.ele2ps_all[ID]= 0
        self.ele2ps_all[ID]= Pm  # 更新为当前处理包的值
        # --------------test---------------------
        for i in range(self.d):
            pos = hash2int(self.hash_functions[i], src, self.w)
            self.sketch[i][pos].h.update(h1m, h2_pre0s + 1)  # 更新h
            if self.sketch[i][pos].l < h2_pre0s:
                self.sketch[i][pos].l = check_convert(h2_pre0s, self.l_bits)  # 更新l
                self.sketch[i][pos].x = check_convert(src, self.x_bits)  # 更新x
            # --------------test---------------------
            self.b2srcs[i][pos].add(src)
            self.b2src[i][pos] = src
            self.b2es_hll[i][pos].update(h1m, str_pre0(hm[self.b:]) + 1)
            if [i, pos] in [[0, 1639], [1, 950], [2, 2897], [3, 2256], [4, 489]]:  # 特定src寄存器对应的element
                self.src_r2es[i][int(h1m, 2)].add(ID)
            if src == "8e6d3ac4a10746ca" and now_epoch == 3 and i == 4:  # 特定src在特定轮的寄存器情况
                self.epoch_r2e += 1
                self.epoch_r2p += Pm
                self.epoch_r2es.append([ID, self.epoch_r2e] + [len(r) for r in self.src_r2es[i]] + [self.b2es_hll[i][pos].estimate()] + self.b2es_hll[i][pos].register)
                self.epoch_r2ps.append([ID, self.epoch_r2p] + [len(r) for r in self.src_r2es[i]] + [self.sketch[i][pos].h.estimate()] + self.sketch[i][pos].h.register)
            # --------------test---------------------
    
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
                # --------------test---------------------
                # if src == "8e6d3ac4a10746ca":
                #     print(pos_list)
                if src not in self.src2e_hll.keys():
                    self.src2e_hll[src] = [0] * epoch_num
                ele_hll_list = [self.b2es_hll[x][y].estimate() for x, y in pos_list]
                self.src2e_hll[src][now_epoch] = min(ele_hll_list)
                # --------------test---------------------
        # --------------test---------------------
        for src, elements in self.src2es.items():  # 计算真实ps值
            if src not in self.src2ps.keys():
                self.src2ps[src] = [0 for _ in range(epoch_num)]
            for e in elements[now_epoch]:  # 遍历当前轮中出现过的element
                self.src2ps[src][now_epoch] += self.ele2ps_all[e]
        setr = set.union(*self.src_r2es[0])  # src在第0行对应的桶中，所有寄存器中存储的element ID的并集
        self.src_r2ep[now_epoch].extend([len(setr), self.b2es_hll[0][1639].estimate(), sum(self.ele2ps_all[key] for key in setr), self.sketch[0][1639].h.estimate()])
        
        # 清空sketch
        self.b2es_hll, self.src_r2es = [], []
        for _ in range(self.d):
            self.b2es_hll.append([HLL_Counter(self.b, self.alpha, self.mi_bits) for _ in range(self.w)])
            self.src_r2es.append([set() for _ in range(2 ** self.b)])
        self.epoch_r2e = 0
        # --------------test---------------------
        print(f"Epoch{now_epoch} is done")
        self.sketch_init()  # 清空sketch

    def save(self, save_dir):
        """存储所有epoch结束后的结果"""
        df = pd.DataFrame.from_dict(self.spread, orient='index')
        df.to_csv(f"{save_dir}spread_simulation.csv", header=False)

    def get_size(self):
        """返回所有HLL Counter中单个寄存器的最大位数，整个HLL Counter的最大大小，x字段和l字段的最大位数，以及整个sketch的最大大小"""
        _, h_bits = self.sketch[0][0].h.get_size()
        return self.mi_bits, h_bits, self.x_bits, self.l_bits, self.d * self.w * (h_bits + self.x_bits + self.l_bits) / 1000000
    
    def test_save(self, save_dir):
        """存储test过程值"""
        # df_coll = pd.DataFrame(np.array(self.num_collision))
        # df_coll.to_csv(f"{save_dir}epoch_collision_srcs.csv", header=False)

        # num_covered = [0 for _ in range(48)]
        # for src, epochs in self.srcs_covered.items():
        #     for i in range(len(epochs)):
        #         if len(self.src2es[src][i]) != 0 and epochs[i] == 0:  # 当前epoch有element抵达，但h2_pre0s之和为0
        #             num_covered[i] += 1
        # df_covs = pd.DataFrame(np.array(num_covered))
        # df_covs.to_csv(f"{save_dir}simulation_covs.csv", header=False)

        num_src2es = []  # 每个src在每一轮出现的element个数
        for src, epochs in self.src2es.items():
            num_src2es.append([src] + [len(e) for e in epochs])

        df_src2es = pd.DataFrame(np.array(num_src2es))  # 每个src在每轮真实的element个数
        df_src2es.to_csv(f"{save_dir}simulation_src2es.csv", header=False, index=False)

        # df_src2ps = pd.DataFrame.from_dict(self.src2ps, orient='index')  # 每个src在每轮真实的persistent spread
        # df_src2ps.to_csv(f"{save_dir}simulation_src2ps.csv", header=False)

        df_hll2es = pd.DataFrame.from_dict(self.src2e_hll, orient='index')  # 从hll中取出的，每个src在每一轮对应的element个数
        df_hll2es.to_csv(f"{save_dir}simulation_hll2es.csv", header=False)

        # df_e2rs = pd.DataFrame(np.array(self.src_r2ep))  # 特定src对应第0行寄存器，每轮存储的element个数、映射到每个寄存器的element个数、估计值、每个寄存器存储的值
        # df_e2rs.to_csv(f"{save_dir}simulation_r2eps.csv", header=False)

        # df_11e2r = pd.DataFrame(np.array(self.epoch_r2es))  # 特定src特定轮对应第4行寄存器，每个element抵达后存储的element个数、映射到每个寄存器的element个数、估计值、每个寄存器存储的值
        # df_11e2r.to_csv(f"{save_dir}simulation_3r2es.csv", header=False)

        # df_11e2r = pd.DataFrame(np.array(self.epoch_r2ps))  # 特定src特定轮对应第4行寄存器，每个element抵达后存储的pm和、映射到每个寄存器的element个数、估计值、每个寄存器存储的值
        # df_11e2r.to_csv(f"{save_dir}simulation_3r2ps.csv", header=False)
    
    def collision(self, now_epoch):
        """记录每轮多少个src被覆盖"""
        srcs_untilnow_epoch = set()  # 统计截至当前epoch出现了哪些src
        for array in self.srcs:
            num_array = []
            for bucket in array:
                num_array.append(len(bucket))
                for s in bucket:
                    srcs_untilnow_epoch.add(s)
            self.num_srcs.append(num_array)
        self.num_collision[now_epoch][0] = len(srcs_untilnow_epoch)

        srcs_untilnow_bucket = set()  # 统计当前bucket内hll counter存储了哪些src
        # pos_list = [[0, 246], [1, 1337], [2, 1812], [3, 1075], [4, 361]]
        # print([self.srcs_in_bucket[x][y] for x, y in pos_list])
        # print([self.sketch[x][y].x for x, y in pos_list])
        for array in self.srcs_in_bucket:
            for s in array:
                srcs_untilnow_bucket.add(s)
        srcs_untilnow_bucket.remove("")
        self.num_collision[now_epoch][1] = len(srcs_untilnow_epoch) - len(srcs_untilnow_bucket)



if __name__ == "__main__":
    d1, d2 = 5, 5  # P-Sketch和S-Sketch的行数
    w1, w2 = 524288, 4096 # P-Sketch和S-Sketch的列数
    c_bits, ts_bits, mi_bits, x_bits, l_bits = 32, 32, 5, 16, 16  # float, float, int, char, int
    b = 4  # 编码字符串切分H1(m)和H2(m)的位置
    L = 32  # 编码字符串长度
    alpha = 0.673  # HLL Counter估计值的补偿因子，根据s=2**b计算
    tau = 0.1  # MAWI中取0.05，Facebook中取0.1
    
    epoch_len = 300  # 10  # 300  # 1个epoch的时间范围
    start_time = 1475305136  # 0.001  # 1475305136
    end_time = 1475319422  # 100  # 1475319422
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    print("epoch_num = ", epoch_num)
    csv_file_path = "./7.23/test_0.02/easy_packets.csv"  # "./7.23/test/easy_packets.csv"  # "./7.12/data/ca_1.csv"
    save_dir = "./7.23/test_0.02/"  # "./7.23/test/"  # "./7.23/ca_1/"

    p_sketch = P_Sketch(d1, w1, tau, c_bits, ts_bits)
    s_sketch = S_Sketch(d2, w2, b, L, alpha, mi_bits, x_bits, l_bits)

    chunk_size = 20**5  # 可以根据内存调整这个值
    now_start = start_time
    now_epoch = 0
    # --------------test---------------------
    eles_per_epoch = [set() for _ in range(epoch_num)]  # 每轮出现的element个数
    srcs_per_epoch = [set() for _ in range(epoch_num)]  # 每轮出现的src个数
    packets_psketch_processed = [0 for _ in range(epoch_num)]  # 每轮p-sketch处理的包个数
    packets_ssketch_processed = [0 for _ in range(epoch_num)]  # 每轮s-sketch处理的报个数
    # --------------test---------------------
    for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size, on_bad_lines='skip'):
        for _, row in chunk.iterrows():
            ID = f"{row[1]}>{row[2]}"
            ts = row[0]
            if ts - epoch_len >= now_start:  # 当前数据包属于下一epoch
                # --------------test---------------------
                # p_sketch.collision(now_epoch)
                # s_sketch.collision(now_epoch)
                # --------------test---------------------
                s_sketch.report_per_epoch(epoch_num, now_epoch)  # s_sketch收集结果，对当前epoch进行一次report
                now_start += epoch_len  # 更新为下一epoch的开始时间和标号
                now_epoch += 1
            flag, Pm = p_sketch.new_packet(ID, ts, start_time, now_start, epoch_len)
            packets_psketch_processed[now_epoch] +=1
            if flag:
                s_sketch.new_packet(ID, Pm, epoch_num, now_epoch)
                packets_ssketch_processed[now_epoch] +=1
            # --------------test---------------------
            eles_per_epoch[now_epoch].add(ID)
            srcs_per_epoch[now_epoch].add(row[1])
            # --------------test---------------------

    s_sketch.report_per_epoch(epoch_num, now_epoch)  # s_sketch收集结果，对最后一个epoch进行一次report
    s_sketch.save(save_dir)  # 存储结果
    # --------------test---------------------
    # p_sketch.collision(now_epoch)
    # p_sketch.test_save(save_dir)
    # s_sketch.collision(now_epoch)
    s_sketch.test_save(save_dir)
    print("eles_per_epoch = ", [len(e)for e in eles_per_epoch], "\nsrcs_per_epoch = ", [len(s) for s in srcs_per_epoch])
    print("packets_psketch_processed = ", packets_psketch_processed, "\npackets_ssketch_processed = ", packets_ssketch_processed)
    # --------------test---------------------

    c_bits, ts_bits, p_sketch_mbs = p_sketch.get_size()  # 统计P-Sketch大小
    print(f"d = {d1}, w = {w1}, c_bits = {c_bits}, ts_bits = {ts_bits}, p_sketch_mbs = {p_sketch_mbs}")
    r_bits, h_bits, x_bits, l_bits, s_sketch_mbs = s_sketch.get_size()  # 统计S-Sketch大小
    print(f"d = {d2}, w = {w2}, r_bits = {r_bits}, h_bits = {h_bits}, "
          f"x_bits = {x_bits}, l_bits = {l_bits}, s_sketch_mbs = {s_sketch_mbs}")
    
    print("All done")


