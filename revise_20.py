import pandas as pd
from collections import Counter
import math
import csv
import time
from utils import hash2int



class appearance():
    def __init__(self, buckets, b_id, b_flag = False) -> None:
        self.lifetime = {}
        self.e_time = Counter()
        self.buckets = buckets
        self.b_id = b_id
        self.b_flag = b_flag
    
    def enumerate(self, epoch_num, pre_path, save_dir) -> None:
        s_rows = []
        for epoch_now in range(epoch_num):
            file = open(pre_path)
            reader = csv.reader(file)
            f_e = {}
            for row in reader:
                e, value = row[0], int(row[epoch_now + 1])  # 计算element_id和对应epoch
                f = e.split(">")[0]
                if self.b_flag and hash2int("md5", f, self.buckets) != self.b_id:  # 跳过不属于当前桶的flow
                    continue
                self.e_time[e] += value  # 更新element出现次数，初始默认为0
                if f not in self.lifetime:
                    self.lifetime[f] = epoch_now  # 只记录首次出现的epoch
                if f not in f_e:
                    f_e[f] = [e]
                else:
                    f_e[f].append(e)
            file.close()

            for f, e_list in f_e.items():
                spread = len(e_list)
                appearance = sum([self.e_time[e] for e in e_list]) / (epoch_now - self.lifetime[f] + 1) / spread
                s_rows.append([epoch_now, f, appearance, spread])  # 更新到列表
            print(f"Epoch{epoch_now}: appearance is done")

        s_df = pd.DataFrame(s_rows, columns=['Epoch', 'flow_id', 'appearance', 'spread'])
        save_path = f"{save_dir}appe_spread_{self.b_id}_{self.buckets}.csv" if self.b_flag else f"{save_dir}appe_spread.csv"
        # with open(save_path, 'w', newline='') as file:  # 用csv写入
        #     writer = csv.writer(file)
        #     for row in s_rows:
        #         writer.writerow(row)
        s_df.to_csv(save_path, header=False, index=False)



if __name__ == "__main__":
    buckets = 6  # 分桶数
    b_id = 0  # 当前计算哪个桶
    
    epoch_len = 60  # fb: 60 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1475305136  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1475392025  # fb: 1475392025 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    print("epoch_num = ", epoch_num)
    csv_file_path = "./7.12/data/202304112345_packets.csv"  # fb: "./7.12/data/ca_1.csv" MAWI: "./7.12/data/202304112345_packets.csv"
    save_dir = "./2.20/FB/"  # fb: "./7.23/ca_1/" MAWI: "./7.23/2345/"

    a_s = appearance(buckets, b_id, True)  # 默认不分桶
    a_s.enumerate(epoch_num, "./2.20/FB/pre/pre_1_4.csv", save_dir)
