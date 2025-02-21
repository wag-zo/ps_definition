import pandas as pd
import math
from utils import hash2int



class appearance():
    def __init__(self, buckets, b_id, b_flag = False) -> None:
        self.lifetime = {}
        self.e_time = {}
        self.buckets = buckets
        self.b_id = b_id
        self.b_flag = b_flag
    
    def enumerate(self, epoch_num, pre_path, save_dir) -> None:
        df = pd.read_csv(pre_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
        s_rows = []
        for epoch in df.columns[1:]:
            f_e = {}  # 判断当前轮每个flow中出现了哪些element
            for index, value in enumerate(df[epoch]):
                if value == 0:  # 跳过没出现的element
                    continue
                e = df.loc[index, 'ID']
                f = e.split(">")[0]
                if self.b_flag and hash2int("md5", f, self.buckets) != self.b_id:  # 跳过不属于当前桶的flow
                    continue
                if e not in self.e_time:  # 更新element出现次数
                    self.e_time[e] = 1
                else:
                    self.e_time[e] += 1
                epoch_now = int(epoch[5:])  # 计算对应epoch
                if f not in self.lifetime:
                    self.lifetime[f] = epoch_now  # 只记录首次出现的epoch
                if f not in f_e:
                    f_e[f] = [e]
                else:
                    f_e[f].append(e)

            for f, e_list in f_e.items():
                spread = len(e_list)
                appearance = sum([self.e_time[e] for e in e_list]) / (epoch_now - self.lifetime[f] + 1) / spread
                s_rows.append([epoch_now, f, appearance, spread])  # 更新到列表
            print(f"Epoch{epoch_now}: appearance is done")

        s_df = pd.DataFrame(s_rows, columns=['Epoch', 'flow_id', 'appearance', 'spread'])
        save_path = f"{save_dir}appe_spread_{self.b_id}.csv" if self.b_flag else f"{save_dir}appe_spread.csv"
        s_df.to_csv(save_path, header=False, index=False)



if __name__ == "__main__":
    tau = 0.05  # fb: 0.1 MAWI: 0.05
    buckets = 6  # 分桶数
    b_id = 1  # 当前计算哪个桶
    
    epoch_len = 60  # fb: 300 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1681224300.077974000  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1681225200.150813000  # fb: 1475319422 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    print("epoch_num = ", epoch_num)
    csv_file_path = "./7.12/data/202304112345_packets.csv"  # fb: "./7.12/data/ca_1.csv" MAWI: "./7.12/data/202304112345_packets.csv"
    save_dir = "./2.20/MAWI/"  # fb: "./7.23/ca_1/" MAWI: "./7.23/2345/"

    a_s = appearance(buckets, b_id, False)  # 默认不分桶
    a_s.enumerate(epoch_num, "./7.12/results/202304112345/pre.csv", save_dir)
