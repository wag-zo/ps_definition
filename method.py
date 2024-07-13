import pandas as pd
import math
import csv


class set_pre():
    def __init__(self) -> None:
        self.occur = {}
    
    def enumerate(self, epoch_num, epoch_len, start_time, csv_file_path, save_dir) -> None:
        df = pd.read_csv(csv_file_path, on_bad_lines='skip')
        for _, row in df.iterrows():
            e = row[1] + '>' + row[2]  # 获取ID
            if e not in self.occur:
                self.occur[e] = [0] * epoch_num  # 添加第一次出现的element
            epoch_now = math.floor((row[0] - start_time) / epoch_len)  # 计算对应epoch
            # if epoch_now >= epoch_num:
            #     print(row[0], epoch_now)
            self.occur[e][epoch_now] = 1
        
        pre_df = pd.DataFrame.from_dict(self.occur, orient='index')
        pre_df.to_csv(save_dir + "pre.csv", header=False)

 
  
class chunk_set_pre:
    """分块读取逐行写入，防止pandas难以处理过大数据"""
    def __init__(self) -> None:  
        self.occur = {}  
  
    def enumerate(self, epoch_num, epoch_len, start_time, csv_file_path, save_dir):
        chunk_size = 10**5  # 可以根据内存调整这个值
        for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size, on_bad_lines='skip'):
            for _, row in chunk.iterrows():
                e = f"{row[1]}>{row[2]}"
                epoch_now = math.floor((row[0] - start_time) / epoch_len)
                if e not in self.occur:
                    self.occur[e] = [0] * epoch_num
                self.occur[e][epoch_now] = 1
  
        with open(save_dir + "pre.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for key, epochs in self.occur.items():  # 逐行写入
                row = [key] + epochs
                writer.writerow(row)



class set_C():
    def __init__(self, thresh) -> None:
        self.occur = {}
        self.thresh = thresh
    
    def enumerate(self, epoch_num, pre_path, save_dir) -> None:
        df = pd.read_csv(pre_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
        df['Sum'] = df.iloc[:, 1:].astype(int).sum(axis=1)  # 对第一列后的元素求和，计算sum
        filtered_df = df[df['Sum'] > self.thresh]  # 抽出>thresh的行
        self.occur = filtered_df[['ID', 'Sum']].set_index('ID').to_dict()['Sum']  # (ID, Sum) 存入字典
        
        c_df = pd.DataFrame.from_dict(self.occur, orient='index', columns=['Value'])
        c_df.to_csv(f"{save_dir}set_C_thresh={self.thresh}.csv", header=False)



class set_A():
    def __init__(self, thresh, t = 0.1) -> None:
        self.persis = {}
        self.detect = {}
        self.thresh = thresh
        self.t = t
    
    def enumerate(self, epoch_num, pre_path, save_dir) -> None:
        df = pd.read_csv(pre_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
        for epoch in df.columns[1:]:
            for index, value in enumerate(df[epoch]):
                e = df.loc[index, 'ID']
                if value == 1:  # 原值衰减+1
                    self.persis[e] = self.persis.get(e, 0) * math.exp(-self.t) + 1
                elif value != 1 and e in self.persis:  # 只衰减
                    self.persis[e] *= math.exp(-self.t)
            
            epoch_now = int(epoch[5:])  # 计算对应epoch
            for e, value in self.persis.items():  # 遍历persis
                if value > self.thresh:  # 抽出超过阈值部分
                    if e not in self.detect:
                        self.detect[e] = [0] * epoch_num
                    self.detect[e][epoch_now] = 1
        
        a_df = pd.DataFrame.from_dict(self.detect, orient='index')
        a_df.to_csv(f"{save_dir}set_A_t={self.t}_thresh={self.thresh}.csv", header=False)



class set_B():
    def __init__(self, thresh, T = 8) -> None:
        self.persis = {}
        self.detect = {}
        self.thresh = thresh
        self.T = T
    
    def enumerate(self, epoch_num, pre_path, save_dir) -> None:
        df = pd.read_csv(pre_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
        for epoch in df.columns[1:]:
            epoch_now = int(epoch[5:])  # 计算对应epoch
            for index, value in enumerate(df[epoch]):
                e = df.loc[index, 'ID']
                if e not in self.persis:
                    self.persis[e] = 0
                self.persis[e] += value  # 加上当前epoch出现情况
                if epoch_now - self.T >= 1:  # 减去超过T的epoch出现情况
                    self.persis[e] -= df.iat[index, epoch_now - self.T]
            
            for e, value in self.persis.items():  # 遍历persis
                if value > self.thresh:  # 抽出超过阈值的部分
                    if e not in self.detect:
                        self.detect[e] = [0] * epoch_num
                    self.detect[e][epoch_now] = 1
        
        b_df = pd.DataFrame.from_dict(self.detect, orient='index')
        b_df.to_csv(f"{save_dir}set_B_T={self.T}_thresh={self.thresh}.csv", header=False)



if __name__ == "__main__":
    threshA = 8
    t = 0.05  # 衰减因子
    threshB = 10  # <= T
    T = 10  # 往前看的epoch数量
    threshC = 8
    epoch_len = 60  # 1个epoch的时间范围
    start_time = 1720328400.269537000
    end_time = 1720329300.167247000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    csv_file_path = "./7.12/data/202407071400_packets.csv"
    save_dir = "./7.12/results/202407071400"

    # _set_pre = set_pre()
    # _set_pre.enumerate(epoch_num, epoch_len, start_time, csv_file_path, save_dir)

    c_set_pre = chunk_set_pre()
    c_set_pre.enumerate(epoch_num, epoch_len, start_time, csv_file_path, save_dir)

    # _set_C = set_C(threshC)
    # _set_C.enumerate(epoch_num, save_dir + "pre.csv", save_dir)

    # _set_A = set_A(threshA, t)
    # _set_A.enumerate(epoch_num, save_dir + "pre.csv", save_dir)

    # _set_B = set_B(threshB, T)
    # _set_B.enumerate(epoch_num, save_dir + "pre.csv", save_dir)

