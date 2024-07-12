import pandas as pd
import math
import csv


class set_pre():
    def __init__(self) -> None:
        self.occur = {}
    
    def enumrate(self, epoch_num, epoch_len, start_time, csv_file_path, save_dir) -> None:
        df = pd.read_csv(csv_file_path)
        for _, row in df.iterrows():
            e = row[1] + '>' + row[2]  # 获取ID
            if e not in self.occur:
                self.occur[e] = [0] * epoch_num  # 添加第一次出现的element
            epoch_now = math.floor((row[0] - start_time) / epoch_len)  # 计算对应epoch
            self.occur[e][epoch_now] = 1
        
        pre_df = pd.DataFrame.from_dict(self.occur, orient='index')
        pre_df.to_csv(save_dir + "pre.csv", header=False)



class set_C():
    def __init__(self, thresh) -> None:
        self.occur = {}
        self.thresh = thresh
    
    def enumrate(self, epoch_num, pre_path, save_dir) -> None:
        df = pd.read_csv(pre_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
        df['Sum'] = df.iloc[:, 1:].astype(int).sum(axis=1)  # 对第一列后的元素求和，计算sum
        filtered_df = df[df['Sum'] > self.thresh]  # 抽出>thresh的行
        self.occur = filtered_df[['ID', 'Sum']].set_index('ID').to_dict()['Sum']  # (ID, Sum) 存入字典
        
        c_df = pd.DataFrame.from_dict(self.occur, orient='index', columns=['Value'])
        c_df.to_csv(save_dir + "set_C.csv", header=False)


class set_A():
    def __init__(self, thresh, t = 0.1) -> None:
        self.persis = {}
        self.detect = {}
        self.thresh = thresh
        self.t = t
    
    def enumrate(self, epoch_num, pre_path, save_dir) -> None:
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
        a_df.to_csv(save_dir + "set_A.csv", header=False)



class set_B():
    def __init__(self, thresh, T = 8) -> None:
        self.persis = {}
        self.detect = {}
        self.thresh = thresh
        self.T = T
    
    def enumrate(self, epoch_num, pre_path, save_dir) -> None:
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
        b_df.to_csv(save_dir + "set_B.csv", header=False)




if __name__ == "__main__":
    threshA = 2
    t = 0.1  # 衰减因子
    threshB = 8  # <= T
    T = 8  # 往前看的epoch数量
    threshC = 8
    epoch_num = 12  # epoch的数量
    epoch_len = 30  # 1个epoch的时间范围
    start_time = 1261067509.401605
    csv_file_path = "./7.9/results/univ1_pt2/univ1_pt2_packets.csv"
    save_dir = "./7.11/results/"

    # _set_pre = set_pre()
    # _set_pre.enumrate(epoch_num, epoch_len, start_time, csv_file_path, save_dir)

    # _set_C = set_C(threshC)
    # _set_C.enumrate(epoch_num, save_dir + "pre.csv", save_dir)

    # _set_A = set_A(threshA, t)
    # _set_A.enumrate(epoch_num, save_dir + "pre.csv", save_dir)

    _set_B = set_B(threshB, T)
    _set_B.enumrate(epoch_num, save_dir + "pre.csv", save_dir)

