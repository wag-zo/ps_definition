import pandas as pd
import math
import csv
import time


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
            print("chunk finished")
  
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
        filtered_df = df[df['Sum'] >= self.thresh]  # 抽出>thresh的行
        self.occur = filtered_df[['ID', 'Sum']].set_index('ID').to_dict()['Sum']  # (ID, Sum) 存入字典
        
        c_df = pd.DataFrame.from_dict(self.occur, orient='index', columns=['Value'])
        c_df.to_csv(f"{save_dir}set_C_thresh={self.thresh}.csv", header=False)



class chunk_set_C():
    def __init__(self, thresh) -> None:
        self.occur = {}
        self.thresh = thresh
    
    def enumerate(self, epoch_num, pre_path, save_dir) -> None:
        chunk_size = 20**5  # 可以根据内存调整这个值
        for chunk in pd.read_csv(pre_path, header=None, chunksize=chunk_size, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)]):
            chunk['Sum'] = chunk.iloc[:, 1:].astype(int).sum(axis=1)  # 对第一列后的元素求和，计算sum
            filtered_df = chunk[chunk['Sum'] >= self.thresh]  # 抽出>thresh的行
            update_sums = filtered_df[['ID', 'Sum']].set_index('ID').to_dict()['Sum']
            for id, sum in update_sums.items():  # (ID, Sum) 存入字典
                self.occur[id] = sum
        
        c_df = pd.DataFrame.from_dict(self.occur, orient='index', columns=['Value'])
        c_df.to_csv(f"{save_dir}set_C_thresh={self.thresh}.csv", header=False)
        # with open(f"{save_dir}set_C_thresh={self.thresh}.csv", 'w', newline='', encoding='utf-8') as f:
        #     writer = csv.writer(f)
        #     for key, epochs in self.occur.items():  # 逐行写入
        #         row = [key] + epochs
        #         writer.writerow(row)



class set_A():
    def __init__(self, thresh, tau = 0.1) -> None:
        self.persis = {}
        self.detect = {}
        self.thresh = thresh
        self.tau = tau
    
    def enumerate(self, epoch_num, pre_path, save_dir) -> None:
        df = pd.read_csv(pre_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
        for epoch in df.columns[1:]:
            for index, value in enumerate(df[epoch]):
                e = df.loc[index, 'ID']
                if value == 1:  # 原值衰减+1
                    self.persis[e] = self.persis.get(e, 0) * math.exp(-self.tau) + 1
                elif value != 1 and e in self.persis:  # 只衰减
                    self.persis[e] *= math.exp(-self.tau)
            
            epoch_now = int(epoch[5:])  # 计算对应epoch
            for e, value in self.persis.items():  # 遍历persis
                if value >= self.thresh:  # 抽出超过阈值部分
                    if e not in self.detect:
                        self.detect[e] = [0] * epoch_num
                    self.detect[e][epoch_now] = 1
        
        a_df = pd.DataFrame.from_dict(self.detect, orient='index')
        a_df.to_csv(f"{save_dir}set_A_tau={self.tau}_thresh={self.thresh}.csv", header=False)



class chunk_set_A():
    def __init__(self, thresh, tau = 0.1) -> None:
        self.persis = {}
        self.detect = {}
        self.thresh = thresh
        self.tau = tau
    
    def enumerate(self, epoch_num, pre_path, save_dir) -> None:
        chunk_size = 20**5  # 可以根据内存调整这个值
        for chunk in pd.read_csv(pre_path, header=None, chunksize=chunk_size, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)]):
            for epoch in chunk.columns[1:]:
                epoch_now = int(epoch[5:])  # 计算对应epoch
                ids = chunk['ID']
                epoch_values = chunk[epoch]
                for _, (id_value, epoch_value) in enumerate(zip(ids, epoch_values)):
                    e = id_value
                    if epoch_value == 1:  # 原值衰减+1
                        self.persis[e] = self.persis.get(e, 0) * math.exp(-self.tau) + 1
                    elif epoch_value != 1 and e in self.persis:  # 只衰减
                        self.persis[e] *= math.exp(-self.tau)
            
                for e, value in self.persis.items():  # 遍历persis
                    if value >= self.thresh:  # 抽出超过阈值部分
                        if e not in self.detect:
                            self.detect[e] = [0] * epoch_num
                        last_1_pos = next((i for i, x in enumerate(self.detect[e]) if x == 1), -1)
                        if epoch_now > last_1_pos:  # 避免不同chunk间相互影响
                            self.detect[e][epoch_now] = 1
            print("chunk finished")
        
        a_df = pd.DataFrame.from_dict(self.detect, orient='index')
        a_df.to_csv(f"{save_dir}set_A_tau={self.tau}_thresh={self.thresh}.csv", header=False)
        # with open(f"{save_dir}set_A_t={self.tau}_thresh={self.thresh}.csv", 'w', newline='', encoding='utf-8') as f:
        #     writer = csv.writer(f)
        #     for key, epochs in self.detect.items():  # 逐行写入
        #         row = [key] + epochs
        #         writer.writerow(row)



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
                if epoch_now - self.T >= 0:  # 减去超过T的epoch出现情况
                    self.persis[e] -= df.iat[index, epoch_now - self.T + 1]
            
            for e, value in self.persis.items():  # 遍历persis
                if value >= self.thresh:  # 抽出超过阈值的部分
                    if e not in self.detect:
                        self.detect[e] = [0] * epoch_num
                    self.detect[e][epoch_now] = 1
        
        b_df = pd.DataFrame.from_dict(self.detect, orient='index')
        b_df.to_csv(f"{save_dir}set_B_T={self.T}_thresh={self.thresh}.csv", header=False)



class chunk_set_B():
    def __init__(self, thresh, T = 8) -> None:
        self.persis = {}
        self.flag = {}
        self.detect = {}
        self.thresh = thresh
        self.T = T
    
    def enumerate(self, epoch_num, pre_path, save_dir) -> None:
        chunk_size = 20**5  # 可以根据内存调整这个值
        for chunk in pd.read_csv(pre_path, header=None, chunksize=chunk_size, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)]):
            # print(chunk.head())
            for epoch in chunk.columns[1:]:
                epoch_now = int(epoch[5:])  # 计算对应epoch
                ids = chunk['ID']
                epoch_values = chunk[epoch]
                if epoch_now - self.T >= 0:
                    old_epoch_values = chunk[f'Epoch{epoch_now - self.T}']
                else:
                    old_epoch_values = pd.Series(0, index=chunk.index)  # 创建一个全 0 的 Series
                
                for _, (id_value, epoch_value, old_epoch_value) in enumerate(zip(ids, epoch_values, old_epoch_values)):
                    e = id_value
                    if e not in self.persis:
                        self.persis[e] = 0
                    self.persis[e] += epoch_value - old_epoch_value

                for e, value in self.persis.items():  # 遍历persis
                    if value >= self.thresh:  # 抽出超过阈值的部分
                        if e not in self.detect:
                            self.detect[e] = [0] * epoch_num
                        last_1_pos = next((i for i, x in enumerate(self.detect[e]) if x == 1), -1)
                        if epoch_now > last_1_pos:  # 避免不同chunk间相互影响
                            self.detect[e][epoch_now] = 1
            print("chunk finished")

        b_df = pd.DataFrame.from_dict(self.detect, orient='index')
        b_df.to_csv(f"{save_dir}set_B_T={self.T}_thresh={self.thresh}.csv", header=False)
        # with open(f"{save_dir}set_B_T={self.T}_thresh={self.thresh}.csv", 'w', newline='', encoding='utf-8') as f:
        #     writer = csv.writer(f)
        #     for key, epochs in self.detect.items():  # 逐行写入
        #         row = [key] + epochs
        #         writer.writerow(row)



class spread_gt():
    def __init__(self, tau = 0.1) -> None:
        self.persis = {}
        self.spread = {}
        self.tau = tau
    
    def enumerate(self, epoch_num, pre_path, save_dir) -> None:
        df = pd.read_csv(pre_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
        for epoch in df.columns[1:]:
            e_flag = {}
            es = set()
            for index, value in enumerate(df[epoch]):
                e = df.loc[index, 'ID']
                e_flag[e] = True if value == 1 else False  # 出现过则为True，没出现过则为False
                if value == 1:  # 原值衰减+1
                    self.persis[e] = self.persis.get(e, 0) * math.exp(-self.tau) + 1
                elif value != 1 and e in self.persis:  # 只衰减
                    self.persis[e] *= math.exp(-self.tau)

            epoch_now = int(epoch[5:])  # 计算对应epoch
            print(f"Epoch{epoch_now}: Pm is done")
            for e, value in self.persis.items():  # 遍历persis
                if e_flag[e] == False:  # 没出现过则不计算
                    continue
                src = e.split(">")[0]  # 取出src
                if src not in self.spread:
                    self.spread[src] = [0] * epoch_num
                self.spread[src][epoch_now] += value
            print(f"Epoch{epoch_now}: Sm is done")
        
        s_df = pd.DataFrame.from_dict(self.spread, orient='index')
        s_df.to_csv(f"{save_dir}spread_groundtruth.csv", header=False)



if __name__ == "__main__":
    threshA = 3
    tau = 0.1  # fb: 0.1 MAWI: 0.5
    threshB = 8  # <= T
    T = 8  # 往前看的epoch数量
    threshC = 40
    
    epoch_len = 300  # fb: 300 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1475305136  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1475319422  # fb: 1475319422 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    print("epoch_num = ", epoch_num)
    csv_file_path = "./7.12/data/ca_1.csv"  # "./7.12/data/ca_1.csv"
    save_dir = "./7.23/ca_1/"  # "./7.23/ca_1/"

    # _set_pre = set_pre()
    # _set_pre.enumerate(epoch_num, epoch_len, start_time, csv_file_path, save_dir)

    # c_set_pre = chunk_set_pre()
    # c_set_pre.enumerate(epoch_num, epoch_len, start_time, csv_file_path, save_dir)

    # _set_C = set_C(threshC)
    # _set_C.enumerate(epoch_num, save_dir + "pre.csv", save_dir)

    # c_set_C = chunk_set_C(threshC)
    # c_set_C.enumerate(epoch_num, save_dir + "pre.csv", save_dir)
    
    # _set_A = set_A(threshA, tau)
    # _set_A.enumerate(epoch_num, save_dir + "pre.csv", save_dir)

    # c_set_A = chunk_set_A(threshA, tau)
    # c_set_A.enumerate(epoch_num, save_dir + "pre.csv", save_dir)

    # _set_B = set_B(threshB, T)
    # _set_B.enumerate(epoch_num, save_dir + "pre.csv", save_dir)

    # c_set_B = chunk_set_B(threshB, T)
    # c_set_B.enumerate(epoch_num, save_dir + "pre.csv", save_dir)

    spread = spread_gt(tau)
    spread.enumerate(epoch_num, save_dir + "pre.csv", save_dir)

