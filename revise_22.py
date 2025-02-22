import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import math
import time


def num_in_t(occur, t):
    """occur是所有epoch的出现情况，t是要查找的周期长度"""
    cumsum_occur = np.cumsum(occur, axis=1)  # 计算前缀和
    result = np.empty_like(cumsum_occur)
    result[:, :t] = cumsum_occur[:, :t]  # <=t
    result[:, t:] = cumsum_occur[:, t:] - cumsum_occur[:, :-t]  # >t
    
    return result
 
def kt_persis(epoch_num, pre_path, save_dir, t):
    """计算所有元素的kt-persistence，即在最近t个周期内的出现次数"""
    df = pd.read_csv(pre_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
    ids = df['ID'].values
    data = df.drop(columns=['ID']).values
    
    window_sums = num_in_t(data, t)  # 计算滑动窗口和

    s_df = pd.DataFrame(np.hstack((ids.reshape(-1, 1), window_sums)), columns=['ID'] + [f'Epoch{i}' for i in range(window_sums.shape[1])])
    s_df.to_csv(f"{save_dir}kt_persis.csv", header=False, index=False)

def kt_ps(epoch_num, persis_path, save_dir, k) -> None:
    """按照epoch和flow_id索引，计算flow中出现过多少元素(spread)，其中满足k的元素个数(kt-spread)，出现过元素的平均persistence(mean kt-persistence)"""
    df = pd.read_csv(persis_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
    s_rows = []
    for epoch in df.columns[1:]:
        f_e = defaultdict(list)  # 本轮所有出现过的element_id
        f_kte = Counter()  # 本轮满足kt的element个数
        epoch_now = int(epoch[5:])  # 计算对应epoch
        for index, value in enumerate(df[epoch]):
            e = df.loc[index, 'ID']
            f = e.split(">")[0]
            f_e[f].append(value)
            if value >= k:
                f_kte[f] += 1
        
        for f, e_list in f_e.items():
            s_rows.append([epoch_now, f, f_kte[f], len(e_list), sum(e_list) / len(e_list)])  # 更新到列表
        print(f"Epoch{epoch_now}: kt is done")
    
    s_df = pd.DataFrame(s_rows, columns=['Epoch', 'flow_id', 'spread', 'kt_ps', 'mean_kt_p'])
    s_df.to_csv(f"{save_dir}kt_metrics.csv", header=False, index=False)



if __name__ == "__main__":
    t = 8  # 向前查找的周期数
    k = 4  # 判断persistent元素的阈值

    epoch_len = 60  # fb: 300 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1681224300.077974000  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1681225200.150813000  # fb: 1475319422 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    print("epoch_num = ", epoch_num)
    csv_file_path = "./7.12/data/202304112345_packets.csv"  # fb: "./7.12/data/ca_1.csv" MAWI: "./7.12/data/202304112345_packets.csv"
    save_dir = "./2.22/MAWI/"  # fb: "./7.23/ca_1/" MAWI: "./7.23/2345/"

    # kt_start = time.time()
    # kt_persis(epoch_num, "./7.12/results/202304112345/pre.csv", save_dir, t)  # MAWI:26.839514017105103
    # print("Need time:", time.time() - kt_start)

    kt_start = time.time()
    kt_ps(epoch_num, save_dir+"kt_persis.csv", save_dir, k)  # MAWI:586.3150606155396
    print("Need time:", time.time() - kt_start)
