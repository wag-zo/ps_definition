import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import math
import time


def num_in_t(occur, t, batch_size = 100000):
    """
    occur是所有epoch的出现情况，t是要查找的周期长度
    batch_size指定了每次处理的行数，默认值为10000
    """
    num_rows, num_cols = occur.shape
    result = np.empty((num_rows, num_cols), dtype=np.uint8)
    for i in range(0, num_rows, batch_size):
        end = min(i + batch_size, num_rows)
        cumsum_batch = np.cumsum(occur[i:end, :], axis=1)  # 对当前批次计算前缀和

        result[i:end, :t] = cumsum_batch[:, :t]
        result[i:end, t:] = cumsum_batch[:, t:] - cumsum_batch[:, :-t]
    
    return result
 
def kt_persis(epoch_num, pre_path, save_dir, t, batch_size = 100000):
    df = pd.read_csv(pre_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
    ids = df['ID'].values
    data = df.drop(columns=['ID']).values
    
    window_sums = num_in_t(data, t)  # 计算滑动窗口和
    num_rows = ids.shape[0]
    num_cols = window_sums.shape[1]

    with open(f"{save_dir}kt_persis_t={t}.csv", 'w', newline='') as f:  # 打开文件准备写入
        for i in range(0, num_rows, batch_size):  # 分批处理数据
            end = min(i + batch_size, num_rows)
            batch_df = pd.DataFrame(np.hstack((ids[i:end].reshape(-1, 1), window_sums[i:end, :])),\
                                    columns=['ID'] + [f'Epoch{i}' for i in range(num_cols)])  # 创建当前批次的 DataFrame
            batch_df.to_csv(f, header=False, index=False, mode='a')

    # s_df = pd.DataFrame(np.hstack((ids.reshape(-1, 1), window_sums)), columns=['ID'] + [f'Epoch{i}' for i in range(window_sums.shape[1])])
    # s_df.to_csv(f"{save_dir}kt_persis.csv", header=False, index=False)

def kt_ps(epoch_num, persis_path, save_dir, t, k) -> None:
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
            s_rows.append([epoch_now, f, len(e_list), f_kte[f], sum(e_list) / len(e_list)])  # 更新到列表
        print(f"Epoch{epoch_now}: kt is done")
    
    s_df = pd.DataFrame(s_rows, columns=['Epoch', 'flow_id', 'spread', 'kt_ps', 'mean_kt_p'])
    s_df.to_csv(f"{save_dir}kt_metrics_t={t}_k={k}.csv", header=False, index=False)



if __name__ == "__main__":
    t = 8  # 向前查找的周期数
    k = 4  # 判断persistent元素的阈值

    epoch_len = 60  # fb: 300 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1475305136  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1475392025  # fb: 1475392025 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    print("epoch_num = ", epoch_num)
    csv_file_path = "./7.12/data/202304112345_packets.csv"  # fb: "./7.12/data/ca_1.csv" MAWI: "./7.12/data/202304112345_packets.csv"
    save_dir = "./2.22/FB/"  # fb: "./7.23/ca_1/" MAWI: "./7.23/2345/"

    kt_start = time.time()
    kt_persis(epoch_num, "./2.22/FB/pre_1_40.csv", save_dir, t)  # MAWI:26.839514017105103
    print("Need time:", time.time() - kt_start)

    kt_start = time.time()
    kt_ps(epoch_num, f"{save_dir}kt_persis_t={t}.csv", save_dir, t, k)  # MAWI:586.3150606155396
    print("Need time:", time.time() - kt_start)
