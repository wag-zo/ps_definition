import math
import os
import pandas as pd
from utils import align, suffix



def ps_epochs(epoch_num, gt_path, sim_path, save_dir):
    """记录每个epoch下s-sketch report的flow ID和估计值，并找到对应真实值"""
    df_gt = pd.read_csv(gt_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])  # 读取CSV文件
    df_sim = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
    # align(epoch_num, gt_path, sim_path, suffix(sim_path, "_aligned"))  # 将sim对齐到gt

    for i in range(epoch_num):  # 逐列遍历
        sim_non0 = df_sim[df_sim[f'Epoch{i}'] != 0][['src', f'Epoch{i}']]
        gt_i = df_gt[['src', f'Epoch{i}']]
        df_i = pd.merge(sim_non0, gt_i, on='src', how='left', suffixes=('_sim', '_gt'))  # 根据sim非零行连接gt
        df_i.to_csv(f'{save_dir}ps_epochs.csv', mode='a', index=False, header=False)

def thresh_set(epoch_num, thresh, data_path, save_path):
    """根据thresh截取flow ID，记录超过阈值的次数和第一次超过阈值的索引"""
    df = pd.read_csv(data_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
    new_set = []
    for _, row in df.iterrows():
        count, idx = 0, -1
        for i, r in enumerate(row[1:]):
            if r > thresh:
                count += 1
                if idx == -1:
                    idx = i
        if count != 0:
            new_set.append([row[0], count, idx])
    df_new = pd.DataFrame(new_set, columns=['src', 'count_above_thresh', 'first_above_thresh'])
    df_new.to_csv(save_path, index=False)

def prf_merge(a_path, b_path, save_dir):
    """计算precision，recall和F1分数，以及merge结果"""
    df_setA = pd.read_csv(a_path)
    df_setB = pd.read_csv(b_path)
    df_merge = pd.merge(df_setA, df_setB, how='left', on='src', suffixes=('_sim', '_gt'))  # 根据sim report的src连接gt
    df_merge.to_csv(f'{save_dir}merge_set.csv', index=False)

    # 计算precision, recall, f1_score
    df_ins = pd.merge(df_setA, df_setB, how='inner', on='src', suffixes=('_sim', '_gt'))  # 根据交集连接sim和gt
    count_ins, count_A, count_B = df_ins.shape[0], df_setA.shape[0], df_setB.shape[0]
    precision = count_ins / count_A
    recall = count_ins / count_B
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score




if __name__ == "__main__":
    epoch_len = 60  # fb: 300 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1681224300.077974000  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1681225200.150813000  # fb: 1475319422 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    thresh = 200  # 截取多少计算prf

    save_dir = "./8.22/MAWI/"  # fb: ./8.22/FB/ MAWI: ./8.22/MAWI/
    gt_path = save_dir + "spread_groundtruth.csv"  # 定义计算的pm和
    sim_path = save_dir + "spread_simulation.csv"  # ps-sketch计算的pm和

    # ps_epochs(epoch_num, gt_path, sim_path, save_dir)  # 记录report过的flow ID，估计值，真实值

    if not os.path.exists(save_dir + f"thresh={thresh}/"):  # 创建保存当前thresh的文件夹
        os.makedirs(save_dir + f"thresh={thresh}/")
    save_dir = save_dir + f"thresh={thresh}/"  # 修改save_dir

    thresh_set(epoch_num, thresh, sim_path, f'{save_dir}set_A.csv') # 根据thresh截取估计值
    thresh_set(epoch_num, thresh, gt_path, f'{save_dir}set_B.csv')  # 根据thresh截取真实值
    precision, recall, f1_score = prf_merge(f'{save_dir}set_A.csv', f'{save_dir}set_B.csv', save_dir)  # 存储merge结果并计算prf
    print("precision = {:.4%}, recall = {:.4%}, f1_score = {:.4%}".format(precision, recall, f1_score))
