import pandas as pd
import numpy as np
import math
import os
from collections import Counter


def appear(epoch_num, gt_path, sim_path, save_dir):
    """以groundtruth为基准，若当前epoch src出现在groundtruth和simulation中则存储1，
    仅出现在groundtruth中则存储0，都不出现则存储-1，每一行形如[src, 0, 0, 1, 1, ……]"""
    # 读取CSV文件
    df_gt = pd.read_csv(gt_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
    df_sim = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
    
    # 使用pandas，根据id列合并DataFrame并存储
    df_ap = pd.DataFrame(df_gt['src'])
    for i in range(epoch_num):
        gt_0 = df_gt[df_gt[f'Epoch{i}'] == 0]
        sim_non0 = df_sim[df_sim[f'Epoch{i}'] != 0]

        gt_sim = df_ap['src'].isin(sim_non0['src'])
        not_gt = df_ap['src'].isin(gt_0['src'])
        # print((gt_sim == True).sum(), (not_gt == True).sum())
        df_ap[f'Epoch{i}'] = np.where(gt_sim, 1,  # 如果在groundtruth中且在simulation中，设为1
                      np.where(not_gt, -1,  # 如果不在groundtruth中，设为-1
                              0))  #   # 如果在groundtruth中但不在simulation中，设为0
    
    df_ap.to_csv(f'{save_dir}appear.csv', index=False, header=False)



def compute_me(epoch_num, ap_path, gt_path, sim_path):
    """计算每轮epoch，出现在groundtruth-simulation的src个数，出现在groundtruth∩simulation的RMSE和MRE"""
    df_ap = pd.read_csv(ap_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
    df_gt = pd.read_csv(gt_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
    df_sim = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])

    num_sub, rmse, mre = [], [], []
    for i in range(epoch_num):
        ap_0 = df_ap[df_ap[f'Epoch{i}'] == 0]
        num_sub.append(ap_0.shape[0])

        ap_1 = df_ap[df_ap[f'Epoch{i}'] == 1]
        gt = df_gt.loc[df_gt['src'].isin(ap_1['src']), f'Epoch{i}'].values
        sim = df_sim.loc[df_sim['src'].isin(ap_1['src']), f'Epoch{i}'].values
        rmse.append(np.sqrt(((sim - gt) ** 2).mean()))
        mre.append(np.sum(np.abs((sim - gt) / gt)))
    
    return num_sub, rmse, mre



if __name__ == "__main__":
    epoch_len = 300  # 1个epoch的时间范围
    start_time = 1475305136
    end_time = 1475319422
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量

    save_dir = "./7.23/ca_1/"
    gt_path = "./7.23/ca_1/spread_groundtruth_tau=0.1.csv"
    sim_path = "./7.23/ca_1/spread_simulation_tau=0.1_w=128_b=4.csv"

    # appear(epoch_num, gt_path, sim_path, save_dir)
    num_sub, rmse, mre = compute_me(epoch_num, save_dir + "appear.csv", gt_path, sim_path)
    print(num_sub, rmse, mre)


