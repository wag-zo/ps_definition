import math
import os
import pandas as pd
from comp_22 import thresh_set

def TP_ps_epochs(gt_TP_path, sim_TP_path, save_dir, count_src):
    """记录估计值和真实值找到的TP spreader"""
    df_gt = pd.read_csv(gt_TP_path)  # 读取CSV文件
    df_sim = pd.read_csv(sim_TP_path)

    df_new = pd.merge(df_gt, df_sim, on='src', how='inner', suffixes=('_gt', '_sim'))  # 只要TP情况
    df_new = df_new[['src', 'first_above_thresh_sim', 'first_above_thresh_gt', 'count_above_thresh_sim', 'count_above_thresh_gt']]
    df_new.to_csv(f'{save_dir}TP_ps_epochs.csv', index=False)

    count_TP, count_gt, count_sim = df_new.shape[0], df_gt.shape[0], df_sim.shape[0]
    count_FP, count_FN = count_sim - count_TP, count_gt - count_TP
    count_TN = count_src - count_TP - count_FP - count_FN
    return count_TP, count_TN, count_FP, count_FN

def thresh_set_epochs(epoch_num, thresh, gt_path, sim_path, save_dir):
    """根据thresh截取flow ID，记录超过阈值的次数和第一次超过阈值的索引"""
    df_gt = pd.read_csv(gt_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])  # 读取CSV文件
    df_sim = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
    gt_count, sim_count = (df_gt.iloc[:, 1:] > thresh).sum(), (df_sim.iloc[:, 1:] > thresh).sum()

    df_new = pd.DataFrame({  
        'sim_counts': sim_count,  
        'gt_counts': gt_count  
    })
    df_new.to_csv(f'{save_dir}thresh_counts_epochs.csv')


if __name__ == "__main__":
    data_name = "MAWI"  # FB | MAWI

    epoch_len = 300 if data_name == "FB" else 60 # fb: 300 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1475305136 if data_name == "FB" else 1681224300.077974000  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1475319422 if data_name == "FB" else 1681225200.150813000  # fb: 1475319422 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    thresh = 50 if data_name == "FB" else 200  # fb: 50 MAWI: 200  # 截取多少计算spreader
    count_src = 47261 if data_name == "FB" else 145090  # fb: 47261 MAWI: 145090  # 原始数据集中包含多少src

    save_dir = "./9.8/FB/" if data_name == "FB" else "./9.8/MAWI/"  # fb: ./8.22/FB/ MAWI: ./8.22/MAWI/
    gt_path = save_dir + "spread_groundtruth.csv"  # 定义计算的pm和
    sim_path = save_dir + "spread_simulation.csv"  # ps-sketch计算的pm和

    thresh_set(epoch_num, thresh, sim_path, f'{save_dir}set_sim.csv') # 根据thresh截取估计值
    thresh_set(epoch_num, thresh, gt_path, f'{save_dir}set_gt.csv')  # 根据thresh截取真实值

    count_TP, count_TN, count_FP, count_FN = TP_ps_epochs(
        f'{save_dir}set_gt.csv', f'{save_dir}set_sim.csv', save_dir, count_src)  # 记录report过的flow ID，估计值，真实值
    print(f"TP = {count_TP}, TN = {count_TN}, FP = {count_FP}, FN = {count_FN}")
    thresh_set_epochs(epoch_num, thresh, gt_path, sim_path, save_dir)


