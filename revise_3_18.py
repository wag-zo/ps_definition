import pandas as pd
import numpy as np
import math
import os
from collections import defaultdict, Counter
from revise_22 import kt_persis, kt_all


def thresh_set_flows(gt_path, sim_path, save_dir, epoch_num, thresh):
    """根据thresh截取flow ID, 记录sim和gt中第一次超过阈值的索引, 以及对应的spread"""
    df_gt = pd.read_csv(gt_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])  # 读取CSV文件
    df_sim = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])

    thresh_sim = []
    for _, row in df_sim.iterrows():
        result = [(idx, val) for idx, val in enumerate(row[1:]) if val > thresh]  # 根据thresh查找df, 并记录对应列名和值
        if result:
            thresh_sim.append([row[0], result[0][0], result[0][1]])  # [src, idx, val]
    df_thresh_sim = pd.DataFrame(thresh_sim, columns=['src', 'first_above_thresh_idx', 'first_above_thresh_val'])

    df_thresh = pd.merge(df_thresh_sim, df_gt, on='src', how='left')  # 根据sim连接gt
    for i, row in df_thresh.iterrows():
        result = [idx for idx, val in enumerate(row[3:]) if val > thresh]
        if result:
            df_thresh.loc[i, 'DTD'] = row[1] - result[0]  # DTD = sim的首次检测位置 - gt的首次检测位置
            df_thresh.loc[i, 'first_above_thresh_val_gt'] = row[f'Epoch{row[1] + 3}']  # sim首次检测时gt的值
    df_thresh = df_thresh[['src', 'first_above_thresh_idx', 'DTD', 'first_above_thresh_val', 'first_above_thresh_val_gt']]
    df_thresh.to_csv(f"{save_dir}thresh_flows.csv", header=False, index=False)

def detected_nums_ps(ps_path, save_dir, epoch_num, thresh=10000):
    """找到sim中超过阈值的flow个数"""
    df_ps = pd.read_csv(ps_path, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
    cols = df_ps.columns[1:]
    result = {col: (df_ps[col] > thresh).sum() for col in cols}  # 有多少个epoch超过thresh
    pd.DataFrame.from_dict(result, orient='index').to_csv(f'{save_dir}ps_nums.csv', header=False, index=False)

def detected_nums_kt(pre_path, persis_path, save_dir, epoch_num, ks, thresh=10000):
    """找到kt方法中超过阈值的flow个数"""
    for k in ks:
        kt_all(epoch_num, persis_path, pre_path, f'{save_dir}{k}_', k)
        df = pd.read_csv(f'{save_dir}{k}_kt_all.csv', header=None, names=['Epoch', 'flow_id', 'spread', 'kt_ps', 'mean_kt_p'])
        k_df = df[df['kt_ps'] > thresh / k]  # 截取需要的行
        k_df = k_df[['Epoch', 'flow_id', 'kt_ps']]  # 截取需要的列
        result = df.groupby('Epoch')['flow_id'].nunique().reset_index()  # 根据Epoch分组并提取flow_id中值的数量
        result.to_csv(f'{save_dir}{k}_nums.csv', header=False, index=False)


if __name__ == "__main__":
    tau = 0.1  # fb: 0.1 MAWI: 0.05  # ps-sketch衰减系数
    t = 8  # fb: 16 MAWI: 8  # kt向前查找的周期数
    ks = [2, 3, 4, 5]  # kt判断persistent元素的阈值
    thresh = 10000  # ps/kt判断persistent spreader的阈值

    epoch_len = 60  # fb: 60 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1079884900.754052000  # fb: 1475305136 MAWI: 1681224300.077974000 witty:1079884900.754052000
    end_time = 1079888500.753756000  # fb: 1475392025 MAWI: 1681225200.150813000 witty:1079888500.753756000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    save_dir = "./3.18/witty/"
    pre_path = save_dir + "pre.csv"  # 预处理文件
    gt_path = save_dir + "spread_groundtruth.csv"  # 定义计算的pm和
    sim_path = save_dir + "spread_simulation.csv"  # ps-sketch计算的pm和

    # thresh_set_flows(gt_path, sim_path, save_dir, epoch_num, thresh)  # 实验一

    # kt_persis(epoch_num, pre_path, save_dir, t)  # 所有元素的kt-persistence
    # detected_nums_ps(sim_path, save_dir, epoch_num, thresh)  # ps-sketch对应数量
    detected_nums_kt(pre_path, f'{save_dir}kt_persis.csv', save_dir, epoch_num, ks, thresh)  # kt方法对应数量
