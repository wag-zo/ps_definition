import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import math
import sys
import os
from revise_22 import kt_persis, kt_all, merge_ps_all, ps_thresh

def merge_ps_kt(ps_thresh_path, save_dir, ks):
    """计算不同方法的交集差集"""
    left_2 = ['Epoch', 'flow_id', 'ps_sim', 'ps_gt']
    right_3 = ['Epoch', 'flow_id', 'spread', 'kt_ps', 'mean_kt_p']
    
    df_merge = pd.read_csv(ps_thresh_path, header=None, names=left_2)
    for k in ks:
        df_kt_i = pd.read_csv(f'{save_dir}k={k}/kt_all.csv', header=None, names=right_3)
        df_kt_i.drop(['spread', 'mean_kt_p'], axis=1, inplace=True)  # 删除不需要的列
        df_merge = pd.merge(df_merge, df_kt_i, on=['Epoch', 'flow_id'], how='left', suffixes=('', f'_{k}'))

    df_merge.to_csv(f'{save_dir}result/diff_ks.csv', index=False, header=False)




if __name__ == "__main__":
    data_type = "fb"  # fb or MAWI
    t = 16 if data_type == "fb" else 8  # fb: 16 MAWI: 8  # kt向前查找的周期数
    ks = [6, 8, 10, 12] if data_type == "fb" else [3, 4, 5, 6]  # fb: [6, 8, 10, 12] MAWI: [3, 4, 5, 6]  # kt判断persistent元素的阈值
    thresh_kt = 10  # kt判断persistent spreader的阈值
    thresh_ps = 50 if data_type == "fb" else 200  # fb: 50 MAWI: 200  # ps判断persistent spreader的阈值

    epoch_len = 60  # fb: 60 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1475305136 if data_type == "fb" else 1681224300.077974000  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1475392025 if data_type == "fb" else 1681225200.150813000  # fb: 1475392025 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    save_dir = "./3.5/FB/"  if data_type == "fb" else "./3.5/MAWI/"
    pre_path = save_dir + "pre_1_40.csv"  # 预处理文件
    gt_path = save_dir + "spread_groundtruth.csv"  # 定义计算的pm和
    sim_path = save_dir + "spread_simulation.csv"  # ps-sketch计算的pm和

    kt_persis(epoch_num, pre_path, save_dir, t)  # 所有元素的kt-persistence
    for k in ks:  # 所有元素的kt-persistent spread
        if not os.path.exists(save_dir + f"k={k}/"):  # 创建保存当前kt_all的文件夹
            os.makedirs(save_dir + f"k={k}/")
        kt_all(epoch_num, f"{save_dir}kt_persis.csv", pre_path, f"{save_dir}k={k}/", k)

    # merge_ps_all(epoch_num, gt_path, sim_path, save_dir)  # 将ps-sketch中的模拟和真实值整合在一起, *修改thresh不需要重跑
    # ps_thresh(f"{save_dir}ps_all.csv", save_dir, thresh_ps)  # 截取>thresh的ps-persistent spread

    if not os.path.exists(save_dir + f"result/"):  # 创建保存结果的文件夹
        os.makedirs(save_dir + f"result/")
    merge_ps_kt(f"{save_dir}ps_thresh.csv", save_dir, ks)  # 求ps和kt的交集差集
