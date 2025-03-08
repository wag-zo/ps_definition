import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import math
import sys
import os
from revise_22 import kt_persis, kt_all, merge_ps_all, ps_thresh, example
from method import chunk_set_A, chunk_set_B, chunk_set_C

def merge_ps_kt(ps_thresh_path, save_dir, ks):
    """计算不同方法的交集差集"""
    df_merge = pd.read_csv(ps_thresh_path, header=None, names=['Epoch', 'flow_id', 'ps_sim', 'ps_gt'])
    for k in ks:
        df_kt_i = pd.read_csv(f'{save_dir}k={k}/kt_all.csv', header=None, names=['Epoch', 'flow_id', 'spread', 'kt_ps', 'mean_kt_p'])
        df_kt_i.drop(['spread', 'mean_kt_p'], axis=1, inplace=True)  # 删除不需要的列
        df_merge = pd.merge(df_merge, df_kt_i, on=['Epoch', 'flow_id'], how='left', suffixes=('', f'_{k}'))

    df_merge.to_csv(f'{save_dir}result/diff_ks.csv', index=False, header=False)

def find_diff_flow(sim_path, pre_path, save_dir, flow_id, epoch_num, ks):
    """找到flow_id对应的模拟值 kt值 和element情况"""
    df_sim = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
    dtypes = {'ID': str}
    for i in range(epoch_num):
        dtypes[f'Epoch{i}'] = 'int8'  # 缩小数据空间
    df = pd.read_csv(pre_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)], dtype=dtypes)
    df_persis = pd.read_csv(f"{save_dir}kt_persis.csv", header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)], dtype=dtypes)

    df_i = df_sim[df_sim['src'] == flow_id]
    df_i['src'] = "simulation_ps"
    df_i.to_csv(f"{save_dir}result/example.csv", header=False, index=False, mode='a')  # 追加写入模拟值

    kt_rows = []
    for k in ks:
        kt_rows.append([f"kt_ps_k={k}"])  # 初始化每个k值的kt-ps, 每一行对应一个k值
    for epoch in df_persis.columns[1:]:
        f_kte = Counter()  # 本轮满足kt的element个数
        epoch_now = int(epoch[5:])  # 计算对应epoch
        for index, value in enumerate(df_persis[epoch]):
            e = df_persis.loc[index, 'ID']
            f = e.split(">")[0]
            if f != flow_id or df.loc[index, epoch] == 0:  # 如果不是要查找的流 或者 element没有在当前epoch出现
                continue
            for k in ks:
                if value >= k:
                    f_kte[k] += 1
        
        for i, k in enumerate(ks):
            kt_rows[i].append(f_kte[k])
        print(f"Epoch{epoch_now}: kt is done")
    df_i = pd.DataFrame(kt_rows, columns=['kt_ps'] + [f'Epoch{i}' for i in range(epoch_num)])
    df_i.to_csv(f"{save_dir}result/example.csv", header=False, index=False, mode='a')  # 追加写入kt值
    
    example(epoch_num, pre_path, [flow_id], f"{save_dir}result/")  # 追加写入所有element

def persis_comp(epoch_num, gt_persis_path, sim_persis_path, save_path):
    """记录估计值和真实值找到的persistent element, 以及precision等指标"""
    df_gt = pd.read_csv(gt_persis_path, names=['ID', 'occurance'])  # 读取CSV文件
    df_sim = pd.read_csv(sim_persis_path, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])

    df_TP = pd.merge(df_gt, df_sim, on='ID', how='inner', suffixes=('_gt', '_sim'))  # 只要TP情况
    df_TP.to_csv(save_path, header=False, index=False)

    count_TP, count_gt, count_sim = df_TP.shape[0], df_gt.shape[0], df_sim.shape[0]
    count_FP, count_FN = count_sim - count_TP, count_gt - count_TP
    precision, recall = count_TP / (count_TP + count_FP), count_TP / (count_TP + count_FN)
    return count_TP, count_FP, count_FN, precision, recall


if __name__ == "__main__":
    data_type = "fb"  # fb or MAWI
    tau = 0.1 if data_type == "fb" else 0.05  # fb: 0.1 MAWI: 0.05  # ps-sketch衰减系数
    t = 16 if data_type == "fb" else 8  # fb: 16 MAWI: 8  # kt向前查找的周期数
    ks = [6, 8, 10, 12] if data_type == "fb" else [3, 4, 5, 6]  # fb: [6, 8, 10, 12] MAWI: [3, 4, 5, 6]  # kt判断persistent元素的阈值
    thresh_ps = 50 if data_type == "fb" else 200  # fb: 50 MAWI: 200  # ps判断persistent spreader的阈值
    thresh_p_gt = 400  # 用作ground truth, 判断element是否是persistent的
    thresh_p_ps = [6, 8]  # 以ps-sketch定义判断element是否是persistent的
    thresh_p_kt = [6, 8]  # 以kt-persistence定义判断element是否是persistent的

    epoch_len = 60  # fb: 60 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1475305136 if data_type == "fb" else 1681224300.077974000  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1475392025 if data_type == "fb" else 1681225200.150813000  # fb: 1475392025 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    save_dir = "./3.5/FB/"  if data_type == "fb" else "./3.5/MAWI/"
    pre_path = save_dir + "pre_1_40.csv"  # 预处理文件
    gt_path = save_dir + "spread_groundtruth.csv"  # 定义计算的pm和
    sim_path = save_dir + "spread_simulation.csv"  # ps-sketch计算的pm和

    # ---------------------------------experiment 1---------------------------------
    # kt_persis(epoch_num, pre_path, save_dir, t)  # 所有元素的kt-persistence
    # for k in ks:  # 所有元素的kt-persistent spread
    #     if not os.path.exists(save_dir + f"k={k}/"):  # 创建保存当前kt_all的文件夹
    #         os.makedirs(save_dir + f"k={k}/")
    #     kt_all(epoch_num, f"{save_dir}kt_persis.csv", pre_path, f"{save_dir}k={k}/", k)

    # merge_ps_all(epoch_num, gt_path, sim_path, save_dir)  # 将ps-sketch中的模拟和真实值整合在一起, *修改thresh不需要重跑
    # ps_thresh(f"{save_dir}ps_all.csv", save_dir, thresh_ps)  # 截取>thresh的ps-persistent spread

    # if not os.path.exists(save_dir + f"result/"):  # 创建保存结果的文件夹
    #     os.makedirs(save_dir + f"result/")
    # merge_ps_kt(f"{save_dir}ps_thresh.csv", save_dir, ks)  # 求ps和kt的交集差集
    find_diff_flow(sim_path, pre_path, save_dir, "8b9fdc4916c40130", epoch_num, ks)  # 挑出一个flowexample

    # ---------------------------------experiment 2---------------------------------
    # if not os.path.exists(save_dir + f"set/"):  # 创建保存集合结果的文件夹
    #     os.makedirs(save_dir + f"set/")
    # set_gt, set_ps, set_kt = chunk_set_C(thresh_p_gt), chunk_set_A(0, tau), chunk_set_B(0, t)  # set_C代表ground truth, set_A代表ps-sketch, set_B代表kt-persistence
    # set_gt.enumerate(epoch_num, pre_path, f"{save_dir}set/")  # groundtruth
    # file = open(f"{save_dir}result/set_output.txt", "w")  # 输出保存在result文件夹里
    # sys.stdout = file  # 输出直接存入txt文件
    # for thresh in thresh_p_ps:
    #     set_ps.thresh = thresh
    #     set_ps.enumerate(epoch_num, pre_path, f"{save_dir}set/")  # ps-sketch
    #     count_TP, count_FP, count_FN, precision, recall = persis_comp(epoch_num, f"{save_dir}set/set_C_thresh={thresh_p_gt}.csv", f"{save_dir}set/set_A_tau={tau}_thresh={thresh}.csv", f"{save_dir}set/C_{thresh_p_gt}_A_{thresh}.csv")
    #     print(f'threshold = {thresh}, count_TP = {count_TP}, count_FP = {count_FP}, count_FN = {count_FN}, precision = {precision}, recall = {recall}')
    # for thresh in thresh_p_kt:
    #     set_kt.thresh = thresh
    #     set_kt.enumerate(epoch_num, pre_path, f"{save_dir}set/")  # kt-persis
    #     count_TP, count_FP, count_FN, precision, recall = persis_comp(epoch_num, f"{save_dir}set/set_C_thresh={thresh_p_gt}.csv", f"{save_dir}set/set_B_T={t}_thresh={thresh}.csv", f"{save_dir}set/C_{thresh_p_gt}_B_{thresh}.csv")
    #     print(f'k = {thresh}, count_TP = {count_TP}, count_FP = {count_FP}, count_FN = {count_FN}, precision = {precision}, recall = {recall}')
    # file.close()
