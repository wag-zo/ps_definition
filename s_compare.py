import pandas as pd
import numpy as np
import math
from utils import topk, suffix



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
        gt_non0 = df_gt[df_gt[f'Epoch{i}'] != 0]
        sim_non0 = df_sim[df_sim[f'Epoch{i}'] != 0]

        gt_sim = df_ap['src'].isin(pd.merge(gt_non0, sim_non0, on='src')['src'])
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
        src_set = set(ap_1['src'])  
        filtered_df_gt = df_gt[df_gt['src'].isin(src_set)]  # 只保留在 ap_1['src'] 中的 gt 和 sim 数据
        filtered_df_sim = df_sim[df_sim['src'].isin(src_set)]
        gt = pd.merge(ap_1[['src']], filtered_df_gt, on='src', how='left')[f'Epoch{i}'].values
        sim = pd.merge(ap_1[['src']], filtered_df_sim, on='src', how='left')[f'Epoch{i}'].values

        rmse.append(np.sqrt(((sim - gt) ** 2).mean()))
        mre.append(np.sum(np.abs((sim - gt) / gt)))
    
    return num_sub, rmse, mre

def copmute_prf(epoch_num, k_percent, gt_path, sim_path):
    """计算precision，recall和F1分数"""
    topk_gt_path = suffix(gt_path, "_topk")
    topk(gt_path, k_percent, topk_gt_path, epoch_num)  # 取k_percent%行

    df_gt = pd.read_csv(topk_gt_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
    df_sim = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
    df_ins = pd.merge(df_gt, df_sim, on='src')  # 根据id列合并，生成交集intersection

    count_ins, count_gt, count_sim = len(df_ins), len(df_gt), len(df_sim)
    precision = count_ins / count_sim
    recall = count_ins / count_gt
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score



if __name__ == "__main__":
    epoch_len = 300  # fb: 300 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1475305136  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1475319422  # fb: 1475319422 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    k_percent = 0.3  # 截取多少行计算prf

    save_dir = "./7.23/test_0.05/"
    gt_path = save_dir + "spread_groundtruth.csv"  # 定义计算的pm和
    ps_path = save_dir + "simulation_src2ps_aligned.csv"  # 从p-sketch发送包计算的pm和
    sim_path = save_dir + "spread_simulation.csv"  # ps-sketch计算的pm和
    src2es_path = save_dir + "simulation_src2es.csv"  # 从p-sketch发送包计算的element个数
    hll2es_path = save_dir + "simulation_hll2es.csv"  # ps-sketch计算的element个数

    appear(epoch_num, src2es_path, hll2es_path, save_dir)  # 判断src是否出现且被检测到，1=出现且被检测到，0=出现且没被检测到，-1=没出现
    num_sub, rmse, mre = compute_me(epoch_num, save_dir + "appear.csv", src2es_path, hll2es_path)
    print(f"num_sub = {num_sub}\nrmse = ", \
          [float(f"{r:.4f}") for r in rmse], "\nmre = ", \
          [float(f"{m:.4f}") for m in mre])
    
    # precision, recall, f1_score = copmute_prf(epoch_num, k_percent, gt_path, sim_path)
    # print("precision = {:.4%}, recall = {:.4%}, f1_score = {:.4%}".format(precision, recall, f1_score))


