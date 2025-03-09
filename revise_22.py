import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import math
import sys
import os


def num_in_t(occur, t, batch_size = 100000):
    """计算t轮内出现次数, 其中occur是所有epoch的出现情况, t是要查找的周期长度, batch_size指定了每次处理的行数，默认值为10000"""
    num_rows, num_cols = occur.shape
    result = np.empty((num_rows, num_cols), dtype=np.uint8)
    for i in range(0, num_rows, batch_size):
        end = min(i + batch_size, num_rows)
        cumsum_batch = np.cumsum(occur[i:end, :], axis=1)  # 对当前批次计算前缀和

        result[i:end, :t] = cumsum_batch[:, :t]
        result[i:end, t:] = cumsum_batch[:, t:] - cumsum_batch[:, :-t]
    
    return result
 
def kt_persis(epoch_num, pre_path, save_dir, t, batch_size = 100000):
    """计算每轮每个element对应的kt-persistence, 不足t的情况直接累加, 形状与pre相同"""
    dtypes = {'ID': str}
    for i in range(epoch_num):
        dtypes[f'Epoch{i}'] = 'int8'  # 缩小数据空间
    df = pd.read_csv(pre_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)], dtype=dtypes)
    ids = df['ID'].values
    data = df.drop(columns=['ID']).values
    
    window_sums = num_in_t(data, t)  # 计算滑动窗口和
    num_rows = ids.shape[0]
    num_cols = window_sums.shape[1]

    with open(f"{save_dir}kt_persis.csv", 'w', newline='') as f:  # 打开文件准备写入
        for i in range(0, num_rows, batch_size):  # 分批处理数据
            end = min(i + batch_size, num_rows)
            batch_df = pd.DataFrame(np.hstack((ids[i:end].reshape(-1, 1), window_sums[i:end, :])),\
                                    columns=['ID'] + [f'Epoch{i}' for i in range(num_cols)])  # 创建当前批次的 DataFrame
            batch_df.to_csv(f, header=False, index=False, mode='a')

def kt_all(epoch_num, persis_path, pre_path, save_dir, k):
    """计算每轮每个flow对应的spread, kt-persistent spread, 以及平均kt-persistence"""
    dtypes = {'ID': str}
    for i in range(epoch_num):
        dtypes[f'Epoch{i}'] = 'int8'  # 缩小数据空间
    df_presis = pd.read_csv(persis_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)], dtype=dtypes)
    df = pd.read_csv(pre_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)], dtype=dtypes)
    s_rows = []
    for epoch in df_presis.columns[1:]:
        f_e = defaultdict(list)  # 本轮所有出现过的element_id
        f_kte = Counter()  # 本轮满足kt的element个数
        epoch_now = int(epoch[5:])  # 计算对应epoch
        for index, value in enumerate(df_presis[epoch]):
            e = df_presis.loc[index, 'ID']
            f = e.split(">")[0]
            if df.loc[index, epoch] == 0:  # 如果在预处理中不为0，说明element在当前epoch出现
                continue
            f_e[f].append(value)
            if value >= k:
                f_kte[f] += 1
        
        for f, e_list in f_e.items():
            s_rows.append([epoch_now, f, len(e_list), f_kte[f], sum(e_list) / len(e_list)])  # 更新到列表
        print(f"Epoch{epoch_now}: kt is done")
    
    s_df = pd.DataFrame(s_rows, columns=['Epoch', 'flow_id', 'spread', 'kt_ps', 'mean_kt_p'])
    s_df.to_csv(f"{save_dir}kt_all.csv", header=False, index=False)

def kt_thresh(kt_all_path, save_dir, thresh):
    """从kt_ps_all中截取>=thresh的部分"""
    df = pd.read_csv(kt_all_path, header=None, names=['Epoch', 'flow_id', 'spread', 'kt_ps', 'mean_kt_p'])
    k_df = df[df['kt_ps'] > thresh]
    k_df.to_csv(f"{save_dir}kt_thresh.csv", header=False, index=False)

def merge_ps_all(epoch_num, gt_path, sim_path, save_dir):
    """记录每个epoch下flow的真实值和对应估计值"""
    df_gt = pd.read_csv(gt_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])  # 读取CSV文件
    df_sim = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
    # align(epoch_num, gt_path, sim_path, suffix(sim_path, "_aligned"))  # 将sim对齐到gt

    for i in range(epoch_num):  # 逐列遍历
        sim_i = df_sim[['src', f'Epoch{i}']]
        gt_i = df_gt[['src', f'Epoch{i}']]
        df_i = pd.merge(sim_i, gt_i, on='src', how='left', suffixes=('_sim', '_gt'))  # 根据sim连接gt
        df_i['Epoch'] = i
        df_i = df_i[['Epoch', 'src', f'Epoch{i}_sim', f'Epoch{i}_gt']]  # 添加epoch列并重排顺序
        df_i.to_csv(f'{save_dir}ps_all.csv', mode='a', index=False, header=False)

def ps_thresh(ps_all_path, save_dir, thresh):
    """从kt_ps_all中截取>=thresh的部分"""
    df = pd.read_csv(ps_all_path, header=None, names=['Epoch', 'flow_id', 'ps_sim', 'ps_gt'])
    k_df = df[df['ps_sim'] > thresh]
    k_df.to_csv(f"{save_dir}ps_thresh.csv", header=False, index=False)

def merge_ps_kt(kt_all_path, kt_thresh_path, ps_all_path, ps_thresh_path, save_dir):
    """计算不同方法的交集差集"""
    left_2 = ['Epoch', 'flow_id', 'ps_sim', 'ps_gt']
    right_3 = ['Epoch', 'flow_id', 'spread', 'kt_ps', 'mean_kt_p']

    df_kt = pd.read_csv(kt_all_path, header=None, names=right_3)
    df_kt_thresh = pd.read_csv(kt_thresh_path, header=None, names=right_3)
    df_ps = pd.read_csv(ps_all_path, header=None, names=left_2)
    df_ps_thresh = pd.read_csv(ps_thresh_path, header=None, names=left_2)

    df_ktbase = pd.merge(df_ps, df_kt_thresh, on=['Epoch', 'flow_id'], how='right')
    df_ktbase.to_csv(f'{save_dir}ktbase.csv', index=False, header=False)  # 第一张表

    df_psbase = pd.merge(df_ps_thresh, df_kt, on=['Epoch', 'flow_id'], how='left')
    df_psbase.to_csv(f'{save_dir}psbase.csv', index=False, header=False)  # 第二张表

    kt_keys = df_kt_thresh[['Epoch', 'flow_id']].drop_duplicates()  # 提取kt_thresh中的键
    mask = ~df_psbase[['Epoch', 'flow_id']].apply(tuple, axis=1).isin(kt_keys.apply(tuple, axis=1))
    only_in_psbase = df_psbase[mask]  # 掩码筛选不在kt_thresh的行
    only_in_psbase.to_csv(f'{save_dir}only_in_psbase.csv', index=False, header=False)  # 第三张表

    ps_keys = df_ps_thresh[['Epoch', 'flow_id']].drop_duplicates()  # 提取ps_thresh中的键
    mask = ~df_ktbase[['Epoch', 'flow_id']].apply(tuple, axis=1).isin(ps_keys.apply(tuple, axis=1))
    only_in_ktbase = df_ktbase[mask]  # 掩码筛选不在ps_thresh的行
    only_in_ktbase.to_csv(f'{save_dir}only_in_ktbase.csv', index=False, header=False)  # 第四张表

    print("num in kt = ", df_ktbase.shape[0], ", numb in ps = ", df_psbase.shape[0])  # 所有epoch累加
    print("num in both = ", df_ktbase.shape[0] - only_in_ktbase.shape[0],\
          ", only in ps = ", only_in_psbase.shape[0], ", only in kt = ", only_in_ktbase.shape[0])

def example(epoch_num, pre_path, flow_ids, save_dir):
    """根据flow_id列表挑出所有历史element的出现情况"""
    dtypes = {'ID': str}
    for i in range(epoch_num):
        dtypes[f'Epoch{i}'] = 'int8'  # 缩小数据空间
    df = pd.read_csv(pre_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)], dtype=dtypes)
    for flow_id in flow_ids:
        ex_df = df[df['ID'].str.contains(flow_id, case=False)]
        ex_df.to_csv(f"{save_dir}example.csv", header=False, index=False, mode='a')  # 追加写入当前流的信息



if __name__ == "__main__":
    t = 16  # kt向前查找的周期数
    k = 8  # kt判断persistent元素的阈值
    thresh_kt = 10  # kt判断persistent spreader的阈值
    thresh_ps = 50  # ps判断persistent spreader的阈值

    epoch_len = 60  # fb: 60 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1475305136  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1475392025  # fb: 1475392025 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    save_dir = "./2.22/FB/pre_1_40/"  # fb: "./7.23/ca_1/" MAWI: "./7.23/2345/"
    gt_path = save_dir + "spread_groundtruth.csv"  # 定义计算的pm和
    sim_path = save_dir + "spread_simulation.csv"  # ps-sketch计算的pm和

    # if not os.path.exists(save_dir + f"t={t}_k={k}/"):  # 创建保存当前thresh_kt的文件夹
    #     os.makedirs(save_dir + f"t={t}_k={k}/")
    # kt_persis(epoch_num, f"{save_dir}pre_3_40.csv", f"{save_dir}t={t}_k={k}/", t)  # 所有元素的kt-persistence
    # kt_all(epoch_num, f"{save_dir}t={t}_k={k}/kt_persis.csv", f"{save_dir}pre_1_40.csv", f"{save_dir}t={t}_k={k}/", k)  # 所有元素的kt-persistent spread
    # kt_thresh(f"{save_dir}t={t}_k={k}/kt_all.csv", f"{save_dir}t={t}_k={k}/", thresh_kt)  # 截取>=thresh的kt-persistent spread

    # if not os.path.exists(save_dir + f"tau={thresh_ps}/"):  # 创建保存当前thresh_ps的文件夹
    #     os.makedirs(save_dir + f"tau={thresh_ps}/")
    # merge_ps_all(epoch_num, gt_path, sim_path, f"{save_dir}tau={thresh_ps}/")  # 将ps-sketch中的模拟和真实值整合在一起, *修改thresh不需要重跑
    # ps_thresh(f"{save_dir}tau={thresh_ps}/ps_all.csv", f"{save_dir}tau={thresh_ps}/", thresh_ps)  # 截取>thresh的ps-persistent spread

    if not os.path.exists(save_dir + f"result/"):  # 创建保存结果的文件夹
        os.makedirs(save_dir + f"result/")
    # file = open(f"{save_dir}result/output.txt", "w")
    # sys.stdout = file  # 输出直接存入txt文件
    # print(f"t = {t}, k = {k}, thresh_kt = {thresh_kt}, thresh_ps = {thresh_ps}")
    # merge_ps_kt(f"{save_dir}t={t}_k={k}/kt_all.csv", f"{save_dir}t={t}_k={k}/kt_thresh.csv",\
    #             f"{save_dir}tau={thresh_ps}/ps_all.csv", f"{save_dir}tau={thresh_ps}/ps_thresh.csv",\
    #                 f"{save_dir}result/")  # 求ps和kt的交集差集
    # file.close()

    example(epoch_num, f"{save_dir}pre_1_40.csv", ["f4f5db99cd0b4519", "90cbe97ccb5ff53e",\
            "ba6b20a796ad4eef", "f8edbce27f25603c", "73ca04cc77be6c31"], f"{save_dir}result/")  # 手动选择flow_id并提取对应element情况
    # only in kt:["d30fbe669f2acfa9", "277cfd3e55f6a5da", "86c43594c9deb479", "baf52ae13550b284", "356598f6478b0bb7"], 出现次数为[976, 827, 138, 105, 55]
    # only in ps:["f4f5db99cd0b4519", "90cbe97ccb5ff53e", "ba6b20a796ad4eef", "f8edbce27f25603c", "73ca04cc77be6c31"], 出现次数[1156, 1096, 1058, 855, 728]
