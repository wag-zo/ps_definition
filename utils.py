import pandas as pd
import math
import os


def intersection(epoch_num, c_path, right_path, save_path):
    """计算交集"""
    # 读取CSV文件并生成detect
    df_C = pd.read_csv(c_path, header=None, names=['ID', 'occur'])
    df = pd.read_csv(right_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
    df['detect'] = df.iloc[:, 1:].astype(int).sum(axis=1)  # 对第一列后的元素求和，计算sum
    df_Right = df.loc[:, ['ID', 'detect']]
    
    # # 根据id列合并DataFrame并存储
    # df_ins = pd.merge(df_C, df_Right, on='ID')
    set_C = set(df_C['ID'])
    set_Right = set(df_Right['ID'])
    ins = set_C.intersection(set_Right)
    filtered_df_C = df_C[df_C['ID'].isin(ins)]
    filtered_df_Right = df_Right[df_Right['ID'].isin(ins)]
    df_ins = pd.merge(filtered_df_C, filtered_df_Right, on=df_C.columns[0])

    df_ins.to_csv(save_path, index=False, header=False)

def copmute_prf(c_path, right_path, ins_path):
    """计算precision，recall和F1分数"""
    count_c = 0
    with open(c_path, 'r', encoding='utf-8') as file:
        for _ in file:
            count_c += 1
    
    count_r = 0
    with open(right_path, 'r', encoding='utf-8') as file:
        for _ in file:
            count_r += 1

    count_ins = 0
    with open(ins_path, 'r', encoding='utf-8') as file:
        for _ in file:
            count_ins += 1
    
    precision = count_ins / count_r
    recall = count_ins / count_c
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score

def subtract(ac_path, bc_path):
    """计算差集"""
    df_A = pd.read_csv(ac_path, header=None, names=['ID', 'occur', 'detect'])
    df_B = pd.read_csv(bc_path, header=None, names=['ID', 'occur', 'detect'])
    
    # 找出在A中但不在B中的出现次数并输出
    df_subtract = df_A.loc[~df_A['ID'].isin(df_B['ID']), 'occur']
    max_occur = df_subtract.max() if not df_subtract.empty else 0
    return max_occur




if __name__ == "__main__":
    threshA = 3
    tau = 0.1  # 衰减因子
    threshB = 3  # <= T
    T = 8  # 往前看的epoch数量
    threshC = 8
    epoch_len = 60  # 1个epoch的时间范围
    start_time = 1681224300.077974000
    end_time = 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    # print("epoch_num = ", epoch_num)

    save_dir = "./7.16/202304112345/"
    if not os.path.exists(save_dir + "/intersection/"):  # 创建保存intersection的文件夹
        os.makedirs(save_dir + "/intersection/")
    c_path = f"{save_dir}set_C_thresh={threshC}.csv"
    a_path = f"{save_dir}set_A_tau={tau}_thresh={threshA}.csv"
    b_path = f"{save_dir}set_B_T={T}_thresh={threshB}.csv"
    ac_path = f"{save_dir}/intersection/A_thresh={threshA}_C_thresh={threshC}.csv"
    bc_path = f"{save_dir}/intersection/B_thresh={threshB}_C_thresh={threshC}.csv"

    ins = "A&C"
    if ins == "A&C":
        right_path = a_path
        save_path = ac_path
    else:
        right_path = b_path
        save_path = bc_path
    
    intersection(epoch_num, c_path, right_path, save_path)

    # precision, recall, f1_score = copmute_prf(c_path, right_path, save_path)
    # print("precision = {:.4%}, recall = {:.4%}, f1_score = {:.4%}".format(precision, recall, f1_score))

    # max_occur = subtract(ac_path, bc_path)
    # print("max_occur = {}".format(max_occur))



