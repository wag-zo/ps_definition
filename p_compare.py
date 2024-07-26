import pandas as pd
import math
import os
from collections import Counter


def intersection(epoch_num, c_path, right_path, save_path):
    """计算交集"""
    # 读取CSV文件并生成detect
    df_C = pd.read_csv(c_path, header=None, names=['ID', 'occur'])
    df = pd.read_csv(right_path, header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
    df['detect'] = df.iloc[:, 1:].astype(int).sum(axis=1)  # 对第一列后的元素求和，计算sum
    df_Right = df.loc[:, ['ID', 'detect']]
    
    # 使用pandas，根据id列合并DataFrame并存储
    df_ins = pd.merge(df_C, df_Right, on='ID')

    # 使用set.intersection，根据id列合并DataFrame并存储
    # set_C = set(df_C['ID'])
    # set_Right = set(df_Right['ID'])
    # ins = set_C.intersection(set_Right)
    # filtered_df_C = df_C[df_C['ID'].isin(ins)]
    # filtered_df_Right = df_Right[df_Right['ID'].isin(ins)]
    # df_ins = pd.merge(filtered_df_C, filtered_df_Right, on=df_C.columns[0])

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

def subtract(ac_path, bc_path, save_path):
    """计算差集"""
    df_A = pd.read_csv(ac_path, header=None, names=['ID', 'occur', 'detect'])
    df_B = pd.read_csv(bc_path, header=None, names=['ID', 'occur', 'detect'])
    
    # 找出在A∩C中但不在B∩C中的id并输出  
    df_subtract = df_A.loc[~df_A['ID'].isin(df_B['ID']), ['ID', 'occur']]
    if df_subtract.empty:
        return 0, 0
    max_occur = df_subtract['occur'].max() # 获取occur列的最大值
    max_occur_ids = df_subtract[df_subtract['occur'] == max_occur]['ID']  # 获取对应的ID
    max_occur_ids.to_csv(save_path, index=False, header=False)
    return len(max_occur_ids), max_occur

def max_ids(epoch_num, save_dir):
    # 初始化一个空的Counter对象来存储所有CSV文件中元素的计数
    counts = Counter()
    
    # 遍历指定文件夹下的所有文件
    for file in os.listdir(save_dir + "/subtract/"):
        if file.endswith('.csv') and file != "max_counts.csv":
            file_path = os.path.join(save_dir + "/subtract/", file)
            df = pd.read_csv(file_path, header=None)
            elements = df[0].tolist()
            counts.update(elements)
    
    # 将Counter对象转换为字典，然后按照值（即出现次数）进行排序
    sorted_counts = counts.most_common()

    df_sub = pd.DataFrame(sorted_counts, columns=['ID', 'Count'])
    print("find all")
    df_pre = pd.read_csv(save_dir + "pre.csv", header=None, names=['ID'] + [f'Epoch{i}' for i in range(epoch_num)])
    df_result = pd.merge(df_sub, df_pre, on='ID')  # 找到元素在不同epoch的出现情况
    print("merge all")
    df_result.to_csv(save_dir + "/subtract/max_counts.csv", index=False)


if __name__ == "__main__":
    threshA = 8
    tau = 0.1  # 衰减因子
    threshB = 8  # <= T
    T = 8  # 往前看的epoch数量
    threshC = 32
    epoch_len = 300  # 1个epoch的时间范围
    start_time = 1475305136
    end_time = 1475319422
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    # print("epoch_num = ", epoch_num)

    save_dir = "./7.18/facebook/"
    c_path = f"{save_dir}set_C_thresh={threshC}.csv"
    a_path = f"{save_dir}set_A_tau={tau}_thresh={threshA}.csv"
    b_path = f"{save_dir}set_B_T={T}_thresh={threshB}.csv"
    ac_path = f"{save_dir}/intersection/A_thresh={threshA}_C_thresh={threshC}.csv"
    bc_path = f"{save_dir}/intersection/B_thresh={threshB}_C_thresh={threshC}.csv"
    max_path = f"{save_dir}/subtract/A_thresh={threshA}_B_thresh={threshB}_C_thresh={threshC}.csv"

    ins = "B&C"
    if ins == "A&C":
        right_path = a_path
        save_path = ac_path
    else:
        right_path = b_path
        save_path = bc_path
    
    # if not os.path.exists(save_dir + "/intersection/"):  # 创建保存intersection的文件夹
    #     os.makedirs(save_dir + "/intersection/")
    # intersection(epoch_num, c_path, right_path, save_path)

    # precision, recall, f1_score = copmute_prf(c_path, right_path, save_path)
    # print("precision = {:.4%}, recall = {:.4%}, f1_score = {:.4%}".format(precision, recall, f1_score))

    # if not os.path.exists(save_dir + "/subtract/"):  # 创建保存subtract的文件夹
    #     os.makedirs(save_dir + "/subtract/")
    # max_occur_ids_num, max_occur = subtract(ac_path, bc_path, max_path)
    # print("max_occur = {}, max_occur_ids_num = {}".format(max_occur,max_occur_ids_num))

    max_ids(epoch_num, save_dir)
    print("saved")

