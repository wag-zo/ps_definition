import pandas as pd
import numpy as np
from scipy import stats
import math


def mre(epoch_num, sim_path, gt_path, save_dir):
    """计算mre"""
    df_gt = pd.read_csv(gt_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])  # 读取CSV文件
    df_sim = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
    df_merge = pd.merge(df_sim, df_gt, how="left", on='src', suffixes=('_sim', '_gt'))

    df_new = df_merge[["src"]]  # 存储所有epoch对应mre
    for col in df_merge.columns:
        if col.endswith('_sim'):  # 找到sim列和对应gt列
            sim_col = col
            gt_col = col.replace('_sim', '_gt')
        else:  # 跳过非sim列
            continue
        
        # 计算errors，处理异常值，存入df_new
        errors = ((df_merge[sim_col] - df_merge[gt_col]).abs() / df_merge[gt_col]).rename(col.split("_")[0])
        errors.replace([np.inf, -np.inf], np.nan, inplace=True)
        errors.fillna(value=0, inplace=True)
        df_new = pd.concat([df_new, errors], axis=1)
    
    df_new.to_csv(f'{save_dir}mre.csv', index=False)

def corr(epoch_num, sim_path, gt_path, save_dir):
    """计算相关系数"""
    df_gt = pd.read_csv(gt_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])  # 读取CSV文件
    df_sim = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
    df_merge = pd.merge(df_sim, df_gt, how="left", on='src', suffixes=('_sim', '_gt'))
    
    pccs = []
    sim_col, gt_col = [f'Epoch{i}_sim' for i in range(epoch_num)], [f'Epoch{i}_gt' for i in range(epoch_num)]
    for _, row in df_merge.iterrows():
        x, y = row[sim_col].astype('float16').values, row[gt_col].astype('float16').values
        pcc = np.corrcoef(x, y)
        pccs.append([row[0], pcc[0][1]])
    
    df_new = pd.DataFrame(pccs, columns=['src', 'corr'])
    df_new.to_csv(f'{save_dir}corr.csv', index=False)


def ci(epoch_num, c_level, sim_path, save_dir):
    """计算置信区间"""
    df = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])  # 读取CSV文件

    intervals = []
    for _, row in df.iterrows():
        low, high = stats.t.interval(c_level, df=len(row[1:])-1,
                loc=np.mean(row[1:]),
                scale=stats.sem(row[1:]))
        intervals.append([row[0], low, high])

    df_new = pd.DataFrame(intervals, columns=['src', 'low', 'high'])
    df_new.to_csv(f'{save_dir}ci.csv', index=False)
 
    
    # df_mres = df.columns[1:]
    # means, sems = df[df_mres].mean(), df[df_mres].sem()  # 计算每列的均值和标准误差
    # df_ci = pd.DataFrame( # 计算置信区间
    #     stats.t.interval(c_level, df=len(df)-1, loc=means, scale=sems),
    #     index=df_mres,
    #     columns=[f'Epoch{i}' for i in range(epoch_num)]
    # ).T

    # df_new = pd.concat([df[['src']], df_ci], axis=1)  # 保存到CSV
    # df_new.to_csv(f'{save_dir}ci.csv', index=False)



if __name__ == "__main__":
    epoch_len = 60  # fb: 300 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1681224300.077974000  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1681225200.150813000  # fb: 1475319422 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量
    thresh = 200  # 截取多少计算prf

    save_dir = "./8.28/"  # fb: ./8.22/FB/ MAWI: ./8.22/MAWI/
    gt_path = save_dir + "spread_groundtruth.csv"  # 定义计算的pm和
    sim_path = save_dir + "spread_simulation.csv"  # ps-sketch计算的pm和

    # mre(epoch_num, sim_path, gt_path, save_dir)
    # corr(epoch_num, sim_path, gt_path, save_dir)
    ci(epoch_num, 0.9, sim_path, save_dir)

    # data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9], 'D': [10, 11, 12]}  
    # df = pd.DataFrame(data)  
    # col_1, col_2 = ['A', 'B'], ['C', 'D']
    
    # # 使用iterrows()迭代DataFrame  
    # for index, row in df.iterrows():  
    #     # 注意：这里row['A']和row['B']已经是单个值了，再次使用.values会得到NumPy数组  
    #     print(type(row[col_1].values), type(row[col_2].values))
    #     print(row[col_1].values.shape, row[col_2].values.shape)
    #     print(np.corrcoef(row[col_1].values, row[col_2].values))
