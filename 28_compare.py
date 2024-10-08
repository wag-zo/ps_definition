import pandas as pd
import numpy as np
from scipy import stats
import math


# def mre(epoch_num, sim_path, gt_path, save_dir):
#     """计算mre"""
#     df_gt = pd.read_csv(gt_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])  # 读取CSV文件
#     df_sim = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
#     df_merge = pd.merge(df_sim, df_gt, how="left", on='src', suffixes=('_sim', '_gt'))  # 

#     df_new = df_merge[["src"]]  # 存储所有epoch对应mre
#     for col in [f'Epoch{i}' for i in range(epoch_num)]:
#         sim_col, gt_col = df_merge[f'{col}_sim'], df_merge[f'{col}_gt']
#         errors = ((sim_col - gt_col).abs() / gt_col).rename(col.split("_")[0])
#         errors.replace([np.inf, -np.inf], np.nan, inplace=True)  # 处理异常值
#         errors.fillna(value=0, inplace=True)
#         df_new = pd.concat([df_new, errors], axis=1)
    
#     df_new['mre'] = df_new.iloc[:, 1:].mean(axis=1)
#     df_new.to_csv(f'{save_dir}mre.csv', index=False)
#     mean_meric = df_new['mre'].values  # 取src后所有列
#     return mean_meric

# def ci(epoch_num, confidence, sim_path, save_dir):
#     """计算置信区间"""
#     df = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])  # 读取CSV文件

#     data = df.iloc[:, 1:].to_numpy()  # src列外所有列
#     means = np.mean(data, axis=1)  # 样本均值
#     sems = stats.sem(data, axis=1, ddof=1)  # 样本标准差
#     t_score = stats.t.ppf((1 + confidence) / 2, len(data[0]) - 1)
#     lows, highs = means - t_score * sems, means + t_score * sems  # 计算置信区间
    
#     df_new = pd.DataFrame({'src': df['src'], 'low': lows, 'high': highs})  # 保存结果
#     df_new['width'] = df_new['high'] - df_new['low']  # 计算宽度
#     df_new.to_csv(f'{save_dir}ci.csv', index=False)
#     mean_meric = [np.mean(lows), np.mean(highs)]  # 取宽度
#     return mean_meric

# def corr(epoch_num, sim_path, gt_path, save_dir):
#     """计算相关系数"""
#     df_gt = pd.read_csv(gt_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])  # 读取CSV文件
#     df_sim = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])
#     df_merge = pd.merge(df_sim, df_gt, how="left", on='src', suffixes=('_sim', '_gt'))

#     sim_cols, gt_cols = [f'Epoch{i}_sim' for i in range(epoch_num)], [f'Epoch{i}_gt' for i in range(epoch_num)]
#     x, y = df_merge[sim_cols], df_merge[gt_cols]
#     x.columns, y.columns = x.columns.str.replace('_sim', ''), y.columns.str.replace('_gt', '')  # 匹配列名
#     pccs = x.corrwith(y, axis=1).rename('corr')  # 按行计算相关系数
#     pccs.replace([np.inf, -np.inf], np.nan, inplace=True)  # 处理异常值
#     pccs.fillna(value=0, inplace=True)
    
#     df_new = pd.concat([df_merge['src'], pccs], axis=1)
#     df_new.to_csv(f'{save_dir}corr.csv', index=False)  # 保存结果
#     mean_meric = pccs.mean()  # 取均值
#     return mean_meric


def corrected(epoch_num, sim_path, gt_path, save_dir, confidence):
    """存储+mre+ci+corr"""
    df_gt = pd.read_csv(gt_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])  # 读取CSV文件
    df_sim = pd.read_csv(sim_path, header=None, names=['src'] + [f'Epoch{i}' for i in range(epoch_num)])

    df_new = pd.DataFrame()
    common_columns = ['src', 'ps']
    for i in range(epoch_num):  # 逐列遍历
        # sim_non0 = df_sim[df_sim[f'Epoch{i}'] != 0][['src', f'Epoch{i}']]
        # gt_i = df_gt[['src', f'Epoch{i}']]
        gt_non0 = df_gt[df_gt[f'Epoch{i}'] != 0][['src', f'Epoch{i}']]  # gt非零行
        sim_i = df_sim[['src', f'Epoch{i}']]
        gt_non0.columns = common_columns  # 对齐列名
        sim_i.columns = common_columns
        df_i = pd.merge(sim_i, gt_non0, on='src', how='inner', suffixes=('_sim', '_gt'))  # 根据gt非零行连接sim
        df_new = pd.concat([df_new, df_i], ignore_index=True, axis=0)
    
    sim_col, gt_col = df_new['ps_sim'], df_new['ps_gt']
    mres = (sim_col - gt_col).abs() / gt_col
    df_new['mre'] = mres
    df_new.to_csv(f'{save_dir}mres.csv', index=False)

    data = mres.to_numpy()  # mres
    means = np.mean(data)  # 样本均值
    sems = stats.sem(data, ddof=1)  # 样本标准差
    t_score = stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    lows, highs = means - t_score * sems, means + t_score * sems  # 计算置信区间

    corr = sim_col.corr(gt_col)

    return mres.mean(), lows, highs, corr


if __name__ == "__main__":
    epoch_len = 300  # fb: 300 MAWI: 60  # 1个epoch的时间范围/second
    start_time = 1475305136  # fb: 1475305136 MAWI: 1681224300.077974000
    end_time = 1475319422  # fb: 1475319422 MAWI: 1681225200.150813000
    epoch_num = math.ceil((end_time - start_time) / epoch_len)  # epoch的数量

    save_dir = "./8.28/具体结果/FB/"  # fb: ./8.22/FB/ MAWI: ./8.22/MAWI/
    gt_path = save_dir + "spread_groundtruth.csv"  # 定义计算的pm和
    sim_path = save_dir + "spread_simulation.csv"  # ps-sketch计算的pm和

    mre, low, high, corr = corrected(epoch_num, sim_path, gt_path, save_dir, 0.9)
    print(f"mre = {mre}, 90% confidence = {low}, {high}, corr = {corr}")

    # mres = mre(epoch_num, sim_path, gt_path, save_dir)
    # confidences = ci(epoch_num, 0.9, sim_path, save_dir)
    # corrs = corr(epoch_num, sim_path, gt_path, save_dir)

    # print(f"mre = {mres.mean()}, 90% confidence = {confidences}, corr = {corrs}")

