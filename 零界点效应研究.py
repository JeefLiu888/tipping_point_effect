import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob

# 设置你的 snapshot 路径
SNAPSHOT_DIR = r""

# 获取所有快照文件并按时间排序
snapshot_files = sorted(glob(os.path.join(SNAPSHOT_DIR, "*.csv")))

# 构建 address -> 持仓数量 的时间序列
holder_history = {}

for file in snapshot_files:
    date = os.path.basename(file).split("_")[-1].replace(".csv", "")
    df = pd.read_csv(file)
    for addr, count in df['holder'].value_counts().items():
        if addr not in holder_history:
            holder_history[addr] = {}
        holder_history[addr][date] = count

# 构建 DataFrame：每个地址每个时间点的持仓
snapshot_dates = [os.path.basename(f).split("_")[-1].replace(".csv", "") for f in snapshot_files]
holder_df = pd.DataFrame.from_dict(holder_history, orient='index').fillna(0).astype(int)
holder_df = holder_df[snapshot_dates]  # 确保列顺序正确

# 构建训练数据：从 t 时刻的持仓预测 t+1 时刻是否新增购买
records = []

for i in range(len(snapshot_dates) - 1):
    t0 = snapshot_dates[i]
    t1 = snapshot_dates[i + 1]

    h0 = holder_df[t0]
    h1 = holder_df[t1]

    for addr in holder_df.index:
        before = h0[addr]
        after = h1[addr]
        if not np.isnan(before) and not np.isnan(after):
            bought_more = 1 if after > before else 0
            records.append({
                'address': addr,
                'holdings': before,
                'bought_more': bought_more
            })

df_analyze = pd.DataFrame(records)

# 仅考虑 holdings <= 5（如论文只研究 0/1/2）
df_analyze = df_analyze[df_analyze['holdings'] <= 10]

# Logistic 回归分析（更新版，确保类型正确）
X = pd.get_dummies(df_analyze['holdings'], prefix='h', drop_first=True)
X = sm.add_constant(X)
y = df_analyze['bought_more']

# 👇 类型转换，避免 object 错误
X = X.astype(float)
y = y.astype(float)

# 拟合模型
model = sm.Logit(y, X).fit()
print(model.summary())



plot_data = df_analyze.groupby('holdings')['bought_more'].mean().reset_index()

sns.lineplot(data=plot_data, x='holdings', y='bought_more', marker='o', color='steelblue')
plt.xlabel("Holdings at t")
plt.ylabel("Probability of Buying More at t+1")
plt.title("Collection Tipping Point in NFT Snapshots")
plt.ylim(0, 0.04)
plt.grid(True)
plt.show()
