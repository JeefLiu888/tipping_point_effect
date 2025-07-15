import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.ticker import PercentFormatter

# 设置你的 snapshot 路径
SNAPSHOT_DIR = r"C:\Users\ljfgk\Desktop\BAYC_Snapshots"

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
                'bought_more': bought_more,
                'adoption': 1 if before == 0 and after > 0 else 0,  # 初次购买标记
                'collection': 1 if before > 0 and after > before else 0  # 收藏行为标记
            })

df_analyze = pd.DataFrame(records)

# ------------------------------------------------
# 分析 1: 将数据分为"初次购买"和"收藏行为"两部分
# ------------------------------------------------

# 总体情况
print("==== 总体购买行为分析 ====")
total_records = len(df_analyze)
new_adoptions = df_analyze['adoption'].sum()
collection_behaviors = df_analyze['collection'].sum()

print(f"总记录数: {total_records}")
print(f"初次购买次数: {new_adoptions} ({new_adoptions / total_records:.2%})")
print(f"收藏行为次数: {collection_behaviors} ({collection_behaviors / total_records:.2%})")
print("\n")

# ------------------------------------------------
# 分析 2: 仅考虑收藏行为（已持有至少1个NFT的地址）
# ------------------------------------------------
print("==== 收藏行为分析（已持有NFT的地址）====")

# 筛选出已经持有NFT的地址记录
collection_df = df_analyze[df_analyze['holdings'] > 0].copy()
collection_df = collection_df[collection_df['holdings'] <= 5]  # 限制在5个以内分析

# 对收藏行为进行逻辑回归分析
X_collection = pd.get_dummies(collection_df['holdings'], prefix='h', drop_first=False)
X_collection = sm.add_constant(X_collection)
y_collection = collection_df['bought_more']

# 类型转换
X_collection = X_collection.astype(float)
y_collection = y_collection.astype(float)

# 拟合模型
collection_model = sm.Logit(y_collection, X_collection).fit()
print(collection_model.summary())

# 可视化收藏行为的概率与持有量的关系
collection_plot_data = collection_df.groupby('holdings')['bought_more'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=collection_plot_data, x='holdings', y='bought_more', marker='o', color='steelblue', linewidth=2)
plt.xlabel("持有数量 (t 时刻)", fontsize=12)
plt.ylabel("购买更多的概率 (t+1 时刻)", fontsize=12)
plt.title("NFT 收藏行为的临界点效应（仅已持有NFT的地址）", fontsize=14)
plt.grid(True, alpha=0.3)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.tight_layout()
plt.savefig('collection_behavior_analysis.png', dpi=300)
plt.show()

# ------------------------------------------------
# 分析 3: 初次购买与收藏行为的比较
# ------------------------------------------------

# 计算持有0个与持有1-5个时的购买概率
adoption_rate = df_analyze[df_analyze['holdings'] == 0]['bought_more'].mean()
collection_rates = collection_df.groupby('holdings')['bought_more'].mean()

# 创建比较数据框
comparison_data = pd.DataFrame({
    'holdings': ['0 (初次购买)'] + [f"{i} (收藏行为)" for i in collection_rates.index],
    'probability': [adoption_rate] + list(collection_rates.values)
})

plt.figure(figsize=(12, 7))
ax = sns.barplot(x='holdings', y='probability', data=comparison_data,
                 palette=['#FFA500'] + ['#4682B4'] * len(collection_rates))
plt.xlabel("持有数量 (t 时刻)", fontsize=12)
plt.ylabel("购买更多的概率 (t+1 时刻)", fontsize=12)
plt.title("初次购买 vs 收藏行为的概率比较", fontsize=14)
plt.grid(True, axis='y', alpha=0.3)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

# 添加数值标签
for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 0.001, f"{height:.2%}",
            ha="center", fontsize=10)

plt.tight_layout()
plt.savefig('adoption_vs_collection_comparison.png', dpi=300)
plt.show()

# ------------------------------------------------
# 分析 4: 具有统计显著性的收藏行为临界点 - 修复版
# ------------------------------------------------

# 使用离散的持有数量点，而不是连续预测
holdings_discrete = np.array(sorted(collection_df['holdings'].unique()))

# 使用正确的方式构建二次项模型
X = collection_df['holdings'].values
X_poly = np.column_stack((X, X ** 2))
X_poly = sm.add_constant(X_poly)
y_poly = collection_df['bought_more'].values

# 拟合多项式模型
poly_model = sm.Logit(y_poly, X_poly).fit()
print("\n==== 多项式回归模型（寻找临界点）====")
print(poly_model.summary())

# 获取系数（现在使用.iloc来避免FutureWarning）
b0 = poly_model.params.iloc[0]  # 常数项
b1 = poly_model.params.iloc[1]  # 一次项
b2 = poly_model.params.iloc[2]  # 二次项

# 对实际出现过的持有数量进行预测
X_pred_discrete = np.column_stack((holdings_discrete, holdings_discrete ** 2))
X_pred_discrete = sm.add_constant(X_pred_discrete)
pred_probs_discrete = poly_model.predict(X_pred_discrete)

# 可视化多项式拟合曲线 - 只使用离散点
plt.figure(figsize=(10, 6))
plt.scatter(collection_plot_data['holdings'], collection_plot_data['bought_more'],
            color='blue', s=100, alpha=0.7, label='观察数据')
plt.plot(holdings_discrete, pred_probs_discrete, 'r-', linewidth=2, label='多项式拟合')

# 找出拟合曲线的最高点（如果存在）
if b2 < 0:  # 二次项系数为负，存在最大值
    # 求导数为0的点：b1 + 2*b2*x = 0 => x = -b1/(2*b2)
    critical_point = -b1 / (2 * b2)

    # 检查临界点是否在合理范围内
    if 1 <= critical_point <= 5:
        # 计算最接近的整数点（因为NFT持有量必须为整数）
        nearest_int = int(round(critical_point))

        # 标记临界点位置（使用最接近的整数）
        plt.axvline(x=nearest_int, color='green', linestyle='--', alpha=0.7,
                    label=f'临界点: ≈ {nearest_int} 个NFT')

        # 计算临界点（整数）处的购买概率
        critical_X = sm.add_constant(np.array([[nearest_int, nearest_int ** 2]]))
        critical_prob = poly_model.predict(critical_X)[0]
        plt.plot([nearest_int], [critical_prob], 'go', ms=10)

        print(f"\n找到的临界点: ≈ {nearest_int} 个NFT (实际值: {critical_point:.2f})")
        print(f"此点的购买概率: {critical_prob:.2%}")

        # 理论上的最优点（可能不是整数）
        x_theoretical = np.linspace(min(holdings_discrete), max(holdings_discrete), 100)
        X_theoretical = sm.add_constant(np.column_stack((x_theoretical, x_theoretical ** 2)))
        y_theoretical = poly_model.predict(X_theoretical)

        # 在图中标记理论上的最佳点
        theoretical_max_idx = np.argmax(y_theoretical)
        theoretical_x = x_theoretical[theoretical_max_idx]
        theoretical_y = y_theoretical[theoretical_max_idx]

        plt.plot([theoretical_x], [theoretical_y], 'r*', ms=12,
                 label=f'理论最佳点: {theoretical_x:.2f} 个NFT')

plt.xlabel("持有数量 (t 时刻)", fontsize=12)
plt.ylabel("购买更多的概率 (t+1 时刻)", fontsize=12)
plt.title("NFT 收藏行为的临界点分析（多项式拟合）", fontsize=14)
plt.grid(True, alpha=0.3)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.legend()
plt.tight_layout()
plt.savefig('collection_critical_point.png', dpi=300)
plt.show()

# ------------------------------------------------
# 分析 5: 不同持有量地址的分布情况
# ------------------------------------------------

# 计算每个时间点的持有量分布
holding_distributions = []

for date in snapshot_dates:
    # 计算该时间点的持有量分布
    holdings_count = holder_df[date].value_counts().sort_index()
    # 只关注持有量为1-5的地址
    holdings_count = holdings_count[holdings_count.index.isin(range(1, 6))]
    holdings_count = holdings_count / holdings_count.sum()  # 转换为比例

    holding_distributions.append({
        'date': date,
        **{f'hold_{i}': holdings_count.get(i, 0) for i in range(1, 6)}
    })

# 转换为DataFrame并可视化
dist_df = pd.DataFrame(holding_distributions)
dist_df = dist_df.set_index('date')

plt.figure(figsize=(12, 7))
dist_df.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 7))
plt.xlabel("快照日期", fontsize=12)
plt.ylabel("地址分布比例", fontsize=12)
plt.title("不同持有量地址的分布随时间变化", fontsize=14)
plt.legend(title="持有数量", labels=[f"{i}个NFT" for i in range(1, 6)])
plt.grid(True, axis='y', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('holdings_distribution.png', dpi=300)
plt.show()