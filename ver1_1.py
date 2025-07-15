import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.ticker import PercentFormatter

# 设置你的 snapshot 路径
SNAPSHOT_DIR = r""
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

snapshot_dates = [os.path.basename(f).split("_")[-1].replace(".csv", "") for f in snapshot_files]
holder_df = pd.DataFrame.from_dict(holder_history, orient='index').fillna(0).astype(int)
holder_df = holder_df[snapshot_dates]

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
        bought_more = 1 if after > before else 0
        records.append({
            'address': addr,
            'holdings': before,
            'bought_more': bought_more,
            'adoption': 1 if before == 0 and after > 0 else 0,
            'collection': 1 if before > 0 and after > before else 0
        })

df_analyze = pd.DataFrame(records)

# ==== 分析 1 ====
print("==== 总体购买行为分析 ====")
print(f"总记录数: {len(df_analyze)}")
print(f"初次购买次数: {df_analyze['adoption'].sum()} ({df_analyze['adoption'].mean():.2%})")
print(f"收藏行为次数: {df_analyze['collection'].sum()} ({df_analyze['collection'].mean():.2%})\n")

# ==== 分析 2 ====
print("==== 收藏行为分析（已持有NFT的地址）====")
collection_df = df_analyze[df_analyze['holdings'] > 0].copy()
collection_df = collection_df[collection_df['holdings'] <= 5]

X_collection = pd.get_dummies(collection_df['holdings'], prefix='h', drop_first=True)
X_collection = sm.add_constant(X_collection).astype(float)
y_collection = collection_df['bought_more'].astype(float)

collection_model = sm.Logit(y_collection, X_collection).fit()
print(collection_model.summary())

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

# ==== 分析 3 ====
adoption_rate = df_analyze[df_analyze['holdings'] == 0]['bought_more'].mean()
collection_rates = collection_df.groupby('holdings')['bought_more'].mean()

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

for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 0.001, f"{height:.2%}",
            ha="center", fontsize=10)

plt.tight_layout()
plt.savefig('adoption_vs_collection_comparison.png', dpi=300)
plt.show()

# ==== 分析 4 ====
holdings_discrete = np.array(sorted(collection_df['holdings'].unique()))
X = collection_df['holdings'].values
X_poly = np.column_stack((X, X ** 2))
X_poly = sm.add_constant(X_poly)
y_poly = collection_df['bought_more'].values

poly_model = sm.Logit(y_poly, X_poly).fit()
print("\n==== 多项式回归模型（寻找临界点）====")
print(poly_model.summary())

b0 = poly_model.params.iloc[0]
b1 = poly_model.params.iloc[1]
b2 = poly_model.params.iloc[2]

X_pred_discrete = np.column_stack((holdings_discrete, holdings_discrete ** 2))
X_pred_discrete = sm.add_constant(X_pred_discrete)
pred_probs_discrete = poly_model.predict(X_pred_discrete)

plt.figure(figsize=(10, 6))
plt.scatter(collection_plot_data['holdings'], collection_plot_data['bought_more'],
            color='blue', s=100, alpha=0.7, label='观察数据')
plt.plot(holdings_discrete, pred_probs_discrete, 'r-', linewidth=2, label='多项式拟合')

if b2 < 0:
    critical_point = -b1 / (2 * b2)
    if 1 <= critical_point <= 5:
        nearest_int = int(round(critical_point))
        plt.axvline(x=nearest_int, color='green', linestyle='--', alpha=0.7,
                    label=f'临界点: ≈ {nearest_int} 个NFT')

        # ✅ 修复错误：构造 2D 输入预测
        critical_X = sm.add_constant(np.array([[nearest_int, nearest_int ** 2]]))
        critical_prob = poly_model.predict(critical_X)[0]
        plt.plot([nearest_int], [critical_prob], 'go', ms=10)

        print(f"\n找到的临界点: ≈ {nearest_int} 个NFT (实际值: {critical_point:.2f})")
        print(f"此点的购买概率: {critical_prob:.2%}")

        # 理论最佳点
        x_theoretical = np.linspace(min(holdings_discrete), max(holdings_discrete), 100)
        X_theoretical = np.column_stack((x_theoretical, x_theoretical ** 2))
        X_theoretical = sm.add_constant(X_theoretical)
        y_theoretical = poly_model.predict(X_theoretical)
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

# ==== 分析 5 ====
holding_distributions = []

for date in snapshot_dates:
    holdings_count = holder_df[date].value_counts().sort_index()
    holdings_count = holdings_count[holdings_count.index.isin(range(1, 6))]
    holdings_count = holdings_count / holdings_count.sum()

    holding_distributions.append({
        'date': date,
        **{f'hold_{i}': holdings_count.get(i, 0) for i in range(1, 6)}
    })

dist_df = pd.DataFrame(holding_distributions).set_index('date')

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
