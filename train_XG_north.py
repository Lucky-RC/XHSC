import os
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # 只保存图片，不弹窗
import matplotlib.pyplot as plt
from matplotlib import colors  # 用于热力图增强对比度

# 全局字体统一为 Times New Roman，刻度线朝里
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import xgboost as xgb
import shap

# =============== 1. 路径与基础设置 ===============

feature_dir = r"F:\code_2025\tm_feature\data_north\feature_out"
save_dir    = r"F:\code_2025\tm_feature\python_2\n"
os.makedirs(save_dir, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(feature_dir, "*.csv")))
print("共找到 north CSV 文件数量:", len(csv_files))
if len(csv_files) == 0:
    raise RuntimeError("指定目录下没有 CSV 文件，请检查路径。")

# =========❗ 人为控制训练集/预测集 =========
train_files = csv_files[0:36]     # 训练集 CSV
# 测试集 CSV
predict_dir = r"F:\code_2025\tm_feature\data_north\predict_out"
test_files = sorted(glob.glob(os.path.join(predict_dir, "*.csv")))
print("\n测试集共找到 CSV 文件数量:", len(test_files))    

print("\n训练集使用的 CSV 文件：")
for f in train_files:
    print("  ", os.path.basename(f))

print("\n预测/测试集使用的 CSV 文件：")
for f in test_files:
    print("  ", os.path.basename(f))

# =============== 2. 特征定义 & 工具函数 ===============

feature_cols = ['S_LH', 'S_RH', 'R_pol', 'R_30', 'rho_axis', 'Q_RH', 'R_peak_ratio']

pretty_names = {
    'S_LH':         'LH Power',
    'S_RH':         'RH Power',
    'R_pol':        'Pol. Ratio',
    'R_30':         'Pol. Contrast',
    'rho_axis':     'Anisotropy',
    'Q_RH':         'RH SNR',
    'R_peak_ratio': 'PSLR'
}
pretty_feature_cols = [pretty_names[c] for c in feature_cols]

def clean_X(df, feat_cols):
    """去除 inf，并用 0 填 nan。"""
    return df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

def metrics_for_MYI_and_FYI(y_true_23, proba, thr):
    """
    y_true_23: 原始标签 (2=FYI,3=MYI)
    proba:     p(MYI) 概率
    thr:       判 MYI 的阈值
    """
    # 概率 → 2/3 标签
    y_pred_23 = np.where(proba >= thr, 3, 2)

    # MYI 二值化（正类）
    y_true_bin = (y_true_23 == 3).astype(int)
    y_pred_bin = (y_pred_23 == 3).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_bin, y_pred_bin, labels=[0, 1]
    ).ravel()

    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    # FYI / MYI 各自召回（用 2/3 矩阵）
    cm2 = confusion_matrix(y_true_23, y_pred_23, labels=[2, 3])
    FYI_TP = cm2[0, 0]
    FYI_FN = cm2[0, 1]
    MYI_TP = cm2[1, 1]
    MYI_FN = cm2[1, 0]

    recall_FYI = FYI_TP / max(FYI_TP + FYI_FN, 1)
    recall_MYI_cls = MYI_TP / max(MYI_TP + MYI_FN, 1)

    return {
        "P_MYI": P,
        "R_MYI": R,
        "F1_MYI": F1,
        "recall_FYI": recall_FYI,
        "recall_MYI_cls": recall_MYI_cls,
        "cm_2x2": cm2,
    }

# Score = w1 * R_MYI + w2 * R_FYI + w3 * F1_MYI  （用于模型选择）
w1, w2, w3 = 0.5, 0.4, 0.15

# XGBoost 中正类(MYI)的权重系数候选 & 阈值网格
spw_list = np.arange(1.5, 4.5, 0.1)
thr_grid = np.arange(0.1, 0.40, 0.01)

# =============== 3. 构建训练集 DataFrame（单一训练方案） ===============

df_train = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
df_train = df_train.dropna(subset=['Label'])

# 检查特征列
if not set(feature_cols).issubset(df_train.columns):
    missing = set(feature_cols) - set(df_train.columns)
    raise RuntimeError(f"训练数据缺少特征列: {missing}")

X_all = clean_X(df_train, feature_cols)
y_all = df_train['Label'].astype(int)   # 2=FYI, 3=MYI

print("\n训练数据 Label 分布（2=FYI, 3=MYI）:")
print(y_all.value_counts())

# 训练集内部划分 train / val，用于 (spw, thr) 搜索
Xtr, Xval, ytr_23, yval_23 = train_test_split(
    X_all, y_all, test_size=0.25, stratify=y_all, random_state=42
)
ytr  = (ytr_23  == 3).astype(int)   # 0/1
yval = (yval_23 == 3).astype(int)

print("\n训练/验证划分:")
print("  Xtr:", Xtr.shape, "Xval:", Xval.shape)

# =============== 4. 在 (spw, thr) 上搜索最优模型（基于验证集） ===============

best = None
# 用于画热力图：spw × thr 的 Score
score_matrix = np.zeros((len(spw_list), len(thr_grid)))

print("\n开始在当前训练集上搜索 scale_pos_weight & 阈值 thr ...")
for i_spw, spw in enumerate(spw_list):
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        scale_pos_weight=float(spw),
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss"
    )
    clf.fit(Xtr, ytr)
    proba_val = clf.predict_proba(Xval)[:, 1]  # p(MYI)

    for j_thr, thr in enumerate(thr_grid):
        m = metrics_for_MYI_and_FYI(yval_23, proba_val, thr)
        score = (w1 * m["R_MYI"] +
                 w2 * m["recall_FYI"] +
                 w3 * m["F1_MYI"])

        # 记录到矩阵中，用于画热力图
        score_matrix[i_spw, j_thr] = score

        if (best is None) or (score > best["Score_val"]):
            best = {
                "spw": spw,
                "thr": thr,
                "Score_val": score,
                "R_MYI_val": m["R_MYI"],
                "R_FYI_val": m["recall_FYI"],
                "F1_MYI_val": m["F1_MYI"],
                "cm_val": m["cm_2x2"],
                "clf": clf,
                "proba_val": proba_val,
            }

print("\n=== 验证集上找到的最优组合（单一训练方案） ===")
print(f"  scale_pos_weight (spw) = {best['spw']}")
print(f"  最佳阈值 thr           = {best['thr']:.3f}")
print(f"  Val: R_MYI={best['R_MYI_val']:.3f}, R_FYI={best['R_FYI_val']:.3f}, F1_MYI={best['F1_MYI_val']:.3f}, Score={best['Score_val']:.3f}")
print("  验证集混淆矩阵 (2=FYI,3=MYI):")
print(best["cm_val"])

# =============== 5. 画“阈值 vs 指标”曲线（基于验证集、固定最优 spw） ===============

print("\n绘制验证集上的阈值–指标曲线 ...")

# 使用最佳 spw 重新用 Xtr 训练一个模型
spw_star = best["spw"]
clf_star = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    scale_pos_weight=float(spw_star),
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    eval_metric="logloss"
)
clf_star.fit(Xtr, ytr)
proba_star = clf_star.predict_proba(Xval)[:, 1]

R_list, F1_list, R_FYI_list = [], [], []
for thr in thr_grid:
    m = metrics_for_MYI_and_FYI(yval_23, proba_star, thr)
    R_list.append(m["R_MYI"])
    F1_list.append(m["F1_MYI"])
    R_FYI_list.append(m["recall_FYI"])

plt.figure(figsize=(5, 3.5))
plt.plot(thr_grid, R_list,     label='Recall (MYI)')
plt.plot(thr_grid, F1_list,    label='F1 (MYI)')
plt.plot(thr_grid, R_FYI_list, label='Recall (FYI)', linestyle=':')
plt.axvline(best["thr"], color='k', linestyle='--', label=f'Best thr={best["thr"]:.2f}')

# 字体大小全部设为 10（此图）
plt.xlabel('Threshold for p(MYI)', fontsize=20)
plt.ylabel('Metric value', fontsize=20)
#plt.title(f'spw={3}', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# ✅ 让刻度值离坐标轴远一点
plt.tick_params(axis='both', which='major', pad=8)

plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "metrics_vs_threshold_single_case.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("阈值–指标曲线图已保存。")

# =============== 5b. 画 spw-thr-Score 热力图（增强颜色区分度） ===============

print("\n绘制 spw-thr-Score 热力图 ...")

plt.figure(figsize=(6, 4))

# 使用 PowerNorm 拉高高分区域的对比度，并使用 turbo colormap
norm = colors.PowerNorm(
    gamma=0.6,  # <1：高值区域对比度更强，可在 0.4~0.7 间微调
    vmin=score_matrix.min(),
    vmax=score_matrix.max()
)

extent = [thr_grid[0], thr_grid[-1], spw_list[0], spw_list[-1]]
im = plt.imshow(
    score_matrix,
    aspect='auto',
    origin='lower',
    extent=extent,
    cmap='plasma',   # 比 plasma 更鲜艳、层次更明显
    norm=norm
)

cbar = plt.colorbar(im)
cbar.set_label('Score', fontsize=18)
cbar.ax.tick_params(labelsize=18)

plt.xlabel('Threshold for p(MYI)', fontsize=18)
plt.ylabel('Scale pos weight (spw)', fontsize=18)
#plt.title('Score Heatmap over (spw, threshold)', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# 让刻度值离坐标轴更远
plt.tick_params(axis='both', which='major', pad=8)
cbar.ax.tick_params(pad=8)

# 让横坐标刻度数字整体向右挪一点
# 仅移动第一个 x 轴刻度数字
ax = plt.gca()
ticks = ax.xaxis.get_major_ticks()
if len(ticks) > 0:
    ticks[0].label1.set_x( ticks[0].label1.get_position()[0] + 0.15 )




# 标记最优组合位置
# ===== 用矩形框出最佳格子 =====

best_spw_idx = np.where(np.isclose(spw_list, best['spw']))[0][0]
best_thr_idx = np.where(np.isclose(thr_grid, best['thr']))[0][0]

import matplotlib.patches as patches

# 计算矩形左下角坐标
x0 = thr_grid[best_thr_idx] 
y0 = spw_list[best_spw_idx]

# 格子宽高（thr 与 spw 均为均匀间隔）
dx = thr_grid[1] - thr_grid[0]
dy = spw_list[1] - spw_list[0]

# 添加矩形框（边框粗一点，容易看）
rect = patches.Rectangle(
    (x0, y0), dx, dy,
    linewidth=2.5,
    edgecolor='black',
    facecolor='none'
)
plt.gca().add_patch(rect)


plt.tight_layout()
plt.savefig(os.path.join(save_dir, "spw_thr_score_heatmap.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("spw-thr-Score 热力图已保存。")

# =============== 6. 在“预测/测试集”上评估混淆矩阵 ===============

print("\n在指定的预测/测试集上评估最佳参数组合 ...")

df_test = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)
df_test = df_test.dropna(subset=['Label'])

X_test = clean_X(df_test, feature_cols)
y_test_23 = df_test['Label'].round().astype(int)


# 在全部训练数据 X_all 上按最佳 spw 重训一个最终模型
y_all_bin = (y_all == 3).astype(int)
xgb_final = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    scale_pos_weight=float(spw_star),
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    eval_metric="logloss"
)
xgb_final.fit(X_all, y_all_bin)

proba_test = xgb_final.predict_proba(X_test)[:, 1]
m_test = metrics_for_MYI_and_FYI(y_test_23, proba_test, best["thr"])

print("\n=== 测试集上的整体结果（单一训练配置） ===")
print(f"  Test: R_MYI={m_test['R_MYI']:.3f}, R_FYI={m_test['recall_FYI']:.3f}, F1_MYI={m_test['F1_MYI']:.3f}")
print("  测试集混淆矩阵 (2=FYI,3=MYI):")
print(m_test["cm_2x2"])

# =============== 7. XGBoost 特征重要性图（基于最终模型） ===============
# ※ 字号不调到 23，只使用 Times New Roman

print("\n绘制 XGBoost 特征重要性图 ...")

importance = xgb_final.feature_importances_
order = np.argsort(importance)[::-1]
ordered_imp = importance[order]
ordered_pretty = [pretty_feature_cols[i] for i in order]

plt.figure(figsize=(6, 3))
plt.bar(range(len(order)), ordered_imp)
plt.xticks(range(len(order)), ordered_pretty, rotation=45, ha='right')
plt.ylabel('Feature importance')
plt.title('XGBoost Feature Importance (final model)')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "xgb_feature_importance_single_case.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("特征重要性图已保存。")

# =============== 8. SHAP summary 点图 + SHAP difference（抽样画） ===============
# ※ 字号不放大到 23

print("\n开始计算最终模型的 SHAP 值（抽样点画）...")

n_shap = min(500, len(X_all))

# 保留索引用于拿到对应标签
idx_shap = X_all.sample(n=n_shap, random_state=123).index
X_shap = X_all.loc[idx_shap]
y_shap_23 = y_all.loc[idx_shap].values  # 2=FYI, 3=MYI

n_bg = min(300, len(X_all))
X_bg = X_all.sample(n=n_bg, random_state=321)

explainer = shap.TreeExplainer(
    xgb_final,
    data=X_bg,
    feature_perturbation="interventional",
    model_output="probability"
)
exp = explainer(X_shap)
vals = exp.values
# 如果是 (samples, features, classes)，取正类(MYI)
if vals.ndim == 3:
    vals = vals[:, :, 1]

# ---- 8a) SHAP summary 点图 ----
plt.figure()
shap.summary_plot(
    vals,
    X_shap,
    feature_names=pretty_feature_cols,
    max_display=len(feature_cols),
    plot_type="dot",
    show=False
)
plt.title("SHAP Summary (MYI=positive )")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "xgb_shap_summary_single_case.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("SHAP 点图已保存到:", save_dir)

# ---- 8b) SHAP difference 条形图 ----
print("\n绘制 SHAP difference 图 ...")

# 根据标签划分 FYI / MYI
mask_MYI = (y_shap_23 == 3)
mask_FYI = (y_shap_23 == 2)

vals_MYI = vals[mask_MYI]
vals_FYI = vals[mask_FYI]

mean_MYI = vals_MYI.mean(axis=0)
mean_FYI = vals_FYI.mean(axis=0)

shap_diff = mean_MYI - mean_FYI  # >0 表示该特征整体更推向 MYI

# 按差值绝对值排序，方便展示
order_diff = np.argsort(np.abs(shap_diff))[::-1]
shap_diff_sorted = shap_diff[order_diff]
pretty_sorted = [pretty_feature_cols[i] for i in order_diff]

plt.figure(figsize=(6, 3.5))
plt.bar(range(len(order_diff)), shap_diff_sorted)
plt.xticks(range(len(order_diff)), pretty_sorted, rotation=45, ha='right')
plt.ylabel('Δ mean SHAP (MYI - FYI)')
plt.title('SHAP Difference by Feature')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "xgb_shap_difference_single_case.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("SHAP difference 图已保存。")

# =============== 9. 遍历所有 north CSV，统计每个文件的混淆矩阵 ===============

# =============== 9. 对 predict_out 中的所有 CSV 做预测并输出 Excel ===============

print("\n开始对 predict_out 中的所有 CSV 做预测并输出结果 Excel ...")

pred_out_dir = os.path.join(save_dir, "predict_results")
os.makedirs(pred_out_dir, exist_ok=True)

rows = []   # 收集每个文件的三个指标

for path in test_files:
    fname = os.path.basename(path)
    print(f"\n处理测试文件: {fname}")

    df_day = pd.read_csv(path)

    # 检查字段
    if not {"Label", "Lat", "Lon"}.issubset(df_day.columns):
        print(f"  跳过：{fname} 缺少 Label/Lat/Lon 字段。")
        continue

    if not set(feature_cols).issubset(df_day.columns):
        print(f"  跳过：{fname} 缺少必要特征列。")
        continue

    X_day = clean_X(df_day, feature_cols)
    y_day = df_day["Label"].round().astype(int)


    # ---- 预测 ----
    proba_day = xgb_final.predict_proba(X_day)[:, 1]
    y_pred_day = np.where(proba_day >= best["thr"], 3, 2)

    # ---- 计算召回率与 accuracy ----
    cm_day = confusion_matrix(y_day, y_pred_day, labels=[2, 3])

    FYI_TP = cm_day[0, 0]
    FYI_FN = cm_day[0, 1]
    MYI_TP = cm_day[1, 1]
    MYI_FN = cm_day[1, 0]

    rFYI = FYI_TP / max(FYI_TP + FYI_FN, 1)
    rMYI = MYI_TP / max(MYI_TP + MYI_FN, 1)
    acc = (FYI_TP + MYI_TP) / len(df_day)

    print(f"  FYI Recall = {rFYI:.3f}, MYI Recall = {rMYI:.3f}, Acc = {acc:.3f}")

    # ---- 生成当前 CSV 的完整预测结果 Excel ----
    df_pred = pd.DataFrame({
        "Lat": df_day["Lat"].values,
        "Lon": df_day["Lon"].values,
        "true_label": y_day.values,
        "pred_label": y_pred_day,
        "proba_MYI": proba_day
    })

    excel_path = os.path.join(pred_out_dir, fname.replace(".csv", "_predict_result.xlsx"))
    df_pred.to_excel(excel_path, index=False)
    print(f"  已输出预测结果 Excel：{excel_path}")

    # ---- 保存汇总指标 ----
    rows.append({
        "file_name": fname,
        "recall_FYI": rFYI,
        "recall_MYI": rMYI,
        "accuracy": acc
    })

# =============== 10. 整体汇总表（每个 CSV 三个指标） ===============

summary_path = os.path.join(save_dir, "predict_metrics_summary.xlsx")
pd.DataFrame(rows).to_excel(summary_path, index=False)

print("\n所有测试文件已完成预测，总指标 Excel 输出在：", summary_path)
