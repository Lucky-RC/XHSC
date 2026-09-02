import os
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # 只保存图不弹窗
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches

# ---------------------- 全局图像格式 ----------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import xgboost as xgb
import shap


# =========================================================
# 1. 路径设置
# =========================================================

feature_dir = r"F:\code_2025\tm_feature\data_south\feature_out"
predict_dir = r"F:\code_2025\tm_feature\data_south\predict_out"
save_dir    = r"F:\code_2025\tm_feature\python_2\s"

os.makedirs(save_dir, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(feature_dir, "*.csv")))
print("训练数据 CSV 文件数量:", len(csv_files))

test_files = sorted(glob.glob(os.path.join(predict_dir, "*.csv")))
print("预测（predict_out）CSV 文件数量:", len(test_files))

train_files = csv_files[0:37]
print("\n训练集文件:")
for f in train_files:
    print(" ", os.path.basename(f))

print("\n预测集文件（不用于评估，仅用于预测）:")
for f in test_files:
    print(" ", os.path.basename(f))


# =========================================================
# 2. 特征、工具函数
# =========================================================

feature_cols = ['S_LH', 'S_RH', 'R_pol', 'R_30',
                'rho_axis', 'Q_RH', 'R_peak_ratio']

pretty_names = {
    'S_LH': 'LH Power',
    'S_RH': 'RH Power',
    'R_pol': 'Pol. Ratio',
    'R_30': 'Pol. Contrast',
    'rho_axis': 'Anisotropy',
    'Q_RH': 'RH SNR',
    'R_peak_ratio': 'PSLR'
}
pretty_feature_cols = [pretty_names[c] for c in feature_cols]


def clean_X(df, feat_cols):
    """去除 inf, nan"""
    return df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)


def metrics_for_MYI_and_FYI(y_true_23, proba, thr):
    """训练/验证阶段专用，不用于 predict_out"""
    y_pred_23 = np.where(proba >= thr, 3, 2)
    y_true_bin = (y_true_23 == 3).astype(int)
    y_pred_bin = (y_pred_23 == 3).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

    cm2 = confusion_matrix(y_true_23, y_pred_23, labels=[2, 3])
    FYI_TP = cm2[0, 0]
    FYI_FN = cm2[0, 1]
    MYI_TP = cm2[1, 1]
    MYI_FN = cm2[1, 0]

    recall_FYI = FYI_TP / max(FYI_TP + FYI_FN, 1)
    recall_MYI = MYI_TP / max(MYI_TP + MYI_FN, 1)

    return {
        "P_MYI": P,
        "R_MYI": R,
        "F1_MYI": F1,
        "recall_FYI": recall_FYI,
        "recall_MYI_cls": recall_MYI,
        "cm_2x2": cm2
    }


# =========================================================
# 3. 构建训练集
# =========================================================

df_train = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
df_train = df_train.dropna(subset=['Label'])

X_all = clean_X(df_train, feature_cols)
y_all = df_train['Label'].astype(int)

Xtr, Xval, ytr_23, yval_23 = train_test_split(
    X_all, y_all, test_size=0.25, stratify=y_all, random_state=42)

ytr = (ytr_23 == 3).astype(int)
yval = (yval_23 == 3).astype(int)


# =========================================================
# 4. 网格搜索 (spw, thr)
# =========================================================

w1, w2, w3 = 0.4, 0.5, 0.15
spw_list = np.arange(1.5, 4.5, 0.1)
thr_grid = np.arange(0.1, 0.40, 0.01)

best = None
score_matrix = np.zeros((len(spw_list), len(thr_grid)))

print("\n开始搜索最优 spw 与 thr ...")

for i_spw, spw in enumerate(spw_list):

    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        scale_pos_weight=float(spw),
        random_state=42,
        n_jobs=-1
    )

    clf.fit(Xtr, ytr)
    proba_val = clf.predict_proba(Xval)[:, 1]

    for j_thr, thr in enumerate(thr_grid):
        m = metrics_for_MYI_and_FYI(yval_23, proba_val, thr)
        score = (w1 * m["R_MYI"] +
                 w2 * m["recall_FYI"] +
                 w3 * m["F1_MYI"])

        score_matrix[i_spw, j_thr] = score

        if (best is None) or (score > best["Score_val"]):
            best = {
                "spw": spw,
                "thr": thr,
                "Score_val": score,
                "metrics": m,
                "clf": clf
            }

print("\n最优 scale_pos_weight =", best["spw"])
print("最优 threshold =", best["thr"])


# =========================================================
# 5. 绘制阈值曲线图
# =========================================================

spw_star = best["spw"]

clf_star = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    scale_pos_weight=float(spw_star),
    random_state=42,
    n_jobs=-1
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
plt.plot(thr_grid, R_list, label='Recall (MYI)')
plt.plot(thr_grid, F1_list, label='F1 (MYI)')
plt.plot(thr_grid, R_FYI_list, label='Recall (FYI)', linestyle=':')

plt.axvline(best["thr"], color='k', linestyle='--')

plt.xlabel('Threshold for p(MYI)', fontsize=20)
plt.ylabel('Metric value', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(axis='both', which='major', pad=8)

plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "threshold_curve.png"), dpi=300)
plt.close()


# =========================================================
# 6. 绘制 spw-thr Score 热力图
# =========================================================

plt.figure(figsize=(6, 4))

norm = colors.PowerNorm(
    gamma=0.6,
    vmin=score_matrix.min(),
    vmax=score_matrix.max()
)

extent = [thr_grid[0], thr_grid[-1], spw_list[0], spw_list[-1]]

im = plt.imshow(score_matrix, aspect='auto', origin='lower',
                extent=extent, cmap='viridis', norm=norm)

cbar = plt.colorbar(im)
cbar.set_label('Score', fontsize=20)

plt.xlabel('Threshold for p(MYI)', fontsize=20)
plt.ylabel('Scale pos weight (spw)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(axis='both', which='major', pad=8)

# 画矩形框
best_spw_idx = np.where(np.isclose(spw_list, best["spw"]))[0][0]
best_thr_idx = np.where(np.isclose(thr_grid, best["thr"]))[0][0]

x0 = thr_grid[best_thr_idx]
y0 = spw_list[best_spw_idx]

dx = thr_grid[1] - thr_grid[0]
dy = spw_list[1] - spw_list[0]

rect = patches.Rectangle((x0, y0), dx, dy,
                         linewidth=2.5, edgecolor='black',
                         facecolor='none')
plt.gca().add_patch(rect)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "heatmap_spw_thr.png"), dpi=300)
plt.close()


# =========================================================
# 7. 最终模型 (全训练集)
# =========================================================

y_all_bin = (y_all == 3).astype(int)

xgb_final = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    scale_pos_weight=float(spw_star),
    random_state=42,
    n_jobs=-1
)

xgb_final.fit(X_all, y_all_bin)


# =========================================================
# 8. SHAP summary + SHAP difference
# =========================================================

n_shap = min(500, len(X_all))
idx_shap = X_all.sample(n=n_shap, random_state=123).index

X_shap = X_all.loc[idx_shap]
y_shap = y_all.loc[idx_shap]

n_bg = min(300, len(X_all))
X_bg = X_all.sample(n=n_bg, random_state=321)

explainer = shap.TreeExplainer(xgb_final, data=X_bg,
                               feature_perturbation="interventional",
                               model_output="probability")

exp = explainer(X_shap)
vals = exp.values
if vals.ndim == 3:
    vals = vals[:, :, 1]

# SHAP summary
plt.figure()
shap.summary_plot(vals, X_shap,
                  feature_names=pretty_feature_cols,
                  max_display=len(feature_cols),
                  plot_type="dot",
                  show=False)
plt.savefig(os.path.join(save_dir, "shap_summary.png"), dpi=300)
plt.close()

# SHAP difference
mask_MYI = (y_shap == 3)
mask_FYI = (y_shap == 2)

mean_MYI = vals[mask_MYI].mean(axis=0)
mean_FYI = vals[mask_FYI].mean(axis=0)

shap_diff = mean_MYI - mean_FYI

order_diff = np.argsort(np.abs(shap_diff))[::-1]
shap_sorted = shap_diff[order_diff]
pretty_sorted = [pretty_feature_cols[i] for i in order_diff]

plt.figure(figsize=(6, 3.5))
plt.bar(range(len(order_diff)), shap_sorted)
plt.xticks(range(len(order_diff)), pretty_sorted,
           rotation=45, ha='right')
plt.ylabel('Δ mean SHAP (MYI - FYI)')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "shap_difference.png"), dpi=300)
plt.close()


# =========================================================
# 9. predict_out 预测（不参与评估）
# =========================================================

print("\n开始对 predict_out 文件夹执行预测 ...")

pred_save_dir = os.path.join(save_dir, "predict_results")
os.makedirs(pred_save_dir, exist_ok=True)

for path in test_files:

    fname = os.path.basename(path)
    print("  预测:", fname)

    df_day = pd.read_csv(path)

    if not {"Lat", "Lon"}.issubset(df_day.columns):
        print("   × 缺少 Lat/Lon，跳过")
        continue

    if not set(feature_cols).issubset(df_day.columns):
        print("   × 缺少特征列，跳过")
        continue

    X_day = clean_X(df_day, feature_cols)

    proba_day = xgb_final.predict_proba(X_day)[:, 1]
    pred_label = np.where(proba_day >= best["thr"], 3, 2)

    # —— 输出预测结果 Excel（不做任何评估）——
    df_pred = pd.DataFrame({
        "Lat": df_day["Lat"],
        "Lon": df_day["Lon"],
        "true_label": df_day["Label"],  # 原样输出
        "pred_label": pred_label,
        "proba_MYI": proba_day
    })

    out_path = os.path.join(pred_save_dir,
                            fname.replace(".csv", "_predict.xlsx"))
    df_pred.to_excel(out_path, index=False)

    print("    → 写入:", out_path)


print("\n============================")
print("   全部任务执行完毕！")
print("============================")
