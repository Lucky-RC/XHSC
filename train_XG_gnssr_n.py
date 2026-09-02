import os
import glob
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
import xgboost as xgb


# ============================================================
# 1. 路径与基础设置
# ============================================================
feature_dir = r"I:\code\python\north"
save_dir = r"I:\code\python\Result"

os.makedirs(save_dir, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(feature_dir, "*.csv")))

print("共找到 CSV 文件数量:", len(csv_files))

if len(csv_files) == 0:
    raise RuntimeError("指定目录下没有 CSV 文件，请检查路径。")

# ============================================================
# 2. 划分训练集和测试集
# ============================================================

# 前37个文件用于训练
train_files = csv_files[:37]

# 第38个文件开始用于独立测试
test_files = csv_files[37:]

print("\n训练文件数量:", len(train_files))
print("测试文件数量:", len(test_files))

print("\n训练文件:")
for f in train_files:
    print("  ", os.path.basename(f))

print("\n测试文件:")
for f in test_files:
    print("  ", os.path.basename(f))

# ============================================================
# 3. 特征设置
# ============================================================

feature_cols = [
    'S_LH',
    'rho_axis',
    'R_peak_ratio'
]

# ============================================================
# 4. 工具函数
# ============================================================

def clean_X(df, feat_cols):
    """
    清理模型输入特征：
    inf / -inf -> NaN -> 0
    """
    return (
        df[feat_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

def mask_for_model(df):
    """
    模型使用的数据：

    Label = 2：FYI
    Label = 3：MYI

    同时要求：
    Flag_LH = 0
    Flag_RH = 0
    """
    need_cols = {"Label", "Flag_LH", "Flag_RH"}

    missing = need_cols - set(df.columns)

    if missing:
        raise RuntimeError(f"数据缺少必要字段: {missing}")

    label = df["Label"].round().astype(int)

    m = label.isin([2, 3])

    m = m & (
        df["Flag_LH"].round().astype(int) == 0
    )

    m = m & (
        df["Flag_RH"].round().astype(int) == 0
    )

    return m

def calculate_metrics(y_true, y_pred):
    """
    计算 FYI / MYI 分类指标
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[2, 3]
    )

    FYI_TP = cm[0, 0]
    FYI_FN = cm[0, 1]

    MYI_TP = cm[1, 1]
    MYI_FN = cm[1, 0]

    recall_FYI = FYI_TP / max(FYI_TP + FYI_FN, 1)
    recall_MYI = MYI_TP / max(MYI_TP + MYI_FN, 1)

    accuracy = (
        FYI_TP + MYI_TP
    ) / max(len(y_true), 1)

    return recall_FYI, recall_MYI, accuracy, cm

# ============================================================
# 5. 读取训练数据
# ============================================================

print("\n开始读取训练数据...")

df_train = pd.concat(
    [pd.read_csv(f) for f in train_files],
    ignore_index=True
)

# 检查特征
missing = set(feature_cols) - set(df_train.columns)

if missing:
    raise RuntimeError(
        f"训练数据缺少特征列: {missing}"
    )

# 数据筛选
m_train = mask_for_model(df_train)

df_train_use = df_train.loc[m_train].copy()


print(
    "训练数据过滤前:",
    len(df_train)
)

print(
    "训练数据过滤后:",
    len(df_train_use)
)

# ============================================================
# 6. 构建训练数据
# ============================================================

X_train = clean_X(
    df_train_use,
    feature_cols
)

y_train_23 = (
    df_train_use["Label"]
    .round()
    .astype(int)
)

# 转换为 XGBoost 二分类：
# FYI = 0
# MYI = 1

y_train = (
    y_train_23 == 3
).astype(int)


print("\n训练数据类别数量:")

print(
    "FYI:",
    np.sum(y_train == 0)
)

print(
    "MYI:",
    np.sum(y_train == 1)
)

# ============================================================
# 7. 自动计算类别权重
# ============================================================

n_FYI = np.sum(y_train == 0)
n_MYI = np.sum(y_train == 1)

scale_pos_weight = (
    n_FYI / max(n_MYI, 1)
)
print(
    "\nscale_pos_weight =",
    scale_pos_weight
)


# ============================================================
# 8. 训练最终 XGBoost 模型
# ============================================================

print("\n开始训练 XGBoost...")


model = xgb.XGBClassifier(

    n_estimators=500,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    scale_pos_weight=scale_pos_weight,

    reg_lambda=1.0,

    n_jobs=-1,

    random_state=42,

    eval_metric="logloss"
)


model.fit(
    X_train,
    y_train
)

print("XGBoost 训练完成。")


# ============================================================
# 9. 整体测试集评价
# ============================================================

if len(test_files) > 0:

    print("\n开始整体测试...")


    df_test = pd.concat(
        [pd.read_csv(f) for f in test_files],
        ignore_index=True
    )


    m_test = mask_for_model(df_test)

    df_test_use = (
        df_test.loc[m_test].copy()
    )


    X_test = clean_X(
        df_test_use,
        feature_cols
    )


    y_test = (
        df_test_use["Label"]
        .round()
        .astype(int)
        .values
    )

    # XGBoost 默认 threshold = 0.5
    pred_bin = model.predict(X_test)


    # 0 -> FYI (2)
    # 1 -> MYI (3)

    y_pred = np.where(
        pred_bin == 1,
        3,
        2
    )

    (
        rFYI,
        rMYI,
        acc,
        cm
    ) = calculate_metrics(
        y_test,
        y_pred
    )

    print("\n==============================")
    print("测试集整体结果")
    print("==============================")

    print(
        f"FYI Recall = {rFYI:.4f}"
    )

    print(
        f"MYI Recall = {rMYI:.4f}"
    )

    print(
        f"Accuracy   = {acc:.4f}"
    )

    print(
        "\n混淆矩阵 [FYI, MYI]:"
    )

    print(cm)

# ============================================================
# 10. 每日测试和预测结果输出
# ============================================================

print("\n开始逐日预测...")

pred_out_dir = os.path.join(
    save_dir,
    "predict_results"
)

os.makedirs(
    pred_out_dir,
    exist_ok=True
)


summary_rows = []


for path in test_files:

    fname = os.path.basename(path)

    print(
        "\n处理:",
        fname
    )


    df_day = pd.read_csv(path)


    # 检查字段
    need_cols = {
        "Label",
        "Lat",
        "Lon",
        "Flag_LH",
        "Flag_RH"
    }


    if not need_cols.issubset(df_day.columns):

        print(
            "缺少必要字段，跳过。"
        )

        continue


    if not set(feature_cols).issubset(df_day.columns):

        print(
            "缺少特征列，跳过。"
        )

        continue


    # ========================================================
    # 有效数据
    # ========================================================

    m_use = mask_for_model(df_day)

    idx_use = np.where(
        m_use.values
    )[0]


    # 全部真实 Label
    y_true_all = (
        df_day["Label"]
        .round()
        .astype(int)
        .values
    )


    # 初始化预测结果
    pred_all = np.full(
        len(df_day),
        np.nan
    )

    proba_all = np.full(
        len(df_day),
        np.nan
    )


    # ========================================================
    # 模型预测
    # ========================================================

    if len(idx_use) > 0:

        X_use = clean_X(
            df_day.loc[m_use],
            feature_cols
        )


        # MYI 概率
        proba_use = (
            model.predict_proba(X_use)[:, 1]
        )


        # 默认 threshold = 0.5
        pred_bin = model.predict(X_use)


        pred_use = np.where(
            pred_bin == 1,
            3,
            2
        )


        pred_all[idx_use] = pred_use

        proba_all[idx_use] = proba_use


        # ====================================================
        # 每日精度
        # ====================================================

        y_use = y_true_all[idx_use]


        (
            rFYI,
            rMYI,
            acc,
            cm_day
        ) = calculate_metrics(
            y_use,
            pred_use
        )


    else:

        rFYI = np.nan
        rMYI = np.nan
        acc = np.nan


    print(
        f"有效样本 = {len(idx_use)}, "
        f"FYI Recall = {rFYI:.4f}, "
        f"MYI Recall = {rMYI:.4f}, "
        f"Accuracy = {acc:.4f}"
    )


    # ========================================================
    # 保存逐点预测结果
    # ========================================================

    df_pred = pd.DataFrame({

        "Lat":
            df_day["Lat"].values,

        "Lon":
            df_day["Lon"].values,

        "true_label":
            y_true_all,

        "pred_label":
            pred_all,

        "proba_MYI":
            proba_all

    })


    excel_path = os.path.join(
        pred_out_dir,
        fname.replace(
            ".csv",
            "_predict_result.xlsx"
        )
    )


    df_pred.to_excel(
        excel_path,
        index=False
    )

    # ========================================================
    # 每日汇总
    # ========================================================

    summary_rows.append({

        "file_name":
            fname,

        "n_total":
            len(df_day),

        "n_used":
            len(idx_use),

        "recall_FYI":
            rFYI,

        "recall_MYI":
            rMYI,

        "accuracy":
            acc

    })


# ============================================================
# 11. 保存每日精度汇总
# ============================================================

summary_path = os.path.join(
    save_dir,
    "predict_metrics_summary.xlsx"
)


pd.DataFrame(
    summary_rows
).to_excel(
    summary_path,
    index=False
)

print("\n==============================")
print("所有测试文件处理完成")
print("==============================")

print(
    "结果保存位置:",
    save_dir
)