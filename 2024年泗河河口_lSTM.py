import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler
# 移除了 train_test_split，改用手动切分

import wandb
from wandb.integration.keras import WandbMetricsLogger

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 0. 外部参数加载函数
# ==========================================
def load_config(path='config.json'):
    with open(path, 'r') as f:
        return json.load(f)


DATA_PATH = r'C:\Users\Admin\Desktop\丹江口水文水质数据\用于代码的文件\2024年泗河河口数据.xlsx'


def run_training():
    # 1. 加载配置
    CONFIG = load_config()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_name = CONFIG.get("exp_name", "chla_run")
    run_name = f"{base_name}_{timestamp}"

    # 初始化 WandB
    wandb.init(
        project=CONFIG["project_name"],
        config=CONFIG,
        name=run_name,
        save_code=True
    )
    wandb.run.log_code(".")

    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误：未找到文件 '{DATA_PATH}'")
        return

    # 2. 数据处理
    df = pd.read_excel(DATA_PATH)
    raw_X = df.iloc[:, 1:-1].values
    raw_y = df.iloc[:, -1].values.reshape(-1, 1)

    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_scaled = scaler_x.fit_transform(raw_X)
    y_scaled = scaler_y.fit_transform(raw_y)

    # LSTM 形状转换
    X_final = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # --- 修改部分：手动按 7:3 比例顺序切分数据 ---
    train_size = int(len(X_final) * 0.7)
    X_train, X_val = X_final[:train_size], X_final[train_size:]
    y_train, y_val = y_scaled[:train_size], y_scaled[train_size:]

    print(f"📊 数据切分完成: 训练集 {len(X_train)} 个样本, 测试集 {len(X_val)} 个样本")

    # 3. 构建模型
    model = Sequential([
        LSTM(CONFIG['lstm_units'],
             input_shape=(1, X_train.shape[2]),
             kernel_regularizer=l2(0.001)),
        Dropout(CONFIG['dropout_rate']),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        metrics=['mae']
    )

    # 4. 训练回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6)

    print(f"🚀 正在启动实验: {run_name}")
    model.fit(
        X_train, y_train,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[WandbMetricsLogger(), early_stopping, reduce_lr]
    )

    # 5. 预测与评估
    y_pred_scaled = model.predict(X_val)
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
    y_val_real = scaler_y.inverse_transform(y_val)

    final_mae = np.mean(np.abs(y_pred_real - y_val_real))
    wandb.log({"final_real_mae": final_mae})

    # 6. 可视化 (修改：显示全部预测序列，而不仅仅是 100 个点)
    plt.figure(figsize=(15, 6))
    plt.plot(y_val_real, label='真实观测值 (Test)', color='#1f77b4', alpha=0.8)
    plt.plot(y_pred_real, label='模型预测值 (Pred)', color='#d62728', linestyle='--', alpha=0.9)
    plt.title(f'叶绿素 a 浓度时间序列预测 - {run_name}')
    plt.xlabel('时间样本点 (顺序)')
    plt.ylabel('浓度')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 将完整的预测图上传 WandB
    wandb.log({"full_prediction_plot": wandb.Image(plt)})
    plt.show()

    wandb.finish()
    print(f"✅ 完成！MAE: {final_mae:.4f}")


if __name__ == '__main__':
    run_training()