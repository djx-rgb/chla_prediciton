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
from sklearn.model_selection import train_test_split

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

    # --- 新增：生成年月日时分秒时间戳 ---
    # 格式如：2024-05-20_14-30-05
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # 从 config 读取基础名，如果没有则默认为 "exp"
    base_name = CONFIG.get("exp_name", "chla_run")
    run_name = f"{base_name}_{timestamp}"

    # 初始化 WandB，设置自定义名称
    wandb.init(
        project=CONFIG["project_name"],
        config=CONFIG,
        name=run_name, # 这里应用了带时间戳的名字
        save_code=True)
    wandb.run.log_code(".")

    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误：未找到文件 '{DATA_PATH}'")
        return

    # 2. 数据处理
    df = pd.read_excel(DATA_PATH)
    # 假设第一列是日期，倒数第一列是标签
    raw_X = df.iloc[:, 1:-1].values
    raw_y = df.iloc[:, -1].values.reshape(-1, 1)

    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_scaled = scaler_x.fit_transform(raw_X)
    y_scaled = scaler_y.fit_transform(raw_y)

    # LSTM 需要 [samples, time_steps, features] 格式
    X_final = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    X_train, X_val, y_train, y_val = train_test_split(
        X_final, y_scaled,
        test_size=CONFIG.get('test_size', 0.2),
        random_state=CONFIG.get('random_seed', 42),
        shuffle=True
    )

    # 3. 构建模型 (加入 L2 正则化防过拟合)
    model = Sequential([
        LSTM(CONFIG['lstm_units'],
             input_shape=(1, X_train.shape[2]),
             kernel_regularizer=l2(0.001)),  # L2 正则
        Dropout(CONFIG['dropout_rate']),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        metrics=['mae']
    )

    # 4. 训练回调 (加入 ReduceLROnPlateau 动态调优)
    early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6)

    print(f"🚀 正在启动实验: {run_name}")
    print("📈 同步至 Wandb...")

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

    # 计算真实尺度的 MAE 并记录到 WandB
    final_mae = np.mean(np.abs(y_pred_real - y_val_real))
    wandb.log({"final_real_mae": final_mae})

    # 6. 可视化简易验证 (可选)
    plt.figure(figsize=(10, 5))
    plt.plot(y_val_real[:100], label='真实值', color='blue')
    plt.plot(y_pred_real[:100], label='预测值', color='red', linestyle='--')
    plt.title(f'实验评估 - {run_name}')
    plt.legend()
    # 可以选择把图片也传到 WandB
    wandb.log({"prediction_plot": wandb.Image(plt)})
    plt.show()

    wandb.finish()
    print(f"✅ 完成！该次运行 ID 为: {run_name}")
    print(f"📊 最终真实 MAE: {final_mae:.4f}")


if __name__ == '__main__':
    run_training()