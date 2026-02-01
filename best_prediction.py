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
import wandb
from wandb.integration.keras import WandbMetricsLogger

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 加载 Sweep 产出的最佳配置
# ==========================================
def load_best_config(path='best_config.json'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 未找到 {path}，请先运行 sweep 脚本或手动创建该文件。")
    with open(path, 'r') as f:
        return json.load(f)


DATA_PATH = r'C:\Users\Admin\Desktop\丹江口水文水质数据\用于代码的文件\2024年泗河河口数据.xlsx'


def run_final_prediction():
    # 加载配置
    CONFIG = load_best_config()

    # 创建带时间戳的运行名称
    timestamp = datetime.datetime.now().strftime('%Y年%m月%d日_%H时%M分')
    run_name = f"Final_Best_Model_{timestamp}"

    # 初始化 WandB
    wandb.init(
        project="chla_final_prediction",  # 这里可以单独开一个项目记录最终结果
        config=CONFIG,
        name=run_name,
        save_code=True
    )

    # 2. 数据处理 (严格按照 7:3 顺序切分)
    df = pd.read_excel(DATA_PATH)
    raw_X = df.iloc[:, 1:-1].values
    raw_y = df.iloc[:, -1].values.reshape(-1, 1)

    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_scaled = scaler_x.fit_transform(raw_X)
    y_scaled = scaler_y.fit_transform(raw_y)

    X_final = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    train_size = int(len(X_final) * 0.7)
    X_train, X_val = X_final[:train_size], X_final[train_size:]
    y_train, y_val = y_scaled[:train_size], y_scaled[train_size:]

    # 3. 构建模型 (使用最佳参数)
    model = Sequential([
        LSTM(CONFIG['lstm_units'],
             input_shape=(1, X_train.shape[2]),
             kernel_regularizer=l2(CONFIG.get('l2_reg', 0.01))),
        Dropout(CONFIG['dropout_rate']),
        Dense(8, activation='relu'),
        Dense(1)
    ])

    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        metrics=['mae']
    )

    # 4. 训练回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15)

    print(f"🚀 开始最终训练，使用最佳参数: {CONFIG}")
    model.fit(
        X_train, y_train,
        epochs=CONFIG.get('epochs', 150),
        batch_size=CONFIG['batch_size'],
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[WandbMetricsLogger(), early_stopping, reduce_lr]
    )

    # 5. 预测与反归一化
    y_pred_scaled = model.predict(X_val)
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
    y_val_real = scaler_y.inverse_transform(y_val)

    final_mae = np.mean(np.abs(y_pred_real - y_val_real))
    wandb.log({"final_real_mae": final_mae})

    # 6. 可视化连续的预测曲线
    plt.figure(figsize=(15, 6))
    plt.plot(y_val_real, label='真实观测值 (Test Set)', color='#1f77b4', linewidth=2)
    plt.plot(y_pred_real, label='模型预测值 (Best Model)', color='#d62728', linestyle='--', linewidth=2)
    plt.title(f'最佳模型预测效果对比\n(MAE: {final_mae:.4f})', fontsize=14)
    plt.xlabel('时间样本序号 (70%训练后剩余的30%数据)')
    plt.ylabel('叶绿素 a 浓度')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    # 保存并上传图片
    plt.savefig('final_prediction_plot.png')
    wandb.log({"final_prediction_plot": wandb.Image(plt)})
    plt.show()

    wandb.finish()
    print(f"✅ 复现完成！最佳参数 MAE: {final_mae:.4f}")


if __name__ == '__main__':
    run_final_prediction()