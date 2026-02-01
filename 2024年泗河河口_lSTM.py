import pandas as pd
import numpy as np
import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import wandb
from wandb.integration.keras import WandbMetricsLogger

# ==========================================
# 1. 针对小数据集优化的 Sweep 配置
# ==========================================
sweep_config = {
    'method': 'bayes',  # 对于数据量小的任务，贝叶斯搜索比随机搜索更高效
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        # 小数据集不需要太多神经元，16-64 足够
        'lstm_units': {
            'values': [16, 32, 48]
        },
        # 学习率范围收窄，避免在小样本空间跳来跳去
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.0005,
            'max': 0.005
        },
        # 小数据集通常适合较小的 batch_size（如 4 或 8）
        'batch_size': {
            'values': [4, 8, 16]
        },
        # 增加 Dropout 范围，强力防止过拟合
        'dropout_rate': {
            'values': [0.2, 0.3, 0.4, 0.5]
        },
        'l2_reg': {
            'values': [0.001, 0.01, 0.1]
        },
        'epochs': {
            'value': 150
        }
    }
}

DATA_PATH = r'C:\Users\Admin\Desktop\丹江口水文水质数据\用于代码的文件\2024年泗河河口数据.xlsx'


def train_func():
    # 初始化 Sweep Run
    with wandb.init() as run:
        config = wandb.config

        # --- 核心修改：设置符合中文习惯的时间戳 ---
        # 格式：2024年02月01日_10时15分
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y年%m月%d日_%H时%M分')

        # 拼接名字：时间 + 参数特征（比如神经元数和学习率）
        # 这样你在列表里一眼就能看出哪个时间点的哪组参数效果好
        run_name = f"{timestamp}_{config.lstm_units}u_lr{config.learning_rate:.4f}"

        # 强制将名字同步给 WandB
        run.name = run_name

        # 2. 数据处理 (保持顺序切分)
        df = pd.read_excel(DATA_PATH)
        raw_X = df.iloc[:, 1:-1].values
        raw_y = df.iloc[:, -1].values.reshape(-1, 1)

        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_scaled = scaler_x.fit_transform(raw_X)
        y_scaled = scaler_y.fit_transform(raw_y)

        X_final = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        # 7:3 比例切分
        train_size = int(len(X_final) * 0.7)
        X_train, X_val = X_final[:train_size], X_final[train_size:]
        y_train, y_val = y_scaled[:train_size], y_scaled[train_size:]

        # 3. 构建模型
        model = Sequential([
            LSTM(config.lstm_units,
                 input_shape=(1, X_train.shape[2]),
                 kernel_regularizer=l2(config.l2_reg)),  # 动态 L2
            Dropout(config.dropout_rate),
            Dense(8, activation='relu'),  # 减少了中间层神经元
            Dense(1)
        ])

        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=config.learning_rate),
            metrics=['mae']
        )

        # 4. 回调 (针对小样本波动，增加 patience)
        early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

        # 5. 训练
        model.fit(
            X_train, y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=(X_val, y_val),
            verbose=0,
            callbacks=[WandbMetricsLogger(), early_stopping]
        )

        # 记录训练完成
        print(f"✔️ 完成一组尝试: units={config.lstm_units}, lr={config.learning_rate:.4f}")


# ==========================================
# 主程序入口
# ==========================================
if __name__ == '__main__':
    # 1. 设置你的基本信息（请根据你的 WandB 账号填写）
    YOUR_WANDB_USERNAME = "你的用户名"  # 网页左上角能看到
    YOUR_PROJECT_NAME = "chla_small_data_sweep"

    # 2. 创建 Sweep
    sweep_id = wandb.sweep(sweep_config, project=YOUR_PROJECT_NAME)

    # 3. 运行 Agent
    wandb.agent(sweep_id, function=train_func, count=15)

    # 4. 结束后获取最佳参数 (这里修改了获取方式，避免 NoneType 报错)
    api = wandb.Api()

    # 组合成完整的路径
    sweep_path = f"{YOUR_WANDB_USERNAME}/{YOUR_PROJECT_NAME}/{sweep_id}"

    try:
        sweep = api.sweep(sweep_path)
        best_run = sweep.best_run()

        print("\n" + "=" * 30)
        print("🏆 最佳参数组合已找到！")
        print(best_run.config)
        print(f"最佳 val_loss: {best_run.summary['val_loss']:.6f}")
        print("=" * 30)

        # 自动保存最佳配置
        with open('best_config.json', 'w') as f:
            json.dump(best_run.config, f, indent=4)
        print("💾 最佳参数已保存至 'best_config.json'")

    except Exception as e:
        print(f"❌ 自动提取最佳参数失败: {e}")
        print("💡 请直接前往 WandB 网页端的 Sweep 页面查看最佳结果。")