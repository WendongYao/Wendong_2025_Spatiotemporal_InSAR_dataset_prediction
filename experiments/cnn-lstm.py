import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ===============================
# 1. 数据预处理与插值构造规则网格
# ===============================
csv_filename = 'EGMS_L3_E32N34_100km_U_2018_2022_1.csv'
data = pd.read_csv(csv_filename)

# 假设 easting 和 northing 分别在第1、2列（注意：Python索引从0开始）
easting = data.iloc[:, 1].astype(int).values
northing = data.iloc[:, 2].astype(int).values

# 构造观测点坐标，griddata 要求第一列为 x（easting）、第二列为 y（northing）
points = np.column_stack((easting, northing))

# -------------------------------
# 提取训练和测试数据
# -------------------------------
# 训练输入数据：使用第11到第311列（即索引 11 到 310，共300个时间步）
disp_train_data = data.iloc[:, 11:311].values  # shape: (num_points, 300)
# 测试目标数据：使用第313列（即索引312）
disp_test_data = data.iloc[:, 312].values        # shape: (num_points,)

# 定义规则网格分辨率：256 x 256
num_grid_x, num_grid_y = 256, 256
grid_x, grid_y = np.mgrid[
    easting.min(): easting.max(): complex(0, num_grid_x),
    northing.min(): northing.max(): complex(0, num_grid_y)
]

# -------------------------------
# 生成训练输入的空间地图（300个时间步）
# -------------------------------
time_steps = 300
train_maps = []
for i in range(time_steps):
    disp_values = disp_train_data[:, i]
    grid_disp = griddata(points, disp_values, (grid_x, grid_y), method='linear')
    grid_disp = np.nan_to_num(grid_disp)
    train_maps.append(grid_disp)
# 得到训练数据 shape: (300, 256, 256)
train_maps = np.stack(train_maps, axis=0)
# 添加通道维度 -> (300, 1, 256, 256)
train_maps = train_maps[:, np.newaxis, :, :]
# 构造单个样本，加上 batch 维度，最终形状为 (1, 300, 1, 256, 256)
X = np.expand_dims(train_maps, axis=0)

# -------------------------------
# 生成测试目标的空间地图（使用第313列数据）
# -------------------------------
grid_disp_test = griddata(points, disp_test_data, (grid_x, grid_y), method='linear')
grid_disp_test = np.nan_to_num(grid_disp_test)
# 测试目标数据 shape: (256, 256)，增加通道和 batch 维度 -> (1, 1, 256, 256)
test_map = grid_disp_test[np.newaxis, np.newaxis, :, :]

# 将训练和测试数据转换为 PyTorch 张量，并迁移到 device
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(test_map, dtype=torch.float32).to(device)

print("训练输入数据形状：", X_tensor.shape)   # (1, 300, 1, 256, 256)
print("测试目标数据形状：", y_tensor.shape)      # (1, 1, 256, 256)

# ===============================
# 2. 定义 CNN-LSTM 模型 (PyTorch)
# ===============================
class CNNLSTM(nn.Module):
    def __init__(self, hidden_dim=50, output_size=(256, 256)):
        super(CNNLSTM, self).__init__()
        # CNN部分：对每个时间步的输入图像进行特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 输入: (1,256,256) -> 输出: (32,256,256)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (32,128,128)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # -> (64,128,128)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (64,64,64)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> (128,64,64)
            nn.ReLU(),
            nn.MaxPool2d(2)                               # -> (128,32,32)
        )
        # CNN输出展平特征维度：128 * 32 * 32 = 131072
        self.feature_dim = 128 * 32 * 32

        # LSTM部分：对时间序列的 CNN 特征进行建模
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_dim, batch_first=True)

        # 全连接层：将 LSTM 输出映射回整张 displacement 图像
        # 输出大小为 256 * 256 = 65536
        self.fc = nn.Linear(hidden_dim, output_size[0] * output_size[1])
        self.output_size = output_size

    def forward(self, x):
        # 输入 x 的形状: (batch, time, channel, height, width)
        batch_size, time_steps, C, H, W = x.shape
        # 合并 batch 和 time 维度，使 CNN 对每个时间步单独操作: (batch*time, channel, H, W)
        x = x.view(batch_size * time_steps, C, H, W)
        cnn_features = self.cnn(x)  # 输出形状: (batch*time, 128, 32, 32)
        cnn_features = cnn_features.view(batch_size, time_steps, -1)  # (batch, time, feature_dim)

        # LSTM建模，取最后一个时间步的输出作为整体特征
        lstm_out, _ = self.lstm(cnn_features)  # lstm_out: (batch, time, hidden_dim)
        final_feature = lstm_out[:, -1, :]      # (batch, hidden_dim)

        # 全连接层映射回空间图像大小
        out = self.fc(final_feature)            # (batch, 256*256)
        out = out.view(batch_size, 1, self.output_size[0], self.output_size[1])
        return out

# 实例化模型并迁移到 device
model = CNNLSTM(hidden_dim=50, output_size=(256, 256)).to(device)
print(model)

# ===============================
# 3. 训练模型（添加 tqdm 进度条）
# ===============================
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
num_epochs = 20000  # 根据实际需要调整

model.train()
pbar = tqdm(range(num_epochs), desc="Training epochs")
for epoch in pbar:
    optimizer.zero_grad()
    output = model(X_tensor)  # 输出形状: (batch, 1, 256, 256)
    loss = criterion(output, y_tensor)  # 目标为第313列生成的图像
    loss.backward()
    optimizer.step()
    pbar.set_postfix(loss=loss.item())
torch.save(model.state_dict(),'goodmodelclstm.pth')
# ===============================
# 4. 预测与结果可视化 + 指标计算
# ===============================
model.eval()
with torch.no_grad():
    pred = model(X_tensor)

# 将预测结果和目标转换为 numpy 数组，并转回 CPU
pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()    # (256, 256)
target_np = y_tensor.squeeze(0).squeeze(0).cpu().numpy()  # (256, 256)

# 计算 MSE
mse = np.mean((pred_np - target_np) ** 2)
# 计算 RMSE
rmse = np.sqrt(mse)
# 计算 R^2
ss_res = np.sum((target_np - pred_np) ** 2)
ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
r2 = 1 - ss_res / ss_tot

epsilon = 1e-6  # 防除零
relative_error = np.abs(pred_np - target_np) / (np.abs(target_np) + epsilon)
relative_error1 = np.abs(pred_np - target_np)
accuracy_10pct = np.mean(relative_error < 0.10)
accuracy_20pct = np.mean(relative_error < 0.20)
accuracy_50pct = np.mean(relative_error < 0.50)
accuracy_1mm = np.mean(relative_error1 < 1)
accuracy_05mm = np.mean(relative_error1 < 0.5)
accuracy_02mm = np.mean(relative_error1 < 0.2)
accuracy_01mm = np.mean(relative_error1 < 0.1)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")


print(f"Accuracy (within 10% relative error): {accuracy_10pct * 100:.2f}%")
print(f"Accuracy (within 20% relative error): {accuracy_20pct * 100:.2f}%")
print(f"Accuracy (within 50% relative error): {accuracy_50pct * 100:.2f}%")
print(f"Accuracy (within 1mm relative error): {accuracy_1mm * 100:.2f}%")
print(f"Accuracy (within 0.5mm relative error): {accuracy_05mm * 100:.2f}%")
print(f"Accuracy (within 0.2mm relative error): {accuracy_02mm * 100:.2f}%")
print(f"Accuracy (within 0.1mm relative error): {accuracy_01mm * 100:.2f}%")


# ——— 在这里插入：计算完 pred_np, target_np 和各项指标之后 ———

# 1. 准备数据
y_true = target_np.flatten()
y_pred = pred_np.flatten()
residuals = y_pred - y_true
abs_residuals = np.abs(residuals)

# 2. 散点图：真实值 vs 预测值
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, s=1, alpha=0.3)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=1)
plt.xlabel("True Displacement (mm)")
plt.ylabel("Predicted Displacement (mm)")
plt.title("Scatter: True vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_true_vs_pred.png", dpi=150)

# 3. 残差图：真实值 vs 残差
plt.figure(figsize=(6,4))
plt.scatter(y_true, residuals, s=1, alpha=0.3)
plt.axhline(0, color='r', linestyle='--', linewidth=1)
plt.xlabel("True Displacement (mm)")
plt.ylabel("Residual (Pred – True) (mm)")
plt.title("Residual Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150)

# 4. 分箱误差：按真实值分箱，统计每箱平均绝对误差
num_bins = 20
bins = np.linspace(y_true.min(), y_true.max(), num_bins+1)
bin_indices = np.digitize(y_true, bins) - 1  # 0-based
bin_centers = 0.5 * (bins[:-1] + bins[1:])

mean_abs_err = [
    abs_residuals[bin_indices == i].mean() if np.any(bin_indices == i) else np.nan
    for i in range(num_bins)
]

plt.figure(figsize=(6,4))
plt.plot(bin_centers, mean_abs_err, marker='o', linestyle='-')
plt.xlabel("True Displacement Bin Center (mm)")
plt.ylabel("Mean Absolute Error (mm)")
plt.title("Binned Error Analysis")
plt.grid(True)
plt.tight_layout()
plt.savefig("binned_error.png", dpi=150)



num_bins = 20
bins = np.linspace(y_true.min(), y_true.max(), num_bins+1)
bin_indices = np.digitize(y_true, bins) - 1  # 0-based
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# 为每个 bin 收集残差
residuals_by_bin = [
    residuals[bin_indices == i]
    for i in range(num_bins)
]

# 绘制箱线图
plt.figure(figsize=(8, 4))
# showfliers=True 显示异常值，可根据需求设置
plt.boxplot(residuals_by_bin, showfliers=True, widths=0.6)
plt.xticks(
    np.arange(1, num_bins+1),
    [f"{center:.1f}" for center in bin_centers],
    rotation=45
)
plt.xlabel("True Displacement Bin Center (mm)")
plt.ylabel("Residual (Pred – True) (mm)")
plt.title("Binned Residuals Boxplot")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("binned_residuals_boxplot.png", dpi=150)

print("已生成：binned_residuals_boxplot.png")


print("已生成：scatter_true_vs_pred.png, residual_plot.png, binned_error.png")

# 可视化预测结果
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("real Displacement Map last time step")
plt.imshow(target_np, cmap='viridis',vmin=-40, vmax=20)
plt.colorbar(label="Displacement (mm)")
plt.subplot(1, 2, 2)
plt.title("Estimated Displacement Map")
plt.imshow(pred_np, cmap='viridis',vmin=-40, vmax=20)
plt.colorbar(label="Displacement (mm)")
plt.tight_layout()
plt.savefig("sp-perfect.png", dpi=400)
print("Pipeline complete.")
