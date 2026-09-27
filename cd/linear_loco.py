import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .Linear_LoCo import Model as LinearLoCoModel  # 确保 Linear_LoCo.py 位于同一目录
from tools.tools import EarlyStoppingSimple


def linear_loco_baseline(data_sample, cfg):
    """
    Linear LoCo baseline with early stopping.
    Returns causal matrix of shape (N, N, max_lag).
    """
    # ---- 1. 数据准备（与原版一致） ----
    if hasattr(cfg, 'cut_at') and cfg.cut_at is not None:
        data_sample = data_sample[:cfg.cut_at]

    if hasattr(data_sample, 'values'):
        data_sample = data_sample.values
    data_sample = data_sample.astype(np.float32)

    N = data_sample.shape[1]
    max_lag = cfg.max_lag
    seq_len = max_lag
    topk = getattr(cfg, 'topk', N * seq_len)
    lr = getattr(cfg, 'learning_rate', 0.001)
    epochs = getattr(cfg, 'epochs', 200)                 # 早停后可以把上限提高
    batch_size = getattr(cfg, 'batch_size', 64)
    patience = getattr(cfg, 'patience', 15)              # 早停 patience
    delta = getattr(cfg, 'early_stop_delta', 0.0)        # 最小改善阈值
    val_ratio = getattr(cfg, 'val_ratio', 0.2)           # 验证集比例（按时间顺序取尾段）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_norm = data_sample  # 若需要可恢复标准化

    # 构建滑动窗口
    T = data_norm.shape[0]
    X_list, y_list = [], []
    for t in range(seq_len, T):
        X_list.append(data_norm[t - seq_len:t, :])
        y_list.append(data_norm[t, :])
    X = np.array(X_list)
    y = np.array(y_list)
    n_total = X.shape[0]

    # ---- 2. 时间顺序划分 train/val ----
    n_val = max(1, int(val_ratio * n_total))
    n_train = n_total - n_val
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(),
                      torch.from_numpy(y_train).float()),
        batch_size=batch_size, shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(),
                      torch.from_numpy(y_val).float()),
        batch_size=batch_size, shuffle=False,
    )

    # ---- 3. 构建模型 ----
    args = type('Args', (), {})()
    args.input_dim = N
    args.seq_len = seq_len
    args.topk = topk
    args.mask = None
    args.gpu = device
    model = LinearLoCoModel(args).to(device)

    # print(f'[debug] topk:{topk}, batch: {batch_size}')
    # exit(0)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopper = EarlyStoppingSimple(patience=patience, delta=delta, verbose=True)

    # ---- 4. 训练循环 + 早停 ----
    for epoch in range(epochs):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x).squeeze(1)
            loss = criterion(pred, batch_y)
            loss.backward()
            model.topk_gradient_mask()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / max(1, len(train_loader))

        # ---- validation ----
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x).squeeze(1)
                val_total += criterion(pred, batch_y).item()
        val_loss = val_total / max(1, len(val_loader))

        if epoch % 100 == 0:
            print(f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}")

        # ---- early stopping check ----
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"[EarlyStopping] Triggered at epoch {epoch+1}. "
                  f"Best val_loss={early_stopper.val_loss_min:.6f}")
            break

    # ---- 5. 恢复最优权重 ----
    early_stopper.restore(model)
    model.eval()

    # ---- 6. 提取因果矩阵 ----
    S = model.get_causal_matrix()
    S = S.detach().cpu().numpy()

    # 重塑为 (N, seq_len, N)，列顺序为 [t=0..L-1, var=0..N-1]
    S_reshaped = S.reshape(N, seq_len, N)
    # 转换到 (N, N, seq_len)，其中第三维索引 lag-1 对应滞后 lag
    # 原始 S_reshaped[e, t, c] 中 t=0 表示最早历史，lag = seq_len - t
    # 我们希望 pred[e, c, lag-1] = S_reshaped[e, seq_len-lag, c]
    pred = np.zeros((N, N, seq_len))
    for lag in range(1, seq_len + 1):
        t = seq_len - lag
        pred[:, :, lag - 1] = S_reshaped[:, t, :]  # effect x cause

    # mask = 1 - np.eye(N, dtype=pred.dtype)               # shape (N, N)
    # pred = pred * mask[:, :, np.newaxis] 11 11:53 
    # print(f'[de/bug] pred {pred}')
    print(f'[debug] pred shape {pred.shape}')
    return pred