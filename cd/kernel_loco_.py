import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .Kernel_LoCo import Model as KernelLoCoModel
from tools.tools import EarlyStoppingSimple

def kernel_loco_baseline(data_sample, cfg):
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
    epochs = getattr(cfg, 'epochs', 200)
    batch_size = getattr(cfg, 'batch_size', 64)
    D = getattr(cfg, 'D', 50)
    sigma = getattr(cfg, 'sigma', 1.0)
    patience = getattr(cfg, 'patience', 15)
    delta = getattr(cfg, 'early_stop_delta', 0.0)
    val_ratio = getattr(cfg, 'val_ratio', 0.1)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # 标准化（Kernel 版本建议保留，因为 RFF 对尺度较敏感）
    # mean = np.mean(data_sample, axis=0, keepdims=True)
    # std = np.std(data_sample, axis=0, keepdims=True) + 1e-8
    # data_norm = (data_sample - mean) / std
    data_norm = data_sample # [0924]

    # 滑动窗口
    T = data_norm.shape[0]
    X_list, y_list = [], []
    for t in range(seq_len, T):
        X_list.append(data_norm[t - seq_len:t, :])
        y_list.append(data_norm[t, :])
    X = np.array(X_list)
    y = np.array(y_list)
    n_total = X.shape[0]

    # 时间顺序划分
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

    # 模型
    args = type('Args', (), {})()
    args.input_dim = N
    args.seq_len = seq_len
    args.D = D
    args.sigma = sigma
    args.topk = topk
    args.mask = None
    args.gpu = device
    model = KernelLoCoModel(args).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopper = EarlyStoppingSimple(patience=patience, delta=delta, verbose=True)

    # 训练 + 早停
    for epoch in range(epochs):
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

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x).squeeze(1)
                val_total += criterion(pred, batch_y).item()
        val_loss = val_total / max(1, len(val_loader))

        print(f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.6f} "
              f"val_loss={val_loss:.6f}")

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"[EarlyStopping] Triggered at epoch {epoch+1}. "
                  f"Best val_loss={early_stopper.val_loss_min:.6f}")
            break

    early_stopper.restore(model)
    model.eval()

    # 提取因果矩阵
    S = model.get_causal_matrix()
    S = S.detach().cpu().numpy()

    S_reshaped = S.reshape(N, seq_len, N)
    pred = np.zeros((N, N, seq_len))
    for lag in range(1, seq_len + 1):
        t = seq_len - lag
        pred[:, :, lag - 1] = S_reshaped[:, t, :]

    # 自环置零（如需保留可在配置里加开关）
    # mask = 1 - np.eye(N, dtype=pred.dtype)
    # pred = pred * mask[:, :, np.newaxis]

    return pred