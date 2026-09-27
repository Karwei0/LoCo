"""
LoCo baseline for root cause analysis (RCA) of anomalies in multivariate time
series, adapted to the evaluation protocol of the AERCA repository so that it can
be dropped in as a baseline on the real-world SWaT and MSDS datasets.

LoCo ("LOcal COvariance" style linear forecaster, see the `LoCo/` reference
implementation bundled with this repository) is a *supervised*, lightweight
one-step-ahead linear model.  It predicts the next time step

    x_hat(t) = W @ flatten( x[t-seq_len : t] )          (Linear_LoCo)
    x_hat(t) = W @ phi( flatten( x[t-seq_len : t] ) )   (Kernel_LoCo)

where ``phi`` is a Random Fourier Features (RFF) embedding of the lagged window.
The causal / root-cause structure is encoded in the learned weight matrix ``W``:
its columns are ordered as (lag ``0..seq_len-1``, variable ``0..num_vars-1``).

Root-cause scoring (mapped to AERCA's evaluation protocol)
----------------------------------------------------------
LoCo identifies anomalies by the one-step-ahead prediction residual.  After the
model is fitted on the *normal* training data, LoCo calibrates a set of
per-variable statistics on that same training data (this is the "threshold on
training data" step, ``calibrate_ad_params`` in the reference implementation):

    sigma_j    = std of the training residuals of variable j       (per variable)
    theta_spe  = ad_quantile of SPE = sum_j (residual_j / sigma_j)^2  (scalar)
    tau_j      = max(|q_{beta/2}(e_j)|, |q_{1-beta/2}(e_j)|)       (per variable)

At test time each time step ``t`` of an abnormal sequence is scored by the
standardized residual magnitude

    score_j(t) = | (x_j(t) - x_hat_j(t)) / sigma_j | ,

i.e. the same quantity LoCo uses to pick the most anomalous variable
(``i_star = argmax_j |e_j| / tau_j``).  A larger value means a larger deviation
from the learned normal dynamics, hence a more likely root cause, which matches
the "higher = more likely root cause" convention of ``utils.utils.topk`` and
``utils.utils.topk_at_step``.

Output format
-------------
Following ``models/aerca.py::AERCA._testing_root_cause``, this class produces a
per-variable, per-time-step root-cause score matrix of shape
``(T - seq_len, num_vars)`` for each test sequence, aligned with
``labels[seq_len:]``, and reports the same metrics (AC@k / Avg@k and
AC*@k / Avg*@k) computed by ``utils.utils.topk`` and ``utils.utils.topk_at_step``.

``seq_len`` is LoCo's own forecasting window (the analogue of AERCA's
``2 * window_size`` context): it is the number of past observations required to
produce a one-step prediction.  For SWaT the abnormal slices contain 20 leading
normal steps followed by 10 anomalous steps, so ``seq_len=20`` scores exactly the
10 anomalous steps; for MSDS each slice is ``2 * window_size + 1 = 3`` steps with
one anomalous step, so ``seq_len=2`` scores exactly that step.

Two variants are supported (``kernel=False`` -> Linear_LoCo, ``kernel=True`` ->
Kernel_LoCo), mirroring the two implementations of the reference LoCo code.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

from utils.utils import pot, topk, topk_at_step

import csv
import os
from datetime import datetime


from utils.utils import pot, topk, topk_at_step

# ------------------------------------------------------------------------------
class RFF:
    """Random Fourier Features approximation of a Gaussian (RBF) kernel.

    phi(x) = sqrt(2 / D) * cos(x * omega + bias),  omega ~ N(0, 1 / sigma^2),
    bias ~ Uniform(0, 2*pi).  Maps a flattened lag vector ``(NL,)`` to
    ``(NL * D,)``.
    """

    def __init__(self, N, L, D, sigma=1.0, seed=None, device='cpu'):
        self.N = N
        self.L = L
        self.D = D
        self.NL = N * L
        self.sigma = sigma
        device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)

        # omega ~ N(0, 1 / sigma^2)
        self.omega = torch.randn(self.NL, self.D, device=device) / self.sigma
        # bias ~ Uniform(0, 2*pi)
        self.bias = torch.rand(self.NL, self.D, device=device) * 2 * np.pi

    def __call__(self, x):
        """x: (B, NL) -> phi: (B, NL * D)."""
        B = x.shape[0]

        x_expanded = x.view(B, self.NL, 1)  # (B, NL, 1)
        omega = self.omega.unsqueeze(0)      # (1, NL, D)
        bias = self.bias.unsqueeze(0)        # (1, NL, D)

        # omega * x + bias: (B, NL, D)
        arg = x_expanded * omega + bias
        phi = torch.cos(arg) * np.sqrt(2.0 / self.D)

        return phi.view(B, -1)


class LinearLoCo(nn.Module):
    def __init__(self, num_vars, seq_len, topk):
        super(LinearLoCo, self).__init__()
        self.N = num_vars
        self.L = seq_len
        self.NL = self.N * self.L
        self.topk = topk
        self.linear = nn.Linear(self.NL, self.N, bias=False)
        self.init_weights()

    def init_weights(self, init_type='uniform', **kwargs):
        if init_type == 'uniform':
            a = kwargs.get('a', -0.00001)
            b = kwargs.get('b', 0.00001)
            nn.init.uniform_(self.linear.weight, a, b)
        else:
            raise NotImplementedError(f'init type {init_type} is not supported')

    def forward(self, x, *args):
        x = x.reshape(x.shape[0], -1)
        return self.linear(x).unsqueeze(1)

    @torch.no_grad()
    def topk_gradient_mask(self):
        """Keep only the top-``topk`` gradient entries per output row (sparsity)."""
        if self.linear.weight.grad is None:
            return
        new_grad = torch.zeros_like(self.linear.weight.grad)
        for i in range(self.N):
            abs_grad = torch.abs(self.linear.weight.grad[i])
            _, topk_idx = torch.topk(abs_grad, min(self.topk, len(abs_grad)), sorted=False)
            new_grad[i, topk_idx] = self.linear.weight.grad[i, topk_idx]
        self.linear.weight.grad = new_grad

    def get_causal_matrix(self):
        return torch.abs(self.linear.weight.data)


class KernelLoCo(nn.Module):
    def __init__(self, num_vars, seq_len, D, sigma, topk, device='cpu'):
        super(KernelLoCo, self).__init__()
        self.N = num_vars
        self.L = seq_len
        self.D = D
        self.NL = self.N * self.L
        self.NLD = self.NL * self.D
        self.topk = topk

        self.rff = RFF(self.N, self.L, self.D, sigma=sigma, seed=None, device=device)
        self.linear = nn.Linear(self.NLD, self.N, bias=False)
        self.init_weights()

    def init_weights(self, init_type='uniform', **kwargs):
        if init_type == 'uniform':
            a = kwargs.get('a', -0.00001)
            b = kwargs.get('b', 0.00001)
            nn.init.uniform_(self.linear.weight, a, b)
        else:
            raise NotImplementedError(f'init type {init_type} is not supported')

    def forward(self, x, *args):
        x = x.reshape(x.size(0), -1)
        phi = self.rff(x)
        return self.linear(phi).unsqueeze(1)

    @torch.no_grad()
    def topk_gradient_mask(self):
        """Keep the top-``topk`` lag groups per output row by group norm (sparsity)."""
        if self.linear.weight.grad is None:
            return
        new_grad = torch.zeros_like(self.linear.weight.grad)
        grad = self.linear.weight.grad
        grad_reshaped = grad.view(self.N, self.NL, self.D)
        new_grad_reshaped = new_grad.view(self.N, self.NL, self.D)

        for i in range(self.N):
            group_norms = torch.norm(grad_reshaped[i], dim=1)  # (NL,)
            _, topk_idx = torch.topk(group_norms, min(self.topk, self.NL), sorted=False)
            new_grad_reshaped[i, topk_idx] = grad_reshaped[i, topk_idx]

        self.linear.weight.grad = new_grad

    def get_causal_matrix(self):
        W = self.linear.weight.data                      # (N, NL*D)
        W_reshaped = W.view(self.N, self.NL, self.D)     # (N, NL, D)
        return torch.norm(W_reshaped, dim=2)             # (N, NL)



class LoCo:
    def __init__(self, num_vars, device, window_size, seq_len=20, kernel=False,
                 D=50, sigma=1.0, topk=None, lr=0.001, epochs=30,
                 loss_mode='L1', ad_quantile=0.92,
                 risk=1e-2, initial_level=0.9, num_candidates=100, data_name='', patience=3):
        """
        Args:
            num_vars: number of variables (p).
            device: torch device string (e.g. 'cuda' / 'cpu').
            window_size: AERCA window size.  Kept for interface parity with the
                other models; LoCo's own forecasting window is ``seq_len``, which
                is what determines the label alignment.
            seq_len: LoCo forecasting window length (number of lagged steps).
            kernel: if True use Kernel_LoCo (RFF features), else Linear_LoCo.
            D: RFF feature dimension per lag (kernel variant only).
            sigma: RFF Gaussian kernel bandwidth (kernel variant only).
            topk: sparsity of the ``topk_gradient_mask`` trick (defaults to num_vars).
            lr, epochs: optimizer learning rate and number of training epochs.
            loss_mode: 'L1' (default) or 'L2' prediction loss.
            ad_quantile: quantile used to derive the SPE anomaly threshold
                (``theta_spe``) from the training residuals.
            risk, initial_level, num_candidates: POT hyper-parameters for the
                root-cause metric (see ``utils.utils.pot``).
            data_name: dataset name, used to build the model name.
        """
        self.num_vars = num_vars
        self.device = torch.device(device) if isinstance(device, str) else device
        self.window_size = window_size
        self.seq_len = int(seq_len)
        self.kernel = bool(kernel)
        self.D = D
        self.rff_sigma = sigma
        self.topk = int(topk) if topk is not None else num_vars
        self.lr = lr
        self.epochs = epochs
        self.loss_mode = loss_mode
        self.ad_quantile = ad_quantile
        self.risk = risk
        self.initial_level = initial_level
        self.num_candidates = num_candidates
        self.data_name = data_name
        self.patience = int(patience)

        kernel_str = 'kernel' if self.kernel else 'linear'
        self.model_name = ('LoCo_' + kernel_str + '_' + data_name +
                           '_seqlen_' + str(self.seq_len))

        if self.kernel:
            self.model = KernelLoCo(num_vars, self.seq_len, self.D, self.rff_sigma,
                                    self.topk, device=self.device)
        else:
            self.model = LinearLoCo(num_vars, self.seq_len, self.topk)
        self.model.to(self.device)

        # Thresholds calibrated on the training data in fit().
        self.sigma = None       # (num_vars,)  per-variable residual std
        self.theta_spe = None   # scalar       SPE anomaly threshold
        self.tau_i = None       # (num_vars,)  per-variable residual bounds

    def _log_and_print(self, msg, *args):
        """Helper method to log and print testing results (mirrors AERCA)."""
        final_msg = msg.format(*args) if args else msg
        logging.info(final_msg)
        print(final_msg)

    def _save_rca_metrics(self, ac_at, avg_at, ac_star_at, avg_star):
        """将本次评估的参数与指标追加写入 log/{kernel_loco|linear_loco}_rca.csv"""
        model_name = 'kernel_loco' if self.kernel else 'linear_loco'
        os.makedirs('log', exist_ok=True)
        file_path = os.path.join('log', f'{model_name}_rca.csv')

        header = [
            'datetime', 'seq_len', 'D', 'sigma', 'topk', 'lr', 'epoch',
            'ad_quantile', 'num_candidates',
            'AC@1', 'AC@3', 'AC@5', 'AC@10', 'AVG@10',
            'AC*@1', 'AC*@10', 'AC*@100', 'AC*@500', 'AVG*@500'
        ]
        row = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            self.seq_len,
            self.D,
            self.rff_sigma,
            self.topk,
            self.lr,
            self.epochs,
            self.ad_quantile,
            self.num_candidates,
            ac_at[0], ac_at[1], ac_at[2], ac_at[3], avg_at,
            ac_star_at[0], ac_star_at[1], ac_star_at[2], ac_star_at[3], avg_star
        ]

        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    # --------------------------------------------------------------------------
    # Data preparation: sliding windows (seq_len lags -> next step) on normal data.
    # --------------------------------------------------------------------------
    def _build_dataset(self, xs_normal):
        X_list, Y_list = [], []
        for x in xs_normal:
            x = np.asarray(x, dtype=float)
            if x.shape[0] <= self.seq_len:
                continue
            # (T - seq_len, seq_len + 1, num_vars): [lag..., next]
            windows = sliding_window_view(x, (self.seq_len + 1, self.num_vars))[:, 0, :, :]
            X_list.append(windows[:, :-1, :].reshape(-1, self.seq_len * self.num_vars))
            Y_list.append(windows[:, -1, :])

        if not X_list:
            raise ValueError('No training windows could be built: seq_len is too '
                             'large for the normal data segments.')

        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32))

    def _loss(self, pred, target):
        if self.loss_mode == 'L1':
            return torch.mean(torch.abs(pred - target))
        return torch.mean((pred - target) ** 2)

    # --------------------------------------------------------------------------
    # Training: minimize one-step-ahead prediction loss with the top-k gradient
    # sparsity trick (mirrors LoCo's train loop, minus checkpointing / LR schedule).
    # --------------------------------------------------------------------------
    def _train(self, X, Y):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n = X.shape[0]
        batch_size = 512

        # ---- deterministic train / val split for early stopping ----
        val_ratio = 0.1
        n_val = max(1, int(n * val_ratio))
        if n_val >= n:
            n_val = max(1, n // 5)
        g = torch.Generator().manual_seed(42)
        perm_all = torch.randperm(n, generator=g)
        val_idx = perm_all[:n_val]
        train_idx = perm_all[n_val:]
        if train_idx.numel() == 0:
            # Fallback: too few samples for a split, train on everything, no early stop
            X_train, Y_train = X, Y
            X_val, Y_val = X, Y
        else:
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]
        n_train = X_train.shape[0]
        # ------------------------------------------------------------

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        stopped_epoch = self.epochs

        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                xb = X_train[idx].to(self.device)
                yb = Y_train[idx].to(self.device)

                optimizer.zero_grad()
                out = self.model(xb)         # (B, 1, num_vars)
                pred = out[:, -1, :]         # (B, num_vars)
                loss = self._loss(pred, yb)
                loss.backward()
                self.model.topk_gradient_mask()
                optimizer.step()

                epoch_loss += loss.item() * idx.shape[0]
            epoch_loss /= max(1, n_train)

            # ---- validation loss ----
            self.model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for start in range(0, X_val.shape[0], batch_size):
                    xb = X_val[start:start + batch_size].to(self.device)
                    yb = Y_val[start:start + batch_size].to(self.device)
                    out = self.model(xb)
                    pred = out[:, -1, :]
                    val_loss_sum += self._loss(pred, yb).item() * xb.shape[0]
            val_loss = val_loss_sum / max(1, X_val.shape[0])

            if (epoch + 1) % max(1, self.epochs // 10) == 0 or epoch == 0:
                print('LoCo epoch {}/{} | train loss: {:.6f} | val loss: {:.6f}'.format(
                    epoch + 1, self.epochs, epoch_loss, val_loss))

            # ---- early stopping check ----
            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    stopped_epoch = epoch + 1
                    print('LoCo early stopping at epoch {}/{} '
                          '(best val loss: {:.6f}, patience={})'.format(
                              stopped_epoch, self.epochs, best_val_loss, self.patience))
                    break

        # Restore the best-seen weights so that `_calibrate_ad_params` uses the
        # model that generalized best on held-out normal data.
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.train()
    # --------------------------------------------------------------------------
    # Threshold calibration on TRAINING data (mirrors LoCo's calibrate_ad_params).
    # --------------------------------------------------------------------------
    def _calibrate_ad_params(self, X, Y):
        self.model.eval()
        residuals = []
        batch_size = 128
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                xb = X[start:start + batch_size].to(self.device)
                yb = Y[start:start + batch_size].to(self.device)
                out = self.model(xb)
                pred = out[:, -1, :]
                residuals.append((yb - pred).cpu())
        residuals = torch.cat(residuals, dim=0)  # (M, num_vars)

        # per-variable residual std (used to standardize test residuals)
        self.sigma = residuals.std(dim=0).reshape(-1)
        self.sigma[self.sigma == 0] = 1.0

        # standardized residuals and the SPE anomaly threshold
        e = residuals / self.sigma
        spe = (e ** 2).sum(dim=-1)
        self.theta_spe = torch.quantile(spe, self.ad_quantile)

        # per-variable residual bounds (used by LoCo's root-cause tracer)
        beta = 0.01
        lower = torch.quantile(e, beta / 2, dim=0)
        upper = torch.quantile(e, 1 - beta / 2, dim=0)
        self.tau_i = torch.max(torch.abs(lower), torch.abs(upper))

    def fit(self, xs_normal):
        """Fit the forecaster on the normal data and calibrate thresholds."""
        X, Y = self._build_dataset(xs_normal)
        self._log_and_print('LoCo ({}) fitting on {} training windows (seq_len={}, num_vars={})',
                            'kernel' if self.kernel else 'linear',
                            X.shape[0], self.seq_len, self.num_vars)
        self._train(X, Y)
        self._calibrate_ad_params(X, Y)
        self._log_and_print('LoCo thresholds calibrated on training data (ad_quantile={})',
                            self.ad_quantile)

    @torch.no_grad()
    def _predict_batch(self, x_flat):
        """x_flat: (M, seq_len * num_vars) numpy -> predicted next step (M, num_vars)."""
        xb = torch.tensor(np.asarray(x_flat, dtype=np.float32)).to(self.device)
        out = self.model(xb)
        return out[:, -1, :].cpu().numpy()

    def _score_sample(self, x):
        """Compute the per-variable, per-time-step root-cause score for one sequence.

        For each predictable time step ``t`` in ``[seq_len, T)`` the model predicts
        ``x[t]`` from ``x[t-seq_len:t]`` and the standardized residual magnitude

            score_j(t) = | (x_j(t) - x_hat_j(t)) / sigma_j |

        is returned (higher = more anomalous = more likely root cause).

        Args:
            x: (T, num_vars) numpy array.

        Returns:
            score matrix of shape (T - seq_len, num_vars).
        """
        x = np.asarray(x, dtype=float)
        T = x.shape[0]
        L = self.seq_len
        if T <= L:
            return np.zeros((0, self.num_vars))

        windows = sliding_window_view(x, (L + 1, self.num_vars))[:, 0, :, :]  # (T-L, L+1, N)
        X = windows[:, :-1, :].reshape(-1, L * self.num_vars)                  # (T-L, L*N)
        Y_true = windows[:, -1, :]                                             # (T-L, N)

        sigma = self.sigma.cpu().numpy()                                       # (N,)
        scores = np.empty((T - L, self.num_vars))
        batch_size = 4
        for start in range(0, T - L, batch_size):
            pred = self._predict_batch(X[start:start + batch_size])            # (bs, N)
            residual = Y_true[start:start + batch_size] - pred
            e = residual / sigma
            scores[start:start + batch_size] = np.abs(e)
        return scores

    def _testing_root_cause(self, xs, labels):
        """Evaluate root cause localization (mirrors AERCA._testing_root_cause)."""
        self.model.eval()

        score_sample_list = []
        score_list = []
        for i in range(len(xs)):
            s = self._score_sample(xs[i])
            score_sample_list.append(s)
            score_list.append(s)

        score_all = np.concatenate(score_list, axis=0).reshape(-1, self.num_vars)
        self._log_and_print('=' * 50)
        threshold = np.empty(self.num_vars)
        for j in range(self.num_vars):
            pot_val, _ = pot(score_all[:, j], self.risk, self.initial_level, self.num_candidates)
            threshold[j] = pot_val

        k_all = []
        k_at_step_all = []
        for i in range(len(xs)):
            s = score_sample_list[i]
            label = np.asarray(labels[i])
            lbl = label[self.window_size * 2:]
            m = min(s.shape[0], lbl.shape[0])
            if m == 0:
                continue
            s = s[:m]
            lbl = lbl[:m]
            k_lst = topk(s, lbl, threshold)
            k_at_step = topk_at_step(s, lbl)
            k_all.append(k_lst)
            k_at_step_all.append(k_at_step)

        if len(k_all) == 0:
            self._log_and_print('No test samples longer than seq_len; skipping root cause evaluation.')
            return

        k_all = np.array(k_all).mean(axis=0)
        k_at_step_all = np.array(k_at_step_all).mean(axis=0)
        ac_at = [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]]
        self._log_and_print('Root cause analysis AC@1: {:.5f}', ac_at[0])
        self._log_and_print('Root cause analysis AC@3: {:.5f}', ac_at[1])
        self._log_and_print('Root cause analysis AC@5: {:.5f}', ac_at[2])
        self._log_and_print('Root cause analysis AC@10: {:.5f}', ac_at[3])
        self._log_and_print('Root cause analysis Avg@10: {:.5f}', np.mean(k_at_step_all))

        ac_star_at = [k_all[0], k_all[9], k_all[99], k_all[499]]
        self._log_and_print('Root cause analysis AC*@1: {:.5f}', ac_star_at[0])
        self._log_and_print('Root cause analysis AC*@10: {:.5f}', ac_star_at[1])
        self._log_and_print('Root cause analysis AC*@100: {:.5f}', ac_star_at[2])
        self._log_and_print('Root cause analysis AC*@500: {:.5f}', ac_star_at[3])
        self._log_and_print('Root cause analysis Avg*@500: {:.5f}', np.mean(k_all))

        self._save_rca_metrics(
            ac_at=ac_at,
            avg_at=np.mean(k_at_step_all),
            ac_star_at=ac_star_at,
            avg_star=np.mean(k_all)
        )