import numpy as np
import torch

class RootCauseAnalyzer:
    def __init__(self, model, args, sigma, tau_i, theta_spe, parents_candidates, device):
        self.model = model
        self.args = args
        self.sigma = sigma          # (N,)
        self.tau_i = tau_i          # (N,)
        self.theta_spe = theta_spe
        self.parents_candidates = parents_candidates  # list of list, each (j, tau, strength)
        self.device = device
        self.model.eval()
        self.max_depth = 2 * args.seq_len # 可适当调大

    def analyze(self, idx, x, y, y_hat, sample_data):
        e = (y - y_hat) / self.sigma.cpu().numpy()   # (N,)
        spe = np.sum(e ** 2)
        if spe <= self.theta_spe.cpu().numpy():
            return None

        A = np.abs(e) > self.tau_i.cpu().numpy()
        if not A.any():
            return None
        i_star = np.argmax(np.abs(e) / self.tau_i.cpu().numpy())

        all_roots = []

        # 这里只处理当前时间步 偏离最大的变量 （认定为异常变量） 
        # 但是也可以对其他变量进行根因 （其他大于阈值的变量）也作为根因进行分析
        # 但是为了简洁，这里就只处理最明显的变量
        roots = self._trace(i_star, idx, x, y, y_hat, sample_data,
                            path=[(i_star, 0, 0.0)], accum_contrib=0.0, total_lag=0)
        all_roots.extend(roots)
        # for i in np.where(A)[0]:
        #     print(f'[debug] i: {i}')
        #     roots = self._trace(i, idx, x, y, y_hat, sample_data,
        #                         path=[(i, 0, 0.0)], accum_contrib=0.0, total_lag=0)
        #     all_roots.extend(roots)
        
        # 去重（按变量和滞后合并，保留贡献绝对值最大的）
        root_dict = {}
        for r in all_roots:
            key = (r['var'], r['lag'])
            if key not in root_dict or abs(r['contrib']) > abs(root_dict[key]['contrib']):
                root_dict[key] = r

        # 计算每个根因的归因分数
        for key, r in root_dict.items():
            r['attr'] = self._compute_attr(idx, x, y, y_hat, r['var'], r['lag'], sample_data)

        sorted_roots = sorted(root_dict.values(), key=lambda x: x['attr'], reverse=True)

        return {
            'idx': idx,
            'spe': spe,
            'i_star': i_star,
            'root_causes': sorted_roots,
            'e': e,
            'A': A
        }

    def _trace(self, i, idx, x, y, y_hat, sample_data, path, accum_contrib, total_lag=0, depth=0):
        if depth > self.max_depth:
            return [{'var': i, 'lag': total_lag, 'path': path, 'contrib': accum_contrib}]
        parents = self.parents_candidates[i]
        if not parents:
            return [{'var': i, 'lag': total_lag, 'path': path, 'contrib': accum_contrib}]

        x_base = x.copy()
        batch_inputs = []
        valid_parents = []
        for (j, tau, strength) in parents:
            # print(f"[debug] idx={idx}, tau={tau}, target={idx-tau}, len={len(sample_data)}")
            # if idx - tau < 0 or idx - tau >= len(sample_data):
            #     continue
            y_hat_arr = sample_data[idx - tau]['y_hat']
            # 安全检查：如果 j 超出 y_hat 长度，跳过该父节点并警告
            if j >= len(y_hat_arr):
                # print(f"Warning: j={j} out of range for y_hat length {len(y_hat_arr)}, skipping parent")
                continue
            ref = y_hat_arr[j]
            x_mod = x_base.copy()
            x_mod[-tau, j] = ref
            batch_inputs.append(x_mod)
            valid_parents.append((j, tau))

        if not valid_parents:
            return [{'var': i, 'lag': total_lag, 'path': path, 'contrib': accum_contrib}]

        M = len(valid_parents)
        batch_inputs = np.array(batch_inputs)  # (M, T, N)
        # print(f'[debug] batch_inputs shape: {batch_inputs.shape}')
        # print(f'[debug] y_hat shape: {y_hat.shape}')
        # print(f'[debug] y shape: {y.shape}')
        # print(f'[debug] x shape: {x.shape}')
        

        with torch.no_grad():
            inputs_t = torch.tensor(batch_inputs, dtype=torch.float32, device=self.device)
            y_t = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(M, 1)
            outputs = self.model(inputs_t, y_t)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            y_hat_mod = outputs[:, -1, :]          # (M, N)
            y_hat_orig = torch.tensor(y_hat, dtype=torch.float32, device=self.device)  # (N,)

        contributions = []
        for m, (j, tau) in enumerate(valid_parents):
            c = (y_hat_orig[i] - y_hat_mod[m, i]) / self.sigma[i]  # 标准化贡献
            contributions.append((j, tau, c.item()))

        # 筛选显著贡献
        eta = 0.2 * self.tau_i[i].item()
        significant = [(j, tau, c) for (j, tau, c) in contributions if abs(c) >= eta]
        if not significant:
            return [{'var': i, 'lag': total_lag, 'path': path, 'contrib': accum_contrib}]

        root_causes = []
        for (j, tau, c) in significant:
            # 检查父节点是否异常
            y_parent = sample_data[idx - tau]['y']
            y_hat_parent = sample_data[idx - tau]['y_hat']
            sigma_j = self.sigma[j].item()
            tau_j = self.tau_i[j].item()
            e_parent = (y_parent - y_hat_parent) / sigma_j
            
            # 如果绝对滞后超出窗口长度，不再递归，直接视为根因
            if total_lag + tau > self.args.seq_len:
                root_causes.append({
                    'var': j,
                    'lag': total_lag + tau,
                    'path': path + [(j, total_lag + tau, c)],
                    'contrib': accum_contrib + c
                })
            elif abs(e_parent[j]) > tau_j:
                # 父节点异常，继续递归
                sub_path = path + [(j, total_lag + tau, c)]
                sub_roots = self._trace(j, idx - tau,
                                        sample_data[idx - tau]['x'],
                                        sample_data[idx - tau]['y'],
                                        sample_data[idx - tau]['y_hat'],
                                        sample_data, sub_path,
                                        accum_contrib + c,
                                        total_lag + tau,
                                        depth + 1)
                root_causes.extend(sub_roots)
            else:
                # 父节点正常，视为根因
                root_causes.append({
                    'var': j,
                    'lag': total_lag + tau,
                    'path': path + [(j, total_lag + tau, c)],
                    'contrib': accum_contrib + c 
                })
        return root_causes

    def _compute_attr(self, idx, x, y, y_hat, root_var, root_lag, sample_data):
        if root_var == 0: # 根因为自变量
            return 0.0
        if root_lag > self.args.seq_len:
            return 0.0
        if idx - root_lag < 0 or idx - root_lag >= len(sample_data):
            return 0.0
        ref = sample_data[idx - root_lag]['y_hat'][root_var]
        x_mod = x.copy()
        x_mod[-root_lag, root_var] = ref
        x_mod_t = torch.tensor(x_mod, dtype=torch.float32, device=self.device).unsqueeze(0)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            out = self.model(x_mod_t, y_t)
            if isinstance(out, tuple):
                out = out[0]
            y_hat_mod = out[0, -1, :].cpu().numpy()

        e_orig = (y - y_hat) / self.sigma.cpu().numpy()
        e_mod = (y - y_hat_mod) / self.sigma.cpu().numpy()
        spe_orig = np.sum(e_orig ** 2)
        spe_mod = np.sum(e_mod ** 2)
        attr = spe_orig - spe_mod   # 应为正
        # return max(0.0, attr)
        return abs(attr)