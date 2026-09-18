import torch
import numpy as np
import torch.nn as nn
from sklearn.linear_model import SGDRegressor, Lasso
from sklearn.preprocessing import StandardScaler
from experiments.exp_basic import Exp_Basic
from experiments.exp_detection import Exp_Detection, WeightedL1Loss
from utils.metrics import metric, combine_all_evaluation_score
from utils.tools import EarlyStopping
import os

def _select_criterion():
     criterion = nn.L1Loss(reduction='mean')
     return criterion

class Exp_Detection_Lasso(Exp_Detection):
     def __init__(self, args):
          super().__init__(args)
          self.alpha = args.lamda1

     def _build_model(self):
          model = self.model_dict[self.args.model].Model(self.args).float()

          if self.args.use_multi_gpu and self.args.use_gpu:
               model == nn.DataParallel(model, device_ids=self.args.device_ids)
          
          return model

     def _prepare_data(self, batch_x, batch_y):
          X = batch_x.reshape(batch_x.shape[0], -1).detach().cpu().numpy()
          y = batch_y[:, -1, :].cpu().numpy()
          return X, y
     
     def train(self, setting=None):
          train_data, train_loader = self._get_data(flag='train')
          vali_data, vali_loader = self._get_data(flag='val')
          test_data, test_loader = self._get_data(flag='test')

          criterion = WeightedL1Loss(alpha=self.args.alpha, loss_mode=self.args.loss_mode)

          X_list = []
          y_list = []
          for batch_x, batch_y, *_ in train_loader:
               # batch_x: [B, seq_len, input_dim]
               B = batch_x.shape[0]
               X_batch = batch_x.reshape(B, -1).numpy() 
               # X_batch = batch_x.permute(0, 2, 1).reshape(B, -1).numpy()  # [B, seq_len * input_dim] 
               # batch_y: [B, pred_len, input_dim]，pred_len 一般为 1
               y_batch = batch_y[:, 0, :].numpy()        # [B, input_dim]
               X_list.append(X_batch)
               y_list.append(y_batch)
          X_train = np.concatenate(X_list, axis=0)
          y_train = np.concatenate(y_list, axis=0)

          print(f'fitting lasso with alpha={self.alpha} on {X_train.shape[0]} samples ...')
          print(f'X_train.shape: {X_train.shape}')
          lasso = Lasso(alpha=self.alpha, fit_intercept=False, random_state=41)

          lasso.fit(X_train, y_train)
          coef = lasso.coef_

          self.model.set_weights(coef)
          # print(f'[debug] coef: {coef}')

          path = os.path.join(self.args.checkpoints, setting)
          os.makedirs(path, exist_ok=True)
          torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))

          # criterion = self._select_criterion()
          vali_loss = self.vali(vali_data, vali_loader, criterion)
          test_loss = self.vali(test_data, test_loader, criterion)
          print(f"Train finished. Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")

          self.calibrate_ad_params(train_loader)

          return
     
     def calibrate_ad_params(self, train_loader):
          """
          Call this after the training converges to calculate the statistics needed
            for anomaly detection based on the training set.
          """
          self.model.eval()
          all_residuals = []
          with torch.no_grad():
               for batch_x, batch_y, *_ in train_loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    pred = self.model(batch_x).permute(0, 2, 1)
                    all_residuals.append((batch_y - pred).cpu())
                    # print(f'[debug] batch_y.shape: {batch_y.shape}')
                    # print(f'[debug] pred.shape: {pred.shape}')
          residuals = torch.cat(all_residuals, dim=0)
          # print(f'[debug] residuals.shape: {residuals.shape}')
          self.sigma = residuals.std(dim=0).reshape(-1) # duplicate
          # print(f'[debug] self.sigma.shape: {self.sigma.shape}')
          self.sigma[self.sigma == 0] = 1.0
          
          
          e = residuals / self.sigma
          spe = (e ** 2).sum(dim=-1)
          # print(f'[debug] spe.shape {spe.shape}')
          # print(f'[debug] e.shape: {e.shape}')
          print(f'[debug] self.sigma.shape: {self.sigma.shape}')
          print(f'[debug] residuals.shape: {residuals.shape}')
          self.theta_spe = torch.quantile(spe, self.args.ad_quantile)
          # print(f'[debug] self.theta_spe: {self.theta_spe.shape}, self.theta_spe: {self.theta_spe}')

          self.causal_matrix = self.model.get_causal_matrix()
     
     def anomaly_detection_test(self, test_loader):
          self.model.eval()

          pred_labels = []
          true_labels = []
          pred_scores = []
          e_list = []


          with torch.no_grad():
               for batch_x, batch_y, label in test_loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    y_hat = self.model(batch_x).permute(0, 2, 1)
                    residual = batch_y - y_hat
                    e = residual / self.sigma.to(batch_y.device)
                    # print(f'[debug] e.shape: {e.shape}')
                    spe = (e**2).sum(dim=2)
                    # print(f'[debug] spe.shape: {spe.shape}')
                    is_anom = (spe > self.theta_spe).int()
                    # print(f'[debug] is_anom.shape: {is_anom.shape}')

                    true_labels.append(label)
                    pred_scores.append(spe.cpu().numpy())
                    pred_labels.append(is_anom.cpu().numpy())
                    e_list.append(e.cpu().numpy())
          pred_scores = np.concatenate(pred_scores, axis=0).reshape(-1)
          pred_labels = np.concatenate(pred_labels, axis=0).reshape(-1)
          true_labels = np.concatenate(true_labels, axis=0).reshape(-1)
          e_list = np.concatenate(e_list, axis=0)

          print(f'[debug] pred_labels: {pred_labels}, pred_labels.shape: {pred_labels.shape}')
          print(f'[debug] true_labels: {true_labels}, true_labels.shape: {true_labels.shape}')
          res = combine_all_evaluation_score(pred_labels, true_labels, pred_scores)
          return res