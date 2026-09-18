from typing import List

from experiments.exp_basic import Exp_Basic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

import os, time, random, logging, shutil
import numpy as np

from fvcore.nn import FlopCountAnalysis

from utils.metrics import combine_all_evaluation_score

import warnings
warnings.filterwarnings("ignore")

from data_provider.loader_provider import get_data_loader
from utils.metrics import *
from utils.tools import (EarlyStopping, adjust_learning_rate, visual, write_into_xls, compute_gradient_norm,
                         find_most_recently_modified_subfolder, compare_prefix_before_third_underscore, compute_weights)

from utils.RootCauseAnalyzer import RootCauseAnalyzer

def compute_model_stats(model, args, num_iterations=50):
     assert num_iterations > 10, 'num_iterations must be greater than 10'
     if not args.model_stats_mode:
          print('No compute_model_stats because model_stats_mode is False')
          return False
     logging.getLogger('fvcore').setLevel(logging.ERROR)

     if not torch.cuda.is_available():
          print('CUDA is not available. Cannot measure GPU memory and timings.')
          return False
     
     device = torch.device('cuda')

     input_size = (1, args.seq_len, args.input_dim)
     inputs = torch.randn(input_size).to(device)

     params = sum(p.numel() for p in model.parameters() if p.requires_grad)
     print(f'Parameters: {params:.3f}')

     # use fvcore to get flops
     flops = FlopCountAnalysis(model, inputs)
     flops = flops.total() / 1e6

     print(f'FLOPs: {flops:.3f}M')

     # training and inferring
     inputs = torch.randn(args.batch_size, args.seq_len, args.input_dim).to(device)

     # inference, make gpu memory more precise
     torch.cuda.reset_peak_memory_stats()
     inference_times = []

     for i in range(num_iterations):
          start_time = time.time()
          if args.use_amp:
               with torch.cuda.amp.autocast():
                    _ = model(inputs)
          else:
               with torch.no_grad():
                    _ = model(inputs)

          inference_times.append(time.time() - start_time)
     avg_inference_time = np.mean(inference_times[-10:])
     inference_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

     print(f'Inference Time / iter: {avg_inference_time * 1000:.3f} ms')
     print(f'Inference Memory Usage: {inference_memory:.3f} MB')

     # training
     criterion = WeightedL1Loss(args.lossfun_alpha, args.loss_mode)
     targets = torch.randn(args.batch_size, args.seq_len, args.enc_in).to(device)

     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
     training_times = []
     torch.cuda.reset_peak_memory_stats()
     for _ in range(num_iterations):
          model.train()
          start_time = time.time()
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, targets)

          scaler = None
          if args.use_amp:
               scaler = torch.cuda.amp.GradScaler()
          
          if args.use_amp:
               scaler.scale(loss).backward()
               scaler.step(optimizer)
               scaler.update()
          else:
               loss.backward()
               optimizer.step()
          
          training_times.append(time.time() - start_time)
     avg_training_time = np.mean(training_times[-10:])
     training_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

     with open('model_stats.txt', 'a') as f:
          f.write('============================ stats {args.model_id_ori}============================= ' + '\n')
          args_dict = vars(args)
          for k, v in sorted(args_dict.items()):
               f.writer(f'{k}: {v}, ')
          f.write('\n\n')
     f.write(f"\tParameters(M): {params:.3f}\n")
     f.write(f"\tFLOPS(M): {flops:.3f}\n")
     f.write(f"\tTraining Time / iter: {avg_training_time * 1000:.3f} ms\n")
     f.write(f"\tTraining Memory Usage: {training_memory:.3f} MB\n")
     f.write(f"\tInference Time / iter: {avg_inference_time * 1000:.3f} ms\n")
     f.write(f"\tInference Memory Usage: {inference_memory:.2f} MB\n\n\n")

     # save to folder best_results
     best_log_dataset_path = 'best_results'
     best_log_dataset_txt = os.path.join(best_log_dataset_path, args.model_id_ori + '_stats.txt')
     with open(best_log_dataset_txt, 'a') as f:
          f.write(f'============================ stats {args.model_id_ori}============================= ' + '\n')
          args_dict = vars(args)
          for k, v in sorted(args_dict.items()):
               f.write(f'{k}: {v}, ')
          f.write('\n\n')
          f.write(f"\tParameters(M): {params:.3f}\n")
          f.write(f"\tFLOPS(M): {flops:.3f}\n")
          f.write(f"\tTraining Time / iter: {avg_training_time * 1000:.3f} ms\n")
          f.write(f"\tTraining Memory Usage: {training_memory:.3f} MB\n")
          f.write(f"\tInference Time / iter: {avg_inference_time * 1000:.3f} ms\n")
          f.write(f"\tInference Memory Usage: {inference_memory:.2f} MB\n\n\n")

     # 打印结果

     print(f"Training Time / iter: {avg_training_time * 1000:.3f} ms")
     print(f"Training Memory Usage: {training_memory:.3f} MB")

     return True

class WeightedL1Loss:
     def __init__(self, alpha, loss_mode='L1'):
          self.alpha = alpha
          self.loss_mode = loss_mode
          if self.loss_mode == 'L1':
               self.loss_fun = nn.L1Loss(reduction='none')
          elif self.loss_mode == 'L2':
               self.loss_fun = nn.MSELoss(reduction='none')
          elif self.loss_mode == 'L1_L2':
               self.loss_fun = nn.MSELoss(reduction='none')
     def __call__(self, pred, gt):
          #[b, l, n]
          if pred.ndim == 1:
               mask = torch.isnan(gt)
               if torch.any(mask):
                    pred, gt = pred[~mask], gt[~mask]

               loss_fun = nn.L1Loss(reduction='mean')
               weightedLoss = loss_fun(pred, gt)
          else:
               L = pred.shape[1]
               weights = (torch.tensor([(i + 1) ** (-self.alpha) for i in range(L)]).unsqueeze(dim=0).unsqueeze(dim=-1)
                       .to(pred.device))
               if self.loss_mode in ['L1', 'L2']:
                    loss_vec = self.loss_fun(pred, gt)
                    weightedLoss = torch.mean(loss_vec * weights)
               elif self.loss_mode == 'L1_L2':
                    loss_vec = self.loss_fun1(pred, gt)
                    loss_vec2 = self.loss_fun2(pred, gt)
                    weightedLoss = torch.mean(loss_vec * weights + loss_vec2 * weights)
               else:
                    raise NotImplementedError
          return weightedLoss 

     def _select_criterion():
          criterion = nn.L1Loss(reduction='mean')
          return criterion

     def _select_mse_criterion():
          criterion = nn.MSELoss(reduction='mean')
          return criterion


class Exp_Detection(Exp_Basic):
     def __init__(self, args):
          super().__init__(args)
          self.args = args
          self.imp_mode = args.task_name == 'imputation'
          # self.eval_flag = args.eval_flag
          self.resume_training = args.resume_training
          self.resmue_epoch = args.resume_epoch
          self.folder_path = args.folder_path
          if not os.path.exists(self.folder_path):
               os.makedirs(self.folder_path, exist_ok=True)

          # for root cause analysis
          self.sigma = None
          self.tau_i = None
          self.theta_spe = None
          self.parents_candidates = None

     def _build_model(self):
          model = self.model_dict[self.args.model].Model(self.args).float()

          if self.args.use_multi_gpu and self.args.use_gpu:
               model == nn.DataParallel(model, device_ids=self.args.device_ids)
          
          return model
     
     def compute_model_stats(self):
          if self.args.model_stats_mode:
               compute_model_stats(self.args)
          
     def _get_data(self, flag='train', test_batch_size=None):
          data_set, data_loader = get_data_loader(self.args.root_path, self.args.batch_size, 
                                                  seq_len=self.args.seq_len, pred_len=self.args.pred_len, 
                                                  mode=flag, dataset=self.args.dataset, num_workers=self.args.num_workers)
          return data_set, data_loader

     def _select_optimizer(self):
          model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
          return model_optim
     
     def vali(self, vali_data=None, vali_loader=None, criterion=None):
          total_loss = []
          self.model.eval()
          with torch.no_grad():
               for i, (batch_x, batch_y, *_) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)

                    # batch_x_mark = batch_x_mark.float().to(self.device)
                    # batch_y_mark = batch_y_mark.float().to(self.device)

                    B, T, N = batch_x.shape
                    batch_y = batch_y.float().to(self.device)

                    mask_input = None
                    if self.args.use_amp:
                         with torch.cuda.amp.autocast():
                              # TODO: MODIFY
                              outputs = self.model(batch_x, batch_y)
                    else:
                         outputs = self.model(batch_x, batch_y)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)

                    total_loss.append(loss.item())
               total_loss = np.average(total_loss)
               self.model.train()
               return total_loss
     
     def train(self, setting=None):
          train_data, train_loader = self._get_data(flag='train')
          vali_data, vali_loader = self._get_data(flag='val')
          test_data, test_loader = self._get_data(flag='test')

          path = os.path.join(self.args.checkpoints, setting)
          if not os.path.exists(path):
               os.makedirs(path, exist_ok=True)
          
          time_now = time.time()

          train_steps = len(train_loader)
          early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                         save_every_epoch=self.args.save_every_epoch)
          model_optim = self._select_optimizer()
          criterion = WeightedL1Loss(alpha=self.args.alpha, loss_mode=self.args.loss_mode)

          scaler = None
          if self.args.use_amp:
               scaler = torch.cuda.amp.GradScaler()
          
          if self.resume_training and self.resmue_epoch > 0:
               full_folder, new_setting = find_most_recently_modified_subfolder(self.args.checkpoints,
                                                                             file_name='checkpoint.pth',
                                                                             contain_str=[self.args.model_id_ori,
                                                                                            self.args.model])
               if compare_prefix_before_third_underscore(full_folder, new_setting, num=3):
                    print(f'loading model from {full_folder}')
                    self.model.load_stat_dict(torch.load(os.path.join(full_folder, 'checkpoint.pth')))
                    shutil.copy(os.path.join(full_folder, 'checkpoint.pth'), path)
               else:
                    raise  ValueError('No checkpoint folder found. Please check...')

               current_val_loss = self.vali(vali_data, vali_loader, criterion)
               early_stopping.best_score = -current_val_loss
               early_stopping.val_loss_min = current_val_loss

          start_epoch = self.resmue_epoch if self.resume_training else 0
          if self.resume_training:
               print('Restore the learning rate...')
               adjust_learning_rate(model_optim, start_epoch, self.args)
          
          scheduler = None
          if self.args.lradj == 'TST':
               scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim, 
                                                   steps_per_epoch=train_steps,
                                                   pct_start=self.args.pct_start,
                                                   epochs=self.args.train_epochs,
                                                   max_lr=self.args.learning_rate)
               
          for epoch in range(start_epoch, self.args.train_epochs):
               iter_count = 0
               train_loss = []

               self.model.train()
               epoch_time = time.time()
               for i, (batch_x, batch_y, *_) in enumerate(train_loader):
                    iter_count += 1
                    batch_x = batch_x.float().to(self.device)

                    # batch_x_mark = batch_x_mark.float().to(self.device)
                    # batch_y_mark = batch_y_mark.float().to(self.device)

                    B, T, N = batch_x.shape
                    batch_y = batch_y.float().to(self.device)

                    if self.args.efficient_training:
                         _, _, N = batch_x.shape
                         # TODO: check input_dim or enc_in
                         if N > self.args.input_dim:
                             index = np.stack(random.sample(range(N), self.args.input_dim))
                             batch_x = batch_x[:, :, index]
                             batch_y = batch_y[:, :, index]
                    
                    mask_input = None
                    if self.args.use_amp:
                         with (torch.cuda.amp.autocast()):
                              # outputs0 = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark, mask=mask_input)
                              outputs0 = self.model(batch_x, batch_y)
                              
                              if isinstance(outputs0, tuple):
                                   outputs = outputs0[0]
                              else:
                                   outputs = outputs0

                              f_dim = -1 if self.args.features == 'MS' else 0
                              outputs = outputs[:, -self.args.pred_len:, f_dim:]
                              batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                              if isinstance(outputs0, tuple) and len(outputs0) >= 3:
                                   dec_out_inter = outputs0[2]
                                   if dec_out_inter:
                                        if isinstance(dec_out_inter, List):
                                             loss1_vec = torch.stack([criterion(seq, batch_y) 
                                                                      for seq in dec_out_inter])
                                             weights = compute_weights(self.args.alpha, len(loss1_vec), 
                                                                       self.args.git_multi_stage + 1, multiple_flag=self.imp_mode).to(self.device)
                                        else:
                                             loss1 = criterion(dec_out_inter, batch_y)

                                        loss = loss + self.args.lambda1 * loss1

                              loss = criterion(outputs, batch_y)

                              train_loss.append(loss.item())
                    else:
                         # outputs0 = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark, mask=mask_input)
                         outputs0 = self.model(batch_x, batch_y)
                         # print(f'[debug] outputs0.shape: {len(outputs0)}')
                         if isinstance(outputs0, tuple):
                              outputs = outputs0[0]
                         else:
                              outputs = outputs0
                         f_dim = -1 if self.args.features == 'MS' else 0
                         # print(f'[debug] outputs.shape: {outputs.shape}')
                         outputs = outputs[:, -self.args.pred_len:, f_dim:]
                         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                         loss = criterion(outputs, batch_y)

                         if isinstance(outputs0, tuple) and len(outputs0) >= 2:
                              dec_out_inter = outputs0[2]
                              if dec_out_inter:
                                   if isinstance(dec_out_inter, List):
                                        loss1_vec = torch.stack([criterion(seq, batch_y) for seq in dec_out_inter])
                                        weights = (compute_weights(self.args.alpha, len(loss1_vec), self.args.git_multi_stage + 1, multiple_flag=self.imp_mode).to(self.device))
                                        loss1 = (loss1_vec * weights).sum()
                                   else:
                                        loss1 = criterion(dec_out_inter, batch_y)
                                   loss = loss + self.args.lambda1 * loss1
                         if torch.isnan(loss):
                              print('\tloss is nan. please check...')
                              print("\toutputs.shape: ", outputs.shape, '\tbatch_y.shape: ', batch_y.shape)
                         
                         train_loss.append(loss.item())

                    if (i + 1) % 100 == 0 or i == 0:
                         print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                         speed = (time.time() - time_now) / iter_count
                         left_time = speed * ((self.args.train_epochs - epoch) * train_steps  - i)
                         print('\tspeed: {:.4f}s/iter; left time: {:d} min, {:.2f} s'.format(speed, int(left_time // 60), left_time % 60))
                         iter_count = 0
                         time_now = time.time()

                    if self.args.use_amp:
                         model_optim.zero_grad()
                         scaler.scale(loss).backward()

                         # grad emerges after backward
                         if (i + 1) % 100 == 0 or i == 0:
                              grd_norm = compute_gradient_norm(self.model)
                              if self.args.grad_clip:
                                   print(f"\t\tTotal norm of gradients: {grd_norm:.2f}, "
                                        f"{'clipped!' if grd_norm > self.args.max_norm else ''}")
                              else:
                                   print(f"\t\tTotal norm of gradients: {grd_norm:.2f}")
                         if self.args.grad_clip:
                              clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                         
                         scaler.step(model_optim)
                         scaler.update()
                    else:
                         model_optim.zero_grad()
                         loss.backward()

                         # IMPORTANT!!
                         if epoch >= self.args.warmup_epochs:
                              self.model.topk_gradient_mask() # 0724前面没有被调用 忘记加括号()
                         # else:
                         #      print(f'Warmup epochs')

                         model_optim.step()

                    if self.args.lradj == 'TST':
                         adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler, False)
                         scheduler.step()

               t2 = time.time() - epoch_time
               print("Epoch: {} cost time: {} min {:.1f}s".format(epoch + 1, t2 // 60, t2 % 60))
               train_loss = np.average(train_loss)
               vali_loss = self.vali(vali_data, vali_loader, criterion)
               test_loss = self.vali(test_data, test_loader, criterion)

               print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
                  f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
               early_stopping(vali_loss, self.model, path, epoch=epoch+1)
               if early_stopping.early_stop:
                    print("Early stopping")
                    break
               adjust_learning_rate(model_optim, epoch+1, self.args, scheduler)
          
          best_model_path = os.path.join(path, 'checkpoint.pth')
          if os.path.isfile(best_model_path):
               self.model.load_state_dict(torch.load(best_model_path))

          self.calibrate_ad_params(train_loader)

     def test(self, setting=None, test=0, test_batch_size=None):
          test_data, test_loader = self._get_data(flag='test', test_batch_size=test_batch_size)
          if test < 1:
               print('loading model...')
               self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))
          else:
               train_data, train_loader = self._get_data(flag='train')
               self.calibrate_ad_params(train_loader)
               full_folder, new_setting = find_most_recently_modified_subfolder(self.args.checkpoints, file_name='checkpoint.pth',
                                                                                contain_str=self.args.model_id_ori)
               if full_folder is not None and compare_prefix_before_third_underscore(setting, new_setting):
                    print(f'loading model from {full_folder} ...')
                    self.model.load_state_dict(torch.load(os.path.join(full_folder, 'checkpoint.pth')))
               else:
                    raise ValueError(f'No model found in {self.args.checkpoints} with prefix {self.args.model_id_ori}')
          
          preds = []
          trues = []
          eval_imp = []
          mask_mat_list = []
          folder_path = self.args.folder_path
          if not os.path.exists(folder_path):
               os.makedirs(folder_path)
          
          save_period = max(len(test_loader) // 5, 1)
          eval_stages = self.args.git_multi_stage + 1
          self.model.eval()

          if self.args.root_analysis:
               sample_data = []
               global_idx = 0
          
          with torch.no_grad():
               for i, (batch_x, batch_y, *_) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)

                    # if not self.imp_mode:
                    #      batch_x_mark = batch_x_mark.float().to(self.device)
                    #      batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    B, T, N = batch_x.shape
                    if self.imp_mode:
                         batch_y = batch_x
                         # batch_y_mark = batch_x_mark
                         mask = torch.rand((B, T, N)).to(self.device)
                         mask[mask <= self.args.mask_rate] = 0
                         mask[mask > self.args.mask_rate] = 1
                         mask_mat = mask == 0
                         batch_x = batch_x.masked_fill(mask_mat, 0)
                    else:
                         batch_y = batch_y.float().to(self.device)
                         mask_mat = torch.ones((B, T, N)).to(self.device) != 0
                    
                    mask_input = mask_mat if self.imp_mode else None
                    if self.args.use_amp:
                         with torch.cuda.amp.autocast():
                              outputs_list = self.model(batch_x, None, None, mask_input)
                    else:
                         # outputs_list = self.model(batch_x, batch_x_mark, batch_y_mark, mask_input)
                         outputs_list = self.model(batch_x, batch_y)
                    
                    if isinstance(outputs_list, tuple):
                         outputs = outputs_list[0]
                    else:
                         outputs = outputs_list
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    outputs = outputs.detach().cpu().numpy()

                    batch_y_np = batch_y.detach().cpu().numpy()

                    pred = outputs
                    true = batch_y
                    mask_mat = mask_mat.cpu().numpy()

                    if self.imp_mode:
                         pred[~mask_mat] = true[~mask_mat]

                    preds.append(pred)
                    trues.append(true.cpu().numpy())
                    mask_mat_list.append(mask_mat)

                    if self.args.root_analysis:
                         # batch_x 是输入窗口，形状 (B, T, N)
                         batch_x_np = batch_x.cpu().numpy()
                         for b in range(B):
                              seq_idx = global_idx + b
                              # if b == 0:
                              #      print(f'[debug] y_hat.shape {outputs.shape}')
                              #      print(f'[debug] batch_y_np.shape {batch_y_np.shape}')
                              sample_data.append({
                                   'idx': seq_idx,
                                   'x': batch_x_np[b],               # (T, N)
                                   'y': batch_y_np[b, -1, :],        # 当前时刻真值
                                   'y_hat': outputs[b, -1, :]        # 当前时刻预测
                              })
                         global_idx += B

                    # if i % save_period == 0:
                    #      print(f'processing batch{i} at test phase...')
                    #      input_ = batch_x.cpu().numpy()
                         
                    #      gt = np.concatenate((input_[0, :, -1], true[:, :, -1]), axis=0)
                    #      pd = np.concatenate((input_[0, :, -1], pred[:, :, -1]), axis=0)
                    
                    # TODO: save pdf
                    # if self.args.save_pdf:

          preds_array = np.concatenate(preds, axis=0)
          trues_array = np.concatenate(trues, axis=0)
          mask_mat = np.concatenate(mask_mat_list, axis=0)
          print('test shape:', preds_array.shape, trues_array.shape)

          mse_list = []
          mae_list = []
          if self.imp_mode:
               mae, mse, rmse, mape, mspe = metric(preds_array[mask_mat], trues_array[mask_mat])
               f = open("result_imputation.txt", 'a')
          else:
               mae, mse, rmse, mape, mspe, r2, pear, mase = metric(preds_array, trues_array)
               print(f'preds_array.shape: {preds_array.shape}')
               print(f'final output mse: {mse:.5f}, mae: {mae:.5f}, r2: {r2:.5f}, pear: {pear:.5f}, mase: {mase:.5f}')
               f = open("result_long_term_forecast.txt", 'a')
          
          mse_list.append(mse)
          mae_list.append(mae)

          mse_plus_mae = [a + b for a, b in zip(mse_list, mae_list)]
          min_index = mse_plus_mae.index(min(mse_plus_mae))
          mse = mse_list[min_index]
          mae = mae_list[min_index]

          f.write(setting + "  \n")
          if self.imp_mode:
               f.write('mse:{}, mae:{}'.format(mse, mae))
          else:
               f.write(f'mae: {mae:.5f}, mse: {mse:.5f}, r2: {r2:.5f}, pear: {pear:.5f}, mase: {mase:.5f}')
          f.write('\n')
          f.write('\n')
          f.close()

          if self.imp_mode and self.eval_flag:
            cos_dist = eval_imp[min(min_index, len(eval_imp) - 1)]  # cos_dist could be short
            print(f"Of all stages, best stage: {min_index}; best mse:{mse:.5f}, best mae:{mae:.5f}, "
                  f"cos_dist:{cos_dist:.5f}")
          else:
               print(f"Of all stages, best stage: {min_index}; best mse:{mse:.5f}, best mae:{mae:.5f}")

          np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
          write_into_xls(os.path.join(folder_path, 'metrics.xlsx'), [mae, mse, rmse, mape, mspe])

          # rename
          file_name = f"MSE_{mse:.5f}_MAE_{mae:.5f}_" + setting
          new_folder_path = os.path.join('results', file_name[:254])
          os.rename(folder_path, new_folder_path)

          results = self.anomaly_detection_test(test_loader)
          print(f'[Anomaly Detection] Anomaly Detecting....')
          results = {k: round(v, 5) for k, v in results.items()}
          print(results)

          if self.args.root_analysis:
               self._run_root_cause_analysis(sample_data, setting)
               print(f'[Root Cause Analysis] Root Cause Analysis report saved to {self.args.root_analysis_save_path}')

          if self.args.compress_causal_mat:
               try:
                    causal_mat = self.model.get_causal_matrix()
                    compressed = self._compress_causal_matrix(causal_mat)

                    save_path = os.path.join('causal_matrices', self.args.model, f'{self.args.model_id}.npy')
                    np.save(save_path, compressed.cpu().numpy())
                    print(f"[Causal Matrix] Compressed matrix saved to {save_path}")
               except Exception as e:
                    print(f"[Causal Matrix] Failed to compress/save: {e}")

          return mse, mae, results
     
     def predict(self, setting, load=False):
          pred_data, pred_loader = self._get_data(flag='pred')

          if load:
               path = os.path.join(self.args.checkpoints, setting)
               best_model_path = path + '/' + 'checkpoint.pth'
               self.model.load_state_dict(torch.load(best_model_path))
          
          preds = []
          self.model.eval()
          with torch.no_grad():
               for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.flaot()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zero_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                         with torch.cuda.amp.autocast():
                              if self.args.output_attention:
                                   outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                              else:
                                   outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                         if self.args.output_attention:
                              outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                         else:
                              outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs.detach().cpu().numpy()
                    if pred_data.scale and self.args.inverse:
                         shape = outputs.shape  # [b,s,n]
                         outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    preds.append(outputs)
          preds = np.array(preds)
          preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

          # result save
          folder_path = os.path.join('results', setting)
          if not os.path.exists(folder_path):
               os.makedirs(folder_path)

          np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)

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
                    pred = self.model(batch_x)
                    all_residuals.append((batch_y - pred).cpu())
          residuals = torch.cat(all_residuals, dim=0)

          residuals_flat = residuals.reshape(-1, residuals.shape[-1])
          self.sigma = residuals.std(dim=0).reshape(-1) # duplicate
          self.sigma[self.sigma == 0] = 1.0
          e = residuals_flat / self.sigma
          spe = (e ** 2).sum(dim=-1)
          # print(f'[debug] spe.shape {spe.shape}')
          # print(f'[debug] e.shape: {e.shape}')
          # print(f'[debug] self.sigma.shape: {self.sigma.shape}')
          # print(f'[debug] residuals.shape: {residuals.shape}')
          self.theta_spe = torch.quantile(spe, self.args.ad_quantile)
          # print(f'[debug] self.theta_spe: {self.theta_spe.shape}, self.theta_spe: {self.theta_spe}')

          beta = 0.01  # 可调整，或从args获取
          lower = torch.quantile(e, beta/2, dim=0)
          upper = torch.quantile(e, 1 - beta/2, dim=0)
          self.tau_i = torch.max(torch.abs(lower), torch.abs(upper))
          # print(f'[debug] self.tau_i: {self.tau_i.shape}')

          if self.args.root_analysis:
               print(f'[debug] here root_analysis')
               self._extract_parents_candidates()

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
                    y_hat = self.model(batch_x)
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
     
     def _extract_parents_candidates(self):
          """
          从训练好的模型中提取每个输出变量的候选父节点列表（按强度排序）。
          对于线性 LoCo，使用权重绝对值；对于 Kernel LoCo，计算组范数。
          同时修剪：去除自环，过滤弱连接。
          """
          weight = self.model.linear.weight.detach().cpu()
          N = self.args.input_dim
          L = self.args.seq_len

          if self.args.model == 'Linear_LoCo':
               # weight shape: (N, N*L)
               group_norms = torch.abs(weight)          # (N, N*L)
          else:  # Kernel_LoCo
               D = self.args.D
               # weight shape: (N, N*L*D) -> reshape to (N, N*L, D)
               group_norms = weight.view(N, N * L, D).norm(dim=2)  # (N, N*L)

          k = min(self.args.topk, N * L)
          
          # 可选的弱连接阈值：设为所有非零范数的 1% 分位数，或固定小值
          all_vals = group_norms.flatten()
          non_zero_vals = all_vals[all_vals > 0]
          if len(non_zero_vals) > 0:
               threshold = torch.quantile(non_zero_vals, 0.1)  # 1% 分位数
          else:
               threshold = 1e-6
          # 也可以直接用固定阈值，如 threshold = 1e-4
          # threshold = 1e-4
          
          parents_candidates = []
          for i in range(N):
               row = group_norms[i]                     # (N*L,)
               vals, indices = torch.topk(row, k)       # 默认降序
               parents = []
               for idx, val in zip(indices, vals):
                    if val <= threshold:
                         break  # 因为降序，后续值更小，直接跳出
                    col = idx.item()
                    t = col // N          # 时间步索引，0 ~ L-1，0 对应最早的历史点
                    # t = col // N + 1
                    var = col % N         # 变量索引，0 ~ N-1
                    tau = L - t           # 滞后 1 ~ L
                    if var == i:          # 去除自环
                         continue
                    parents.append((var, tau, val.item()))
               parents_candidates.append(parents)
          
          print(f"[debug] N={N}, L={L}, weight.shape={weight.shape}, threshold={threshold:.6f}")
          for i in range(min(2, N)):
               print(f"Row {i}: {parents_candidates[i][:3]}")
          self.parents_candidates = parents_candidates

     def _run_root_cause_analysis(self, sample_data, setting):
          """
          使用收集的样本数据执行根因分析，并生成报告。
          """
          if not self.parents_candidates:
               print("Warning: No parents candidates available. Skip root cause analysis.")
               return

          print(f'[debug] root cause analysis ...')
          analyzer = RootCauseAnalyzer(
               model=self.model,
               args=self.args,
               sigma=self.sigma,
               tau_i=self.tau_i,
               theta_spe=self.theta_spe,
               parents_candidates=self.parents_candidates,
               device=self.device
          )

          anomalies = []
          total = len(sample_data)
          # 每处理 1% 或至少每 1000 个样本打印一次
          print_interval = max(1, total // 20)  # 5% 的样本数
          start_time = time.time()
          
          for idx, data in enumerate(sample_data):
               # 定期打印进度
               if idx % print_interval == 0 or idx == total - 1:
                    elapsed = time.time() - start_time
                    progress = (idx + 1) / total * 100
                    print(f"Root cause analysis progress: {idx+1}/{total} ({progress:.1f}%) samples processed, elapsed: {elapsed:.1f}s")
               
               result = analyzer.analyze(data['idx'], data['x'], data['y'], data['y_hat'], sample_data)
               if result is not None:
                    anomalies.append(result)
          
          print(f"Root cause analysis completed. Found {len(anomalies)} anomalous samples.")
          
          # report generation
          self._generate_report(anomalies, setting)
     
     def _generate_report(self, anomalies, setting):
          """
          生成简洁美观的根因分析报告（txt格式）。
          """
          reports_dir = os.path.join('.', 'reports', self.args.model)
          os.makedirs(reports_dir, exist_ok=True)
          report_path = os.path.join(reports_dir, f"{self.args.model_id}.txt")

          with open(report_path, 'w') as f:
               f.write('='*80 + '\n')
               f.write(f'Root Cause Analysis Report\n')
               f.write(f'Model ID: {self.args.model_id}\n')
               f.write(f'Dataset: {self.args.dataset}\n')
               f.write(f'Model: {self.args.model}\n')
               f.write(f'Number of anomalies detected: {len(anomalies)}\n')
               f.write("="*80 + "\n\n")

               if not anomalies:
                    f.write("No anomalies were detected during the test.\n")
                    return
               
               for idx, anom in enumerate(anomalies):
                    f.write(f"--- Anomaly #{idx+1} (sample index: {anom['idx']}) ---\n")
                    f.write(f"SPE score: {anom['spe']:.4f}\n")
                    f.write(f"Most anomalous variable: {anom['i_star']} (standardized error: {anom['e'][anom['i_star']]:.3f})\n")
                    f.write("Root causes (sorted by attribution score):\n")
                    for r in anom['root_causes']:
                         if r['var'] == anom['i_star'] and r['lag'] == 0:
                              score_str = "root causality, no upstream cause"
                         else:
                              score_str = f"attribution score: {r['attr']:.4f}"
                         f.write(f"  - Variable {r['var']}, lag {r['lag']} ({score_str})\n")
                         # 打印传播路径
                         path_str = " -> ".join([f"X{p[0]}(t-{p[1]})" for p in r['path']])
                         f.write(f"    Path: {path_str}\n")
                         f.write(f"    Cumulative contribution: {r['contrib']:.4f}\n")
                    f.write("\n")
          print(f"Root cause analysis report saved to {report_path}")

     def _compress_causal_matrix(self, S):
          """
          将原始因果矩阵 S (N, N*L) 列顺序为 [t=0..L-1, var=0..N-1] 压缩为 (N, N) 的全局因果强度矩阵。
          参数：
               S: torch.Tensor, shape (N, N*L)
          返回：
               torch.Tensor, shape (N, N)
          """
          method = self.args.compress_causal_mat_method
          quantile = self.args.compress_causal_mat_quantile
          alpha = self.args.compress_causal_mat_MA_alpha
          N = self.args.input_dim
          L = self.args.seq_len
          el_symmetric = self.args.el_symmetric

          # (N, L, N)
          # 应该是主对角线强相关
          # 直接 view(N, L, N) 因为训练数据的维度是 (L, N)
          S_reshaped = S.view(N, L, N)
          
          # S_reshaped = S.view(N, N, L)
          # S_reshaped = S_reshpaped.permute(0, 2, 1)
          if method == 'AVG':
               C = torch.mean(S_reshaped, dim=1)
          elif method == 'MAX':
               C, _ = torch.max(S_reshaped, dim=1)
          elif method == 'MA':
               S_flipped = S_reshaped.flip(dims=[1])
               weights = (torch.arange(1, L + 1, device=self.device).float()) ** (-alpha)
               weights = weights / weights.sum()
               # C = torch.sum(S_reshaped * weights.view(1, 1, L), dim=2)
               C = torch.einsum('ihj, h -> ij', S_flipped, weights)
          else:
               raise ValueError(f"Invalid compression method: {method}")

          if el_symmetric == 1:
               diag = torch.diag(C)                     # 保存对角线
               C = C - C.T                              # 去掉对称部分
               C = torch.relu(C)                        # 只保留正向差异
               # 恢复对角线（公式 Ai,i = Ai,i）
               C = C + torch.diag(diag)

          if 0 < quantile < 1:
               vals = torch.abs(C).flatten()
               threshold = torch.quantile(vals[vals > 0], quantile) if torch.any(vals > 0) else 0.0
               C = C * (torch.abs(C) >= threshold).float()
          return C