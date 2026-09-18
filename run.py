import argparse
import datetime
import math
import shutil
import os, sys, random
import time 
import torch
import numpy as np
from experiments.exp_detection import Exp_Detection
from experiments.exp_detection_lasso import Exp_Detection_Lasso
from utils.tools import find_most_recently_modified_subfolder

from config import *
def get_setting(args_, iter_=0):
     if args_.task_name == 'detection':
          setting_ = '{}_{}_{}_{}_sl-{}_pl-{}_{}'.format(
                    args_.task_name,
                    args_.model_id,
                    args_.model,
                    args_.dataset,
                    args_.seq_len,
                    args_.pred_len,
                    iter_
               )
     else: 
          setting_ = 'No specific task'
     return setting_[:255]

if __name__ == '__main__':

     parser = argparse.ArgumentParser(description='default')

     # task_name
     parser.add_argument('--task_name', type=str, default='detection',
                        help='task_name', choices=['detection', 'forecast', 'imputation'])

     # basic config
     parser.add_argument('--fix_seed', type=int, default=2023, help='whether to fix seed')
     parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
     parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
     parser.add_argument('--model', type=str, required=True, default='Linear_LoCo', choices=['Linear_LoCo', 'Kernel_LoCo', 'Lasso', 'Linear_LoCo_diff', 'Kernel_LoCo_diff',])
     parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
     parser.add_argument('--checkpoint_check', type=int, default=0, help='if checkpoint exists, skip training')
     parser.add_argument('--save_every_epoch', type=int, default=0, help='save_every_epoch')
     parser.add_argument('--model_stats_mode', type=int, default=0, help='model_stats_flag')
     parser.add_argument('--test_mode', type=int, default=0, help='gpu')

     parser.add_argument('--efficient_training', type=int, default=0, help='whether to use efficient_training ')
     parser.add_argument('--loss_mode', type=str, default='L1', help='loss_mode',
                        choices=['L1', 'L2', 'L1L2', 'MAPE', 'MASE', 'SMAPE'])
     parser.add_argument('--alpha', type=float, default=0.0, help='alpha in loss function')
     parser.add_argument('--lossfun_alpha', type=float, default=0.0, help='alpha in loss function')
     parser.add_argument('--features', type=str, default='M', help='features type')
     parser.add_argument('--copy_file', type=int, default=0, help='copy_file')
     parser.add_argument('--send_mail', type=int, default=0, help='send mail after training completes.')

     # dataset
     parser.add_argument('--dataset', type=str, required=True, default='SWAT', choices=['MSL', 'SMAP', 'PSM', 'SWAT', 'SMD', 'Creditcard', 'GECCO', 'SWAN', 'KDD99', 'CICIDS'], help='data for anomaly detection')
     parser.add_argument('--root_path', type=str, default='', help='root pwd for dataset')
     parser.add_argument('--input_dim', type=int, default=0, help='the channel of input dataset')
     parser.add_argument('--seq_len', type=int, default=20)
     parser.add_argument('--pred_len', type=int, default=1)
     parser.add_argument('--step', type=int, default=1)

     # optimization
     parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')   

     # monte carlo experiments
     parser.add_argument('--iter', type=int, default=1, help='experiments times')
     parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
     parser.add_argument('--warmup_epochs', type=int, default=-1, help='warmup_epochs')
     parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
     parser.add_argument('--test_batch_size', type=int, default=2)
     parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
     parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
     parser.add_argument('--des', type=str, default='test', help='exp description')
     parser.add_argument('--loss', type=str, default='MSE', help='loss function')
     parser.add_argument('--lradj', type=str, default='type1', 
                         choices=['type1', 'type2', 'type3', 'card', 'cosine', 'constant', 'TST'], help='adjust learning rate')
     parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
     parser.add_argument('--grad_clip', type=int, default=0, help='fix_seed')
     parser.add_argument('--max_norm', type=float, default=1e6, help='dp_rank for dynamic projection')
     parser.add_argument('--git_multi_stage', type=int, default=4, help='git_multi_stage')

     parser.add_argument('--seq_inter', type=int, default=1, help='check mse/mae in immediate stages')
     parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

     # continue training
     parser.add_argument('--resume_training', type=int, default=0, help='resume training')
     parser.add_argument('--resume_epoch', type=int, default=0, help='resume epoch')

     # GPU
     parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
     parser.add_argument('--gpu', type=int, default=0, help='gpu')
     parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
     parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

     # loco
     parser.add_argument('--D', type=int, default=50, help='kernel dim for each stamp')
     parser.add_argument('--topk', type=int, default=20, help='topk in gradient clip')
     parser.add_argument('--sigma', type=float, default=1.0, help='sigma in RFF')
     # TODO
     parser.add_argument('--mask', type=int, default=None, help='whether to use mask in loco')

     # lasso
     parser.add_argument('--lasso_alpha', type=float, default=0.1, help='lasso_alpha')

     # other
     parser.add_argument('--lamda1', type=float, default=1.0, help='lamda1 in loss function')
     parser.add_argument('--lamda1_delta', type=float, default=0.0, help='lamda1 in loss function')
     parser.add_argument('--ad_quantile', type=float, default=0.99, help='calibrate_ad_params')
     parser.add_argument('--ad_tau', type=float, default=2.5, help='ad_tau')
     
     # root analysis
     parser.add_argument('--root_analysis', type=int, default=0, help='need root analysis report?')

     # causality matrix compress
     parser.add_argument('--compress_causal_mat', type=int, default=0, help='compress_matrix')
     parser.add_argument('--compress_causal_mat_quantile', type=float, default=0.7, help='compress_matrix_quantile')
     parser.add_argument('--compress_causal_mat_method', type=str, default='MA', choices=['MA', 'MAX', 'AVG'], 
                         help='MA: "MOVING AVERAGE", MAX: "MAXIMUM", AVG: "AVERAGE"')
     parser.add_argument('--compress_causal_mat_MA_alpha', type=float, default=0.5, help='compress_matrix_MA_alpha')
     parser.add_argument('--el_symmetric', type=int, default=0, help='whether to eliminat symmetric')

     args = parser.parse_args()
     args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
     print('args.use_gpu: ', args.use_gpu)

     if args.input_dim == 0:
        args.input_dim = dataset2channels[args.dataset]
     if args.root_path == '' or args.root_path is None:
          args.root_path = dataset2path[args.dataset]

     if args.resume_training > 0 >= args.resume_epoch:
          confirm_again = input('args.resume_traing > 0 >= args.resume_epoch, Continue? (yes/no): ')
          if not confirm_again.lower().startswith('y'):
               print('program exit.')
               sys.exit() 

     if not args.use_gpu:
          confirm_again = input('No using gpu. Continue? (yes/no): ')
          if not confirm_again.lower().startswith('y'):
               print('program exit.')
               sys.exit()
     
     if args.use_gpu and args.use_multi_gpu:
          args.devices = args.devices.replace(' ', '')
          device_ids = args.devices.split(',')
          args.device_ids = [int(id_) for id_ in device_ids]
          args.gpu = args.device_ids[0]
     
     if args.root_analysis == 1:
          reports_dir = './reports'
          os.makedirs(reports_dir, exist_ok=True)
          sub_dir = os.path.join(reports_dir, args.model)  # 使用原始 model_id 作为子文件夹名
          os.makedirs(sub_dir, exist_ok=True)

     if args.compress_causal_mat == 1:
          causal_mat_dir = './causal_matrices'
          os.makedirs(causal_mat_dir, exist_ok=True)
          sub_dir = os.path.join(causal_mat_dir, args.model)
          os.makedirs(sub_dir, exist_ok=True)

          if args.compress_causal_mat_quantile < 0 or args.compress_causal_mat_quantile > 1:
               args.compress_causal_mat_quantile = args.topk / args.input_dim

     print('Args in experiment:')
     if args.model == 'Lasso':
          Exp = Exp_Detection_Lasso
          
     else:
          Exp = Exp_Detection

     # args.model_id
     model_id_ori = args.model_id
     args.model_id_ori = model_id_ori
     args.model_id = model_id_ori + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

     log_txt = 'log.txt'
     best_log_txt = 'best_log.txt'
     best_log_data_path = 'best_results'
     if not os.path.exists(best_log_data_path):
          os.makedirs(best_log_data_path)
     best_log_dataset_txt = os.path.join(best_log_data_path, model_id_ori + '.txt')

     test_batch_size_list = [args.test_batch_size]

     global_time0 = time.time()
     
     if args.fix_seed:
          fix_seed = 2023  # 2023  # if args.task_name == 'forecasting' else 2021
          random.seed(fix_seed)
          torch.manual_seed(fix_seed)
          np.random.seed(fix_seed)

     setting_zero = get_setting(args, 0)
     folder_path = os.path.join('results', setting_zero)
     args.folder_path = folder_path

     best_mse, best_mae = math.inf, math.inf
     time_vec = []

     if args.is_training and not args.model_stats_mode:
          lamda1_ori = args.lamda1
          best_lamda1 = lamda1_ori
          test_batch_size_ori = args.test_batch_size
          best_ii = 0

          # test_batch_size_list
          if test_batch_size_ori not in test_batch_size_list:
               test_batch_size_list.append(test_batch_size_ori)
          
          # copy file
          if not os.path.exists(folder_path):
               os.makedirs(folder_path)
          
          if args.copy_file:
               shutil.copytree('./model', os.path.join(folder_path, 'model'))
               shutil.copytree('./layers', os.path.join(folder_path, 'layers'))
               shutil.copytree('./utils', os.path.join(folder_path, 'utils'))
               shutil.copytree('./experiments', os.path.join(folder_path, 'experiments'))
               shutil.copytree('./scripts', os.path.join(folder_path, 'scripts'))
               shutil.copytree('./data_provider', os.path.join(folder_path, 'data_provider'))
               print('Some python files have been copied...')

          # check
          if args.checkpoint_check and not args.resume_training and args.test_mode == 0:
               full_folder, new_setting = find_most_recently_modified_subfolder('./checkpoints/',
                                                                             file_name='checkpoint.pth',
                                                                             contain_str=args.model_id_ori)
               
               if full_folder is not None:
                    print(f'{args.model_id_ori} checkpoints already exist.')
                    sys.exit()

          idx = 0
          for ii in range(args.iter):
               best_string = ''
               exp = None
               lamda1_delta = 0
               lamda1_list = [args.lamda1]

               itr_count = args.iter * len(lamda1_list)
               for lamda1 in lamda1_list:
                    idx += 1
                    time_now = time.time()

                    args.lamda1 = lamda1
                    args.model_id = model_id_ori + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    setting = get_setting(args, ii) if ii > 0 else setting_zero

                    if ii > 0:
                         args.folder_path = os.path.join('results', setting)

                    exp = Exp(args)

                    if args.test_mode == 0:
                         args_dict = vars(args)
                         for k, v in sorted(args_dict.items()):
                              print(f'{k}: {v}, ', end=' ')
                         print('')
                         print(f'>>>>>>>start training : {setting} (batch_size:{args.batch_size}), ' 
                               f'(best_mse:{best_mse:.5f}, best_mae:{best_mae:.5f}) (Monte Carlo: {idx}/{itr_count}) '
                               f'(epochs per exp:{args.train_epochs})'
                               f'>>>>>>>>>>>>')
                         exp.train(setting)
                    else:
                         args_dict = vars(args)
                         for k, v in sorted(args_dict.items()):
                              print(f'{k}: {v}, ', end=' ')
                         print('')
                    
                    mse = mae = math.inf
                    best_batch_size = np.nan
                    for test_bs in sorted(test_batch_size_list):
                         print('>>>>>>>testing : {} (test_batch_size: {})<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.
                                   format(setting, test_bs))
                         mse0, mae0, res = exp.test(setting, test=args.test_mode, test_batch_size=test_bs)

                         if mse0 < mse:
                              mse = mse0
                              mae = mae0
                              best_batch_size = test_bs

                    print(f'\tbest_test_batch_size: {best_batch_size}, best_mse: {mse:.5f}, best_mae: {mae:.5f}')

                    if mse + mae <= best_mse + best_mae:
                         best_lamda1 = lamda1
                         best_mse, best_mae, best_ii = mse, mae, ii

                    # log into txt
                    mse_mse_string = (f'mse:{mse:.5f}, mae:{mae:.5f}, lamda1:{lamda1:.2f}, '
                                   # f'git_multi_stage:{args.git_multi_stage}, decoder_cat:{args.decoder_cat_num}, '
                                   f'alpha1:{args.alpha}, loss_fun_alpha1:{args.lossfun_alpha}')
                    ano_res_str = " ".join([f"{k}: {v}" for k, v in res.items()])
     
                    print(mse_mse_string)
                    with open(log_txt, 'a') as f:
                         f.write(f'------------ {setting} -------------' + '\n' + '\n')
                         args_dict = vars(args)
                         for k, v in sorted(args_dict.items()):
                              f.write(f'{k}: {v}, ')
                         f.write('\n\n')
                         f.write('\t' + ano_res_str + '\n\n')
                         f.write('--------------------------------- Ends -----------------------------\n\n')

                    time_vec.append(time.time() - time_now)
                    print(f'training time left is {np.mean(time_vec) * (itr_count - idx) / 60.0:.2f} min...')
                    # torch.cuda.empty_cache()

                    # best_string by far
                    best_string = (f'best_mse (by far): {best_mse:.5f}, best_mae: {best_mae:.5f};\t '
                                   f'best_lamda1: {best_lamda1:.2f}, \t'
                                   f'best_ii:{best_ii}, used time: {time_vec[-1] / 60.0: .2f} min(s)')
                    print(best_string)
               
               # write into txt
               with open(log_txt, 'a') as f:
                    f.write(f'============================= {args.model_id}============================= \n')
                    f.write('\n\t' + best_string + '\n\n\n')
                    f.write('\n\t' + ano_res_str + '\n')
                    f.write('================================== end ===================================\n\n')

               with open(best_log_txt, 'a') as f:
                    f.write(f'============================= {args.model_id}=============================\n')
                    f.write('\n\t' + best_string + '\n' + '\n')
                    f.write('\n\t' + ano_res_str + '\n')
                    f.write('================================== end ===================================\n\n')

               args.lamda1 = lamda1_ori

               if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)
          
          # global best
          best_string = (f'global best_mse: {best_mse:.5f}, best_mae: {best_mae:.5f};\t '
                       f'best_lamda1: {best_lamda1:.2f},  \t'
                       f'best_ii:{best_ii}; Avg time: {np.mean(time_vec) / 60.0: .2f} min(s)')
          print(best_string)
          with open(best_log_txt, 'a') as f:
               f.write(f'============================ global best of {model_id_ori}=============================\n')
               f.write('\n\t' + best_string + '\n\n')
               f.write('\n\t' + ano_res_str + '\n')
               f.write('======================================== end ============================================\n\n')
          # torch.cuda.empty_cache()

          # write also into best_log_dataset_txt
          with open(best_log_dataset_txt, 'a') as f:
               f.write(f'============================ global best of {args.model_id}=============================\n')
               args_dict = vars(args)
               for k, v in sorted(args_dict.items()):
                    f.write(f'\t{k}: {v}; ')
               f.write('\n')
               f.write('\n\t' + best_string + '\n\n')
               f.write('\n\t' + ano_res_str + '\n')
               f.write('======================================== end ============================================'
                         + '\n' + '\n')
     elif not args.model_stats_mode and not args.is_training:
          args_dict = vars(args)
          for k, v in sorted(args_dict.items()):
               print(f'{k}: {v}, ', end=' ')
          print('')

          ii = 0
          setting = get_setting(args, ii)

          exp = Exp(args)
          print('>>>>>>> testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
          exp.test(setting, test=1)
          torch.cuda.empty_cache()
     
     elif args.model_stats_mode:
          args_dict = vars(args)
          for k, v in sorted(args_dict.items()):
               print(f'{k}: {v}, ', end=' ')
          print('')
          print(f'============================ model_stats {args.model_id_ori}============================= ')
          exp = Exp(args)  # set experiments
          exp.compute_model_stats()

     print(f'A total of {(time.time() - global_time0) / 60.0: .2f} min(s) used...')


