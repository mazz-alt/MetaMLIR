import numpy as np
from datetime import datetime
import argparse
from models.utils.utils import _logger
from models.model import Backbone
from models.dataloader import data_generator
from models.trainer import Trainer
from models.tester import Tester
import os
import torch

# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

home_dir = os.getcwd()
parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')
parser.add_argument('--seed', default=2023, type=int, help='seed value')

parser.add_argument('--training_mode', default='pre_train', type=str, help='pre_train, fine_tune')
parser.add_argument('--pretrain_dataset', default='all', type=str,
                    help='Dataset of choice: SleepEEG, FD_A, HAR, ECG')

# change!!!!
parser.add_argument('--target_dataset', default='motion34_3', type=str,
                    help='Dataset of choice: Epilepsy, FD_B, Gesture, EMG')

parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
parser.add_argument('--subset', action='store_true', default=False, help='use the subset of datasets')
parser.add_argument('--log_epoch', default=5, type=int, help='print loss and metrix')
parser.add_argument('--draw_similar_matrix', default=10, type=int, help='draw similarity matrix')
parser.add_argument('--pretrain_lr', default=0.0001, type=float, help='pretrain learning rate')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--use_pretrain_epoch_dir', default=None, type=str,
                    help='choose the pretrain checkpoint to finetune')
parser.add_argument('--pretrain_epoch', default=32, type=int, help='pretrain epochs')
parser.add_argument('--finetune_epoch', default=170, type=int, help='finetune epochs')

parser.add_argument('--masking_ratio', default=0.5, type=float, help='masking ratio') 
parser.add_argument('--positive_nums', default=3, type=int, help='positive series numbers')
parser.add_argument('--lm', default=3, type=int, help='average masked lenght')

parser.add_argument('--finetune_result_file_name', default="finetune_result.json", type=str,
                    help='finetune result json name')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature')



def set_seed(seed):
    SEED = seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    return seed


def main(args, configs, seed=None):
    method = 'ICH_TMT'
    sourcedata = args.pretrain_dataset
    targetdata = args.target_dataset
    training_mode = args.training_mode
    run_description = args.run_description

    logs_save_dir = args.logs_save_dir
    masking_ratio = args.masking_ratio
    pretrain_lr = args.pretrain_lr
    pretrain_epoch = args.pretrain_epoch
    lr = args.lr
    finetune_epoch = args.finetune_epoch
    temperature = args.temperature
    experiment_description = f"{sourcedata}_2_{targetdata}"

    os.makedirs(logs_save_dir, exist_ok=True)

    # Load datasets
    sourcedata_path = f"./dataset/{sourcedata}"  # './dataset/Epilepsy'
    targetdata_path = f"./dataset/{targetdata}"

    subset = args.subset  # if subset= true, use a subset for debugging.

    train_dl_all, valid_dl, test_dl = data_generator(targetdata_path, configs, training_mode, subset=subset)

    # set seed
    if seed is not None:
        seed = set_seed(seed)
    else:
        seed = set_seed(args.seed)

    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                      training_mode + f"_{seed}_pt_{masking_ratio}_{pretrain_lr}_{pretrain_epoch}_ft_{lr}_{finetune_epoch}_re")

    os.makedirs(experiment_log_dir, exist_ok=True)

    # Logging
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Pre-training Dataset: {sourcedata}')
    logger.debug(f'Target (fine-tuning) Dataset: {targetdata}')
    logger.debug(f'Seed: {seed}')
    logger.debug(f'Method:  {method}')
    logger.debug(f'Mode:    {training_mode}')
    logger.debug(f'Pretrain Learning rate:    {pretrain_lr}')
    logger.debug(f'Masking ratio:    {masking_ratio}')
    logger.debug(f'Pretrain Epochs:    {pretrain_epoch}')
    logger.debug(f'Finetune Learning rate:    {lr}')
    logger.debug(f'Finetune Epochs:    {finetune_epoch}')
    logger.debug(f'Temperature: {temperature}')
    logger.debug("=" * 45)

    # Load Model
    # Sleep
    model_sleep = Backbone(configs[0], args).to(device)
    params_group_sleep = [{'params': model_sleep.parameters()}]
    model_optimizer_sleep = torch.optim.Adam(params_group_sleep, lr=pretrain_lr, betas=(configs[0].beta1, configs[0].beta2),
                                       weight_decay=0)
    model_scheduler_sleep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer_sleep, T_max=pretrain_epoch)

    model_epil = Backbone(configs[1], args).to(device)
    params_group_epil = [{'params': model_epil.parameters()}]
    model_optimizer_epil = torch.optim.Adam(params_group_epil, lr=pretrain_lr, betas=(configs[1].beta1, configs[1].beta2),
                                       weight_decay=0)
    model_scheduler_epil = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer_epil, T_max=pretrain_epoch)
    
    model_HAR = Backbone(configs[2], args).to(device)
    params_group_HAR = [{'params': model_HAR.parameters()}]
    model_optimizer_HAR = torch.optim.Adam(params_group_HAR, lr=pretrain_lr, betas=(configs[2].beta1, configs[2].beta2),
                                       weight_decay=0)
    model_scheduler_HAR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer_HAR, T_max=pretrain_epoch)

    model_ECG = Backbone(configs[3], args).to(device)
    params_group_ECG = [{'params': model_ECG.parameters()}]
    model_optimizer_ECG = torch.optim.Adam(params_group_ECG, lr=pretrain_lr, betas=(configs[3].beta1, configs[3].beta2),
                                       weight_decay=0)
    model_scheduler_ECG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer_ECG, T_max=pretrain_epoch)

    model_FDA = Backbone(configs[4], args).to(device)
    params_group_FDA = [{'params': model_FDA.parameters()}]
    model_optimizer_FDA = torch.optim.Adam(params_group_FDA, lr=pretrain_lr, betas=(configs[4].beta1, configs[4].beta2),
                                       weight_decay=0)
    model_scheduler_FDA = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer_FDA, T_max=pretrain_epoch)

    model = [model_sleep, model_epil, model_HAR, model_ECG, model_FDA]
    model_optimizer = [model_optimizer_sleep, model_optimizer_epil, model_optimizer_HAR, model_optimizer_ECG, model_optimizer_FDA]
    model_scheduler = [model_scheduler_sleep, model_scheduler_epil, model_scheduler_HAR, model_scheduler_ECG, model_scheduler_FDA]

    # Trainer

    best_performance = Trainer(model, model_optimizer, model_scheduler, train_dl_all, valid_dl, test_dl, device, logger,
                               args, configs, experiment_log_dir, seed)

    best_performance = Tester(model, model_optimizer, model_scheduler, valid_dl, test_dl, device, logger,
                               args, configs, experiment_log_dir, seed)

    return best_performance


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    device = torch.device(args.device)
    
    # exec (f'from config_files.{args.pretrain_dataset} import Config as Configs')
    # configs = Configs()

    exec (f'from config_files.SleepEEG_Configs import Config as Configs_Sleep') # SleepEEG
    configs_Sleep = Configs_Sleep()
    exec ('from config_files.Epilepsy_Configs import Config as Configs_epil')  # Epilepsy
    configs_epil = Configs_epil()
    exec ('from config_files.HAR_Configs import Config as Configs_HAR')  # HAR
    configs_HAR = Configs_HAR() 
    exec ('from config_files.ECG_Configs import Config as Configs_ECG')  # ECG
    configs_ECG = Configs_ECG() 
    exec ('from config_files.FD_A_Configs import Config as Configs_FDA')  # FDA
    configs_FDA = Configs_FDA() 
    
    exec ('from config_files.TMT_Configs import Config as Configs_TMT')  # FDA
    Configs_TMT = Configs_TMT() 

    configs = [configs_Sleep, configs_epil, configs_HAR, configs_ECG, configs_FDA, Configs_TMT]

    main(args, configs)

