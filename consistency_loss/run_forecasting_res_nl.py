from functools import partial
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "genTS", REPO_ROOT / "genTS/external/tslib"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np

# get data loader
from genTS.external.tslib.data_provider import data_factory as tslib_loader

# from genTS.data.all_data_info import get_args_with_datainfo
from genTS.utils.argparser_default import get_parser
from genTS.utils.tools import str2bool

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import _functional_assert_async, nn, Tensor
import tqdm

# from random_transforms import SpectralRandomKernelTransform, Rocket
from synthetic_data import Dataset_Function_MC_Core, Dataset_Function_MC
import time
from genTS.external.tslib.utils.tools import EarlyStopping
from torch.utils.data import Dataset, DataLoader
from genTS.backbones.TSLANet_fc import TSLANet
from genTS.backbones.PDF_fc import PDFForecast
from genTS.backbones.PaiFilter import PaiFilter
from genTS.backbones.iTransformer_fc import iTransformerForecast
from residual_check_general import ResidualCheckGeneral, TFCheckGeneral, BranchResidualCheck, BranchTFCheck, DummyCheckGeneral
import random
from copy import deepcopy
from rollout import rollout_test
from experiment_utils import run_setting, write_json


class SmallMLP(nn.Module):
    """ReLU MLP with biases."""
    def __init__(self, configs):
        super(SmallMLP, self).__init__()

        self.IN = configs.instance_norm

        n_layers = getattr(configs, 'depth', 2)
        if n_layers < 2:
            print(f"[SmallMLP] depth={n_layers} < 2 is invalid, "
                  f"clamping to 2.")
            n_layers = 2

        layers = [nn.Linear(configs.seq_len, configs.d_model, bias=True)]
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(configs.d_model, configs.d_model, bias=True))
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forward(self, x, *args):
        if self.IN:
            x_mean = x.mean(dim=1, keepdim=True)
            x = x - x_mean

        x = x.permute(0, 2, 1)  # x: [B, C, T]

        for layer in self.layers:
            x = F.relu(layer(x))
        y = self.head(x)

        y = y.permute(0, 2, 1)

        if self.IN:
            y = y + x_mean
        return y


class SmallMLPNoBias(nn.Module):
    """ReLU MLP without biases."""
    def __init__(self, configs):
        super(SmallMLPNoBias, self).__init__()

        self.IN = configs.instance_norm

        n_layers = getattr(configs, 'depth', 2)
        if n_layers < 2:
            print(f"[SmallMLPNoBias] depth={n_layers} < 2 is invalid, "
                  f"clamping to 2.")
            n_layers = 2

        layers = [nn.Linear(configs.seq_len, configs.d_model, bias=False)]
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(configs.d_model, configs.d_model, bias=False))
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(configs.d_model, configs.pred_len, bias=False)

    def forward(self, x, *args):
        if self.IN:
            x_mean = x.mean(dim=1, keepdim=True)
            x = x - x_mean

        x = x.permute(0, 2, 1)  # x: [B, C, T]

        for layer in self.layers:
            x = F.relu(layer(x))
        y = self.head(x)

        y = y.permute(0, 2, 1)

        if self.IN:
            y = y + x_mean
        return y


class BaseLinearModel(nn.Module):
    def __init__(self, configs):
        super(BaseLinearModel, self).__init__()
        self.IN = configs.instance_norm
        self.indv = configs.individual
        if self.indv:
            self.l1 = nn.ModuleList(
                [nn.Linear(configs.seq_len, configs.pred_len, bias=False) for _ in range(configs.enc_in)]
            )
        else:
            self.l1 = nn.Linear(configs.seq_len, configs.pred_len, bias=False)

    def forward(self, x, *args):
        if self.IN:
            x_mean = x.mean(dim = 1, keepdim = True)
            x = x-x_mean
        x = x.permute(0,2,1) # x: [B, C, T]
        if self.indv:
            y = [l_c(x[:,[idx],:]) for idx, l_c in enumerate(self.l1)]
            y = torch.cat(y,dim=1).permute(0,2,1)
        else:
            y = self.l1(x).permute(0,2,1)

        if self.IN:
            y = y + x_mean
        return y


class SparseTSF(nn.Module):
    def __init__(self, configs):
        super(SparseTSF, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        assert self.model_type in ['linear', 'mlp']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False
        )

        if self.model_type == 'linear':
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        elif self.model_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y)
            )

    def forward(self, x, *args):
        batch_size = x.shape[0]
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(
            -1, self.enc_in, self.seq_len) + x

        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        if self.model_type == 'linear':
            y = self.linear(x)
        elif self.model_type == 'mlp':
            y = self.mlp(x)

        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) + seq_mean
        return y


def adjust_learning_rate(optimizer, epoch, step, args, printout=False, scheduler=None):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "type4":
        lr_adjust = {epoch: args.learning_rate if step < 3000 else args.learning_rate * (0.9 ** ((step - 2000) // 1000))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}
    elif args.lradj == 'TST':
        assert scheduler is not None
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


def vali(model, vali_loader, criterion, device_, args):
    total_loss_criterion = []
    total_loss_mae = []
    model.eval()
    #criterion = criterion.cpu()
    with torch.no_grad():
        total_time=0
        iteration_count=0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

            batch_x = batch_x.float().to(device_)
            batch_y = batch_y.float().to(device_)
            batch_x_mark = batch_x_mark.float().to(device_)
            batch_y_mark = batch_y_mark.float().to(device_)

            L = batch_y.shape[1]


            start_time = time.time()  # Start timer for this iteration
            outputs = model(batch_x, batch_x_mark, None, None)

            end_time = time.time()  # End timer for this iteration
            iteration_time = end_time - start_time
            total_time += iteration_time
            iteration_count += 1

            outputs = outputs[:, -L:, :].detach()
            batch_y = batch_y.to(device_).detach()

            loss = criterion(outputs, batch_y)

            total_loss_criterion.append(loss.item())
            total_loss_mae.append(torch.abs(batch_y - outputs).mean().item())

    avg_time_per_iter = total_time / iteration_count * 1000
    print(f"Average time per iteration val: {avg_time_per_iter:.4f} ms")

    total_loss_criterion = np.average(total_loss_criterion)
    total_loss_mae = np.average(total_loss_mae)

    if hasattr(model, "ch_weight"):
        print(model.ch_weight)

    model.train()
    return total_loss_criterion, total_loss_mae

def test(model, test_loader, criterion, device_, args):
    all_loss_criterion = []
    batch_length = []
    all_loss_mae = []
    model.eval()
    #criterion = criterion.cpu()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device_)
            batch_y = batch_y.float().to(device_)
            batch_x_mark = batch_x_mark.float().to(device_)
            batch_y_mark = batch_y_mark.float().to(device_)

            L = batch_y.shape[1]

            outputs = model(batch_x, batch_x_mark, None, None)

            outputs = outputs[:, -L:, :].detach()
            batch_y = batch_y.to(device_).detach()

            loss = criterion(outputs, batch_y)
            batch_length.append(batch_x.shape[0])
            all_loss_criterion.append(loss.item())
            all_loss_mae.append(torch.abs(batch_y - outputs).mean().item())

    batch_weight = np.array(batch_length, np.float32)/np.sum(batch_length)
    total_loss_criterion = np.sum(np.array(all_loss_criterion)*batch_weight)
    total_loss_mae = np.sum(np.array(all_loss_mae)*batch_weight)

    if hasattr(model, "ch_weight"):
        print(model.ch_weight)

    model.train()
    return total_loss_criterion, total_loss_mae

def train(model, model_optim, criterion,
          args, base_path, setting,
          train_loader, vali_loader, test_loader, device_, val_every=50):
    path = os.path.join(base_path, setting, args.checkpoints)
    path_res_ = os.path.join(base_path, setting, "results")
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path_res_):
        os.makedirs(path_res_)

    early_stopping = EarlyStopping(patience=args.train_steps, verbose=True)

    cur_step = 0
    epoch = 0
    min_val_mse, min_val_mae = np.inf, np.inf
    min_test_mse, min_test_mae = np.inf, np.inf
    while cur_step < args.train_steps:
        train_loss = []
        loss_item = None
        model.train()
        total_time=0
        iteration_count=0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            start_time = time.time()  # Start timer for this iteration

            model_optim.zero_grad()
            batch_x = batch_x.float().to(device_)
            batch_y = batch_y.float().to(device_)
            batch_x_mark = batch_x_mark.float().to(device_)
            batch_y_mark = batch_y_mark.float().to(device_)

            L = batch_y.shape[1]

            loss_dict = model(batch_x, batch_y)
            loss = loss_dict['loss']

            if loss_item is None:
                loss_item = tuple(loss_dict.keys())
            train_loss.append([loss_dict[k].item() for k in loss_item])

            loss.backward()
            model_optim.step()

            end_time = time.time()  # End timer for this iteration
            iteration_time = end_time - start_time
            total_time += iteration_time
            iteration_count += 1

            # stdout information
            if (cur_step+1) % val_every == 0:

                avg_time_per_iter = total_time / iteration_count
                print(f"Average time per iteration: {avg_time_per_iter:.4f} seconds")
                # Reset counters for next period (optional)
                total_time = 0
                iteration_count = 0

                train_loss_ = np.average(train_loss, 0)
                val_mse, val_mae = vali(model, vali_loader, criterion, device_, args)
                test_mse, test_mae = test(model, test_loader, criterion, device_, args)

                for param_group in model_optim.param_groups:
                    cur_lr = param_group['lr']

                if val_mse <= min_val_mse:
                    min_val_mse, min_val_mae = val_mse, val_mae
                    min_test_mse, min_test_mae = test_mse, test_mae

                train_loss_str = " ".join(f"{item_name_}:{val:.4f}" for item_name_, val in zip(loss_item,train_loss_))
                print(
                    f"Step: {cur_step+1}, lr: {cur_lr:.5e} | Train {train_loss_str} | Vali loss: {val_mse:.3f}/{val_mae:.3f} | Test loss: {test_mse:.3f}/{test_mae:.3f}"
                )
                early_stopping(val_mse, model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            cur_step += 1
            if cur_step >= args.train_steps:
                break

        epoch += 1
        adjust_learning_rate(model_optim, epoch, cur_step, args)

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path, map_location=device_))

    result_file = os.path.join(path_res_, "forecast.txt")
    f = open(result_file, 'a')
    f.write(setting + "  \n")
    f.write(f'mse:{min_test_mse:.5f}, mae:{min_test_mae}')
    f.write('\n')
    f.write('\n')
    f.close()
    write_json(os.path.join(path_res_, "metrics.json"), {
        "selection": "minimum validation MSE",
        "validation_mse": float(min_val_mse), "validation_mae": float(min_val_mae),
        "test_mse": float(min_test_mse), "test_mae": float(min_test_mae),
        "seed": args.seed, "pred_len": args.pred_len,
    })

    return model



def main(argv=None):
    parser = get_parser(description="Run forecasting model")


    # data loader
    parser.add_argument('--data', type=str, default='PAMOS', help='dataset type')
    parser.add_argument('--l1_decay', type=float, default=0)
    parser.add_argument('--l2_decay', type=float, default=0)
    parser.add_argument('--train_steps', type = int, default=10000)
    parser.add_argument("--instance_norm", type = str2bool, default=True)
    parser.add_argument("--individual", type = str2bool, default=False)
    # parser.add_argument("--lradj", type=str, default="type3")

    parser.add_argument("--regu_coef", type=float, default=0.5)
    parser.add_argument("--val_type", type=str, default="res") # res or none
    parser.add_argument("--data_dsample", type=int, default=0)
    parser.add_argument("--detaching", type=str2bool, default=True)
    parser.add_argument("--checker_batch_ratio", type=float, default=1.0,
                        help="Fraction of batch rows used only for JVP; MSE stays full-batch.")
    parser.add_argument("--checker_batch_size", type=int, default=0,
                        help="Fixed JVP subset size; positive values override the ratio; 0 uses the ratio.")
    parser.add_argument("--jvp_bn_policy", default="subset_train_stats",
                        choices=["subset_train_stats", "running_eval_stats"])
    parser.add_argument("--test_pred_lens", type=int, nargs="+", default=None,
                        help="Recursive evaluation horizons, e.g. 336 720 for a model trained at 96.")
    parser.add_argument("--load_ckpt", default=None, help="Checkpoint for --is_training 0 evaluation.")
    parser.add_argument("--output_dir", default="./model_result")
    parser.add_argument("--val_every", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--data_ending", type=int, default=400)
    parser.add_argument("--data_noise", type=float, default=0.5)

    parser.add_argument("--decouple_checker", type=str, default="none",
                        choices=["none", "decouple", "isolate"],
                        help="none: combined ||(J_mlp+J_lin)r||^2; "
                             "decouple: ||J_mlp r||^2 + ||J_lin r||^2 (shared residual); "
                             "isolate: branch-specific residuals (linear uses r_lin=y-f_lin, "
                             "MLP uses r_all=y-y_hat).")
    parser.add_argument("--regu_coef_mlp", type=float, default=None,
                        help="Checker lambda for MLP branch (decouple/isolate modes). "
                             "If None, falls back to --regu_coef.")

    # TSLANet-specific args
    parser.add_argument('--emb_dim', type=int, default=64, help='dimension of model')
    parser.add_argument('--depth', type=int, default=3, help='num of layers')
    parser.add_argument('--patch_size', type=int, default=64, help='size of patches')
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    # PDF-specific args
    parser.add_argument('--period', type=int, nargs='+', default=[24, 12], help='period list for PDF')
    parser.add_argument('--patch_len', type=int, nargs='+', default=[16], help='patch_len per period for PDF')
    parser.add_argument('--stride', type=int, nargs='+', default=None, help='stride per period for PDF (None = patch_len//2)')
    parser.add_argument('--kernel_list', type=int, nargs='+', default=[3, 7, 9], help='local conv kernel sizes for PDF')
    parser.add_argument('--fc_dropout', type=float, default=0.0, help='dropout after local conv / head')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='dropout in forecast head')
    parser.add_argument('--add', type=str2bool, default=False, help='PDF: average branches instead of concat')
    parser.add_argument('--wo_conv', type=str2bool, default=False, help='PDF: disable local convolution')
    parser.add_argument('--serial_conv', type=str2bool, default=False, help='PDF: serial conv mode')

    # FilterNet-specific args
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden dimension for FilterNet')

    # SparseTSF-specific args
    parser.add_argument('--period_len', type=int, default=24, help='period length for SparseTSF')
    parser.add_argument('--model_type', type=str, default='linear', choices=['linear', 'mlp'],
                        help='SparseTSF model type: linear or mlp')

    args = parser.parse_args(argv)

    if args.train_steps < 1 or args.val_every < 1 or args.val_every > args.train_steps:
        parser.error("require 1 <= val_every <= train_steps")
    if args.test_pred_lens and (min(args.test_pred_lens) < 1 or args.data.lower() == "synthetic"):
        parser.error("rollout needs positive target horizons and a benchmark dataset")
    if not args.is_training and not args.load_ckpt:
        parser.error("--is_training 0 requires --load_ckpt")
    if args.val_type not in {"res", "branch_res"} and (args.checker_batch_ratio != 1 or args.checker_batch_size):
        parser.error("checker subsampling is supported for res and branch_res")
    base_path_ = args.output_dir

    if args.model.lower() == "baselinearmodel":
        _model = BaseLinearModel
    elif args.model.lower() == "mlp":
        _model = SmallMLP
    elif args.model.lower() == "mlp_nobias":
        _model = SmallMLPNoBias
    elif args.model.lower() == "tslanet":
        _model = TSLANet
    elif args.model.lower() == "pdf":
        _model = PDFForecast
    elif args.model.lower() == "filternet":
        _model = PaiFilter
    elif args.model.lower() == "itransformer":
        _model = iTransformerForecast
    elif args.model.lower() == "sparsetsf":
        _model = SparseTSF
    else:
        raise NotImplementedError("Unsupported variant")

    if args.val_type.lower() == "res":
        _wrapper = ResidualCheckGeneral
    elif args.val_type.lower() == "branch_res":
        _wrapper = BranchResidualCheck
    elif args.val_type.lower() == "tf":
        _wrapper = TFCheckGeneral
    elif args.val_type.lower() == "branch_tf":
        _wrapper = BranchTFCheck
    elif args.val_type.lower() == "none":
        _wrapper = DummyCheckGeneral
    else:
        raise NotImplementedError("Unsupported variant")


    if args.data.lower() == "synthetic":
        args.data_id = None
        args.enc_in = 1

        # length/noise
        def simulation_func_n(x, sigma):
            return np.sin(2*x) + np.cos(5*x) + 0.5*x + np.random.normal(loc=0.0, scale=1.0, size=x.shape)*sigma


        sim_func = partial(simulation_func_n, sigma=args.data_noise)
        data_core = Dataset_Function_MC_Core(
            functions = [sim_func], random_generator=None,
            size = [args.seq_len, 0, args.pred_len],
            scale=False, x_end=args.data_ending
        )


        train_dataset = Dataset_Function_MC(data_core, flag="train")
        val_dataset = Dataset_Function_MC(data_core, flag="val")
        test_dataset = Dataset_Function_MC(data_core, flag="test")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False
        )

    else:
        args.data_id = None
        if args.is_training:
            train_dataset, train_loader = tslib_loader.data_provider(args=args, flag="train")
            val_dataset, val_loader = tslib_loader.data_provider(args=args, flag="val")
        test_dataset, test_loader = tslib_loader.data_provider(args = args, flag = "test")

    setting = run_setting(args)
    print(setting)
    config_name = "config.json" if args.is_training else "evaluation_config.json"
    write_json(os.path.join(base_path_, setting, config_name), vars(args))



    from optimizer import WeightDecayOptimizer

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    device_ = args.device
    if device_ == "auto":
        device_ = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device_} as training device!")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)




    print("================start_train================")
    base_model = _wrapper(args, _model)
    model_ = base_model.to(device_)


    total_params = count_parameters(model_)
    print(f"Total trainable parameters: {total_params:,}")

    model_criterion = nn.MSELoss()
    model_optimizer = WeightDecayOptimizer(model_.parameters(), torch.optim.Adam,
                                        l1_decay=args.l1_decay, l2_decay=args.l2_decay,
                                        lr=args.learning_rate)

    if args.is_training:
        train(model_, model_optimizer, model_criterion,
              args, base_path_, setting, train_loader, val_loader, test_loader,
              device_, val_every=args.val_every)
    else:
        model_.load_state_dict(torch.load(args.load_ckpt, map_location=device_))
        mse, mae = test(model_, test_loader, model_criterion, device_, args)
        write_json(os.path.join(base_path_, setting, "results", "evaluation.json"),
                   {"test_mse": float(mse), "test_mae": float(mae), "checkpoint": args.load_ckpt})

    if args.test_pred_lens:
        rollout_results = []
        for target_horizon in args.test_pred_lens:
            evaluation_args = deepcopy(args)
            evaluation_args.pred_len = target_horizon
            _, rollout_loader = tslib_loader.data_provider(args=evaluation_args, flag="test")
            mse, mae = rollout_test(model_, rollout_loader, device_, args.pred_len, target_horizon)
            result = {"train_pred_len": args.pred_len, "test_pred_len": target_horizon,
                      "mse": float(mse), "mae": float(mae)}
            rollout_results.append(result)
            print(f"[rollout] H_train={args.pred_len}, H_test={target_horizon}: mse={mse:.6f}, mae={mae:.6f}")
        write_json(os.path.join(base_path_, setting, "results", "rollout.json"), rollout_results)

    print("================end_train================")


if __name__ == "__main__":
    main()
