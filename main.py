import os
import sys
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
from model.HetDDI import HetDDI
from utils.scl_loss import DDI_SCL_Loss
from utils.data_loader import load_data, get_train_test
from train_test import train_one_epoch, test
from utils.pytorchtools import EarlyStopping
from utils.logger import Logger
from utils.scl_cluster import update_subclass_and_temperature


def torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_m2umol_graph_cache(args, data_path):
    if args.mode == 'kg':
        return None
    cache_path = args.m2umol_graph_cache_path or os.path.join(data_path, 'm2umol_graph_cache.pt')
    graph_cache = torch_load(cache_path, map_location='cpu')
    print(f"Loaded M2UMol graph cache from {cache_path}")
    print(f"Successfully built M2UMol graphs: {len(graph_cache['graphs'])}")
    print(f"Failed M2UMol drugs: {len(graph_cache.get('failed_drugs', []))}")
    return graph_cache


def load_m2umol_weights(model, args):
    if not hasattr(model, 'm2umol_encoder') or not args.m2umol_pretrained_path:
        return

    from model.m2umol_encoder import load_m2umol_pretrained

    loaded, matched, total = load_m2umol_pretrained(
        model.m2umol_encoder,
        args.m2umol_pretrained_path,
        map_location='cpu',
    )
    print(
        f"M2UMol pretrained load success: {loaded}. "
        f"Matched parameters: {matched}/{total}. Path: {args.m2umol_pretrained_path}"
    )


def print_m2umol_embedding_shape(model, graph_cache, device):
    if graph_cache is None or not hasattr(model, 'm2umol_encoder') or len(graph_cache['graphs']) == 0:
        return
    from torch_geometric.data import Batch

    sample_graphs = graph_cache['graphs'][:min(2, len(graph_cache['graphs']))]
    sample_batch = Batch.from_data_list(sample_graphs).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        emb = model.m2umol_encoder(sample_batch)
    if was_training:
        model.train()
    print(f"M2UMol output embedding shape: {tuple(emb.shape)}")


def build_optimizer(model, args):
    if hasattr(model, 'm2umol_encoder'):
        m2u_param_ids = {id(p) for p in model.m2umol_encoder.parameters()}
        m2u_params = [p for p in model.m2umol_encoder.parameters() if p.requires_grad]
        weight_p, bias_bn_emb_p = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad or id(p) in m2u_param_ids:
                continue
            if 'bias' in name or 'bn' in name or 'embedding' in name or 'norm' in name:
                bias_bn_emb_p.append(p)
            else:
                weight_p.append(p)
        model_parameters = [
            {'params': weight_p, 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': bias_bn_emb_p, 'weight_decay': 0, 'lr': args.lr},
        ]
        if m2u_params:
            model_parameters.append(
                {'params': m2u_params, 'weight_decay': args.weight_decay, 'lr': args.m2umol_lr}
            )
            print(f"M2UMol encoder is trainable with lr={args.m2umol_lr}.")
        else:
            print("M2UMol encoder is frozen.")
        return optim.AdamW(model_parameters, lr=args.lr)

    weight_p, bias_bn_emb_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name or 'bn' in name or 'embedding' in name:
            bias_bn_emb_p += [p]
        else:
            weight_p += [p]
    model_parameters = [
        {'params': weight_p, 'weight_decay': args.weight_decay},
        {'params': bias_bn_emb_p, 'weight_decay': 0},
    ]
    return optim.AdamW(model_parameters, lr=args.lr)



def run(args):
    # np.random.seed(42)
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(42)

    data_path = os.path.join(args.data_path, args.kg_name+'+'+args.ddi_name)
    kg_g, smiles = load_data(data_path, device=device)  # STEP 2銆佽繑鍥炲姞杞界殑鐭ヨ瘑鍥捐氨鍜孲MILES鍒楄〃
    m2umol_graph_cache = load_m2umol_graph_cache(args, data_path)
    train_sample, valid_sample, test_sample = get_train_test(
        data_path,
        fold_num=args.fold_num,
        condition=args.condition)    # STEP 3銆佽繑鍥?鎶樿缁冩牱鏈拰娴嬭瘯鏍锋湰锛屽苟涓旀瘡鎶樼敓鎴愪簡鐩稿悓鏁伴噺鐨勮礋璁粌鏍锋湰鍜岃礋娴嬭瘯鏍锋湰

    # # save train and test sample
    # for i in range(5):
    #     np.savetxt(f'train_{i}.txt', train_sample[i], fmt='%d')
    #     np.savetxt(f'valid_{i}.txt', valid_sample[i], fmt='%d')
    #     np.savetxt(f'test_{i}.txt', test_sample[i], fmt='%d')

    scores = []
    for i in range(0, args.fold_num):
        print(f"============================ Fold {i} Start ============================")
        model_save_dir = os.path.join(args.data_path, args.kg_name+'+'+args.ddi_name, args.condition, f'Fold{i}')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        train_x_left = train_sample[i][:, 0]
        train_x_right = train_sample[i][:, 1]
        train_y = train_sample[i][:, 2:]

        valid_x_left = valid_sample[i][:, 0]
        valid_x_right = valid_sample[i][:, 1]
        valid_y = valid_sample[i][:, 2:]

        test_x_left = test_sample[i][:, 0]
        test_x_right = test_sample[i][:, 1]
        test_y = test_sample[i][:, 2:]

        train_y = torch.from_numpy(train_y).long()
        valid_y = torch.from_numpy(valid_y).long()
        test_y = torch.from_numpy(test_y).long()

        # STEP 4銆乴oad model
        class_num = 65 if args.ddi_name == 'Deng' else 86
        args.class_num = class_num
        model = HetDDI(
            kg_g,
            smiles,
            args.hidden_dim,
            args.num_layer,
            args.mode,
            class_num,
            args.condition,
            freeze_m2umol=args.freeze_m2umol,
            proj_dim=args.proj_dim,
        ).to(device)
        load_m2umol_weights(model, args)
        if i == 0:
            print_m2umol_embedding_shape(model, m2umol_graph_cache, device)
        loss_func = nn.CrossEntropyLoss()

        if i == 0:
            print(model)

        # train setting
        optimizer = build_optimizer(model, args)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        scl_criterion = DDI_SCL_Loss(
            tau1=args.scl_tau1,
            tau2=args.scl_tau2,
            beta=args.scl_beta,
        ) if args.use_scl else None
        train_subclass_labels = None
        scl_train_loader = {
            "train_x_left": train_x_left,
            "train_x_right": train_x_right,
            "train_y": train_y,
            "batch_size": args.batch_size,
            "m2umol_graph_cache": m2umol_graph_cache,
        }

        best_val_score = None
        for epoch in range(args.epoch):
            scl_updated = False
            if args.use_scl and epoch >= args.scl_warmup_epoch:
                if epoch == args.scl_warmup_epoch or epoch % args.cluster_update_interval == 0:
                    train_subclass_labels = update_subclass_and_temperature(
                        model, scl_train_loader, device, args, epoch=epoch
                    )
                    scl_updated = True
            if args.use_scl:
                print(
                    "[SCL] epoch:{} warmup:{} updated_subclass_labels:{}".format(
                        epoch, epoch < args.scl_warmup_epoch, scl_updated
                    )
                )
            train_one_epoch(model, loss_func, optimizer, train_x_left, train_x_right, train_y,
                            i, epoch, args.batch_size, device, m2umol_graph_cache,
                            args=args,
                            train_subclass_labels=train_subclass_labels,
                            scl_criterion=scl_criterion)
            
            val_score = test(model, loss_func, valid_x_left, valid_x_right, valid_y,
                            i, epoch, args.batch_size, device, m2umol_graph_cache)

            val_f1 = val_score[1]   # best score set to F1
            # test_acc = test_score[1]    # best score set to ACC
            if epoch > 50:
                early_stopping(val_f1, model)
                if early_stopping.counter == 0:
                    # best_test_score = test_score
                    best_val_score = val_score
                    # save model
                    torch.save(model.state_dict(), os.path.join(model_save_dir, f'best_model_{args.num_layer}_{args.hidden_dim}.pth'))
                if early_stopping.early_stop or epoch == args.epoch - 1:
                    break
            # one epoch end
            # print(best_test_score)
            if best_val_score is None:
                print(best_val_score)
            else:
                val_acc, val_f1, val_precision, val_recall, val_kappa = best_val_score
                print(
                    "val acc:{:.5f}, val f1:{:.5f}, val precision:{:.5f}, val recall:{:.5f}, val kappa:{:.5f}".format(
                        val_acc, val_f1, val_precision, val_recall, val_kappa
                    ))

        print(f"\n========== Fold {i} Training Finished. Evaluating on Test Set. ==========")
        best_model_path = os.path.join(model_save_dir, f'best_model_{args.num_layer}_{args.hidden_dim}.pth')
        model.load_state_dict(torch.load(best_model_path))

        final_test_score = test(model, loss_func, test_x_left, test_x_right, test_y,
                                i, 0, args.batch_size, device, m2umol_graph_cache)

        # one fold end
        scores.append(final_test_score)
        print('Test set score:', scores)

    # all fold end, output the final result
    scores = np.array(scores)
    mean_scores = scores.mean(axis=0)
    std_scores = scores.std(axis=0)

    print(f"\033[1;31mFinal DDI Result ({args.condition}):\n"
            f"Acc: {mean_scores[0]:.4f} +/- {std_scores[0]:.4f}\n"
            f"F1 : {mean_scores[1]:.4f} +/- {std_scores[1]:.4f}\n"
            f"Pre: {mean_scores[2]:.4f} +/- {std_scores[2]:.4f}\n"
            f"Rec: {mean_scores[3]:.4f} +/- {std_scores[3]:.4f}\n"
            f"Kap: {mean_scores[4]:.4f} +/- {std_scores[4]:.4f}\033[0m")


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--batch_size', type=int, default=2 ** 12)
    ap.add_argument('--fold_num', type=int, default=5)
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--num_layer', type=int, default=3)
    ap.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--weight_decay', type=float, default=None)
    ap.add_argument('--condition', type=str, choices=['S1', 'S2', 'S3'], default='S3')
    ap.add_argument('--mode', type=str, choices=['kg', 'use_m2umol_online', 'concat'], default='concat')
    ap.add_argument('--data_path', type=str, default='./data')
    ap.add_argument('--kg_name', type=str, default='DRKG')
    ap.add_argument('--ddi_name', type=str, choices=['Deng', 'Ryu'], default='Deng')
    ap.add_argument('--m2umol_graph_cache_path', type=str, default='')
    ap.add_argument('--m2umol_pretrained_path', type=str, default='./pre-trained_M2UMol.pt')
    ap.add_argument('--freeze_m2umol', action='store_true')
    ap.add_argument('--m2umol_lr', type=float, default=None)
    ap.add_argument('--use_scl', action='store_true', default=True)
    ap.add_argument('--no_scl', action='store_false', dest='use_scl')
    ap.add_argument('--scl_lambda', type=float, default=0.01)
    ap.add_argument('--scl_beta', type=float, default=0.5)
    ap.add_argument('--scl_tau1', type=float, default=0.07)
    ap.add_argument('--scl_tau2', type=float, default=0.2)
    ap.add_argument('--scl_warmup_epoch', type=int, default=10)
    ap.add_argument('--cluster_update_interval', type=int, default=5)
    ap.add_argument('--target_cluster_size', type=int, default=128)
    ap.add_argument('--max_clusters_per_class', type=int, default=20)
    ap.add_argument('--proj_dim', type=int, default=128)

    args = ap.parse_args()
    condition_defaults = {
        'S1': {'lr': 2e-4, 'weight_decay': 5e-4, 'm2umol_lr': 1e-4},
        'S2': {'lr': 1e-4, 'weight_decay': 5e-4, 'm2umol_lr': 5e-5},
        'S3': {'lr': 1e-4, 'weight_decay': 1e-3, 'm2umol_lr': 1e-5},
    }
    for name, value in condition_defaults[args.condition].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    print(args)


    terminal = sys.stdout
    log_file = './log/{}_{}_{}_{}_{}.txt'. \
        format(args.ddi_name, args.mode, args.condition, args.hidden_dim, args.num_layer)
    sys.stdout = Logger(log_file, terminal)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running on', device)

    try:
        run(args)   # STEP 1
    finally:
        sys.stdout.end()
        sys.stdout = terminal
