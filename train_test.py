import numpy as np
import torch
from utils.metrics import multi_class_eval


def build_m2u_batch(drug_ids, graph_cache):
    if graph_cache is None:
        return None

    from torch_geometric.data import Batch

    drug_id_to_idx = graph_cache['drug_id_to_idx']
    graphs = graph_cache['graphs']
    data_list = []
    missing = []
    for drug_id in drug_ids:
        drug_id = int(drug_id)
        if drug_id not in drug_id_to_idx:
            missing.append(drug_id)
            continue
        data_list.append(graphs[drug_id_to_idx[drug_id]])
    if missing:
        raise ValueError(f'Missing M2UMol graph cache for drug ids: {missing[:10]}')
    return Batch.from_data_list(data_list)


def train_one_epoch(model, loss_func, optimizer, train_x_left, train_x_right, train_y,
                    fold_index, epoch, batch_size, device, m2umol_graph_cache=None,
                    args=None, train_subclass_labels=None, scl_criterion=None):
    use_scl = bool(args is not None and getattr(args, 'use_scl', False))
    scl_ready = (
        use_scl
        and epoch >= args.scl_warmup_epoch
        and train_subclass_labels is not None
        and scl_criterion is not None
    )
    train_index = np.arange(len(train_y))
    np.random.shuffle(train_index)
    step = 0
    total_seen = 0
    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    scl_loss_sum = 0.0
    model.train()
    for j in range(0, len(train_index), batch_size):
        index = train_index[j:j + batch_size]
        x_left = train_x_left[index]
        x_right = train_x_right[index]

        y = train_y[index].squeeze(dim=-1)
        y_device = y.to(device)
        if m2umol_graph_cache is None:
            if scl_ready:
                y_pred, pair_repr = model(x_left, x_right, return_pair_emb=True)
            else:
                y_pred = model(x_left, x_right)
        else:
            left_m2u_graph = build_m2u_batch(x_left, m2umol_graph_cache)
            right_m2u_graph = build_m2u_batch(x_right, m2umol_graph_cache)
            if scl_ready:
                y_pred, pair_repr = model(x_left, x_right, left_m2u_graph, right_m2u_graph, return_pair_emb=True)
            else:
                y_pred = model(x_left, x_right, left_m2u_graph, right_m2u_graph)

        loss_ce = loss_func(y_pred, y_device)
        if scl_ready:
            subclass_labels = train_subclass_labels[index].to(device)
            z1 = model.scl_project(pair_repr)
            z2 = model.scl_project(pair_repr)
            features = torch.stack([z1, z2], dim=1)
            loss_scl = scl_criterion(
                features,
                y_device,
                subclass_labels,
            )
            train_loss = loss_ce + args.scl_lambda * loss_scl
        else:
            loss_scl = y_pred.sum() * 0.0
            train_loss = loss_ce

        batch_count = len(index)
        total_seen += batch_count
        total_loss_sum += train_loss.item() * batch_count
        ce_loss_sum += loss_ce.item() * batch_count
        scl_loss_sum += loss_scl.item() * batch_count

        y_pred = torch.softmax(y_pred, dim=-1).detach().cpu().numpy()
        train_acc, train_f1, train_precision, train_recall, train_kappa = multi_class_eval(
            y.numpy(),
            y_pred)
        if use_scl:
            print(
                "fold:{} epoch:{} step:{} "
                "train loss:{:.6f}, loss_ce:{:.6f}, loss_scl:{:.6f}, scl_warmup:{}, "
                "train acc:{:.3f}, train f1:{:.3f}, train precision:{:.3f}, train recall:{:.3f}, train kappa:{:.3f}"
                .format(
                    fold_index, epoch, step,
                    train_loss.item(), loss_ce.item(), loss_scl.item(), epoch < args.scl_warmup_epoch,
                    train_acc, train_f1, train_precision, train_recall, train_kappa
                ))
        else:
            print(
                "fold:{} epoch:{} step:{} "
                "train loss:{:.6f}, train acc:{:.3f}, train f1:{:.3f}, train precision:{:.3f}, train recall:{:.3f}, train kappa:{:.3f}"
                .format(
                    fold_index, epoch, step,
                    train_loss.item(), train_acc, train_f1, train_precision, train_recall, train_kappa
                ))

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

    if use_scl and total_seen > 0:
        print(
            "[SCL epoch summary] fold:{} epoch:{} "
            "ce_loss:{:.6f}, scl_loss:{:.6f}, total_loss:{:.6f}".format(
                fold_index,
                epoch,
                ce_loss_sum / total_seen,
                scl_loss_sum / total_seen,
                total_loss_sum / total_seen,
            )
        )


def test(model, loss_func, valid_x_left, valid_x_right, valid_y,
         fold_index, epoch, batch_size, device, m2umol_graph_cache=None):
    model.eval()
    with torch.no_grad():
        valid_index = np.arange(len(valid_y))
        y = torch.tensor([])
        y_pred = torch.tensor([]).to(device)
        for j in range(0, len(valid_index), batch_size):
            index = valid_index[j:j + batch_size]
            x_left = valid_x_left[index]
            x_right = valid_x_right[index]
            if len(y) == 0:
                y = torch.LongTensor([])
            y = torch.concat([y, valid_y[index].squeeze(dim=-1)])
            if m2umol_graph_cache is None:
                pred = model(x_left, x_right)
            else:
                left_m2u_graph = build_m2u_batch(x_left, m2umol_graph_cache)
                right_m2u_graph = build_m2u_batch(x_right, m2umol_graph_cache)
                pred = model(x_left, x_right, left_m2u_graph, right_m2u_graph)
            y_pred = torch.concat([y_pred, pred])

        valid_loss = loss_func(y_pred, y.to(device))

        y_pred = torch.softmax(y_pred, dim=-1).cpu().detach().numpy()
        valid_acc, valid_f1, valid_precision, valid_recall, valid_kappa = multi_class_eval(y.numpy(), y_pred)
        print(
            "fold:{} epoch:{}        "
            "valid loss:{:.6f}, valid acc:{:.3f}, valid f1:{:.3f}, valid precision:{:.3f}, valid recall:{:.3f}, valid kappa:{:.3f}"
            .format(
                fold_index, epoch,
                valid_loss.item(), valid_acc, valid_f1, valid_precision, valid_recall, valid_kappa
            ))
        score = [valid_acc, valid_f1, valid_precision, valid_recall, valid_kappa]

        return score
