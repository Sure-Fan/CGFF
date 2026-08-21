import argparse
import logging
import os
import time

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('agg')
import yaml

from dataset.semi import SemiDataset
from model.semseg.prototype import Prototyp_text
from evaluate import evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import init_log
from util.thresh_helper import ThreshController
from util.model_info import log_model_parameter_info, save_model_parameter_info
import random


parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, default=r"configs\glas_10.yaml")
parser.add_argument('--labeled-id-path', type=str, default=r"partitions\glas_10\labeled.txt")
parser.add_argument('--unlabeled-id-path', type=str, default=r"partitions\glas_10\unlabeled.txt")
parser.add_argument('--save-path', type=str, default=r"exp\glas\10\HibouText")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument("--no-ddp", action="store_true")



def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.enabled = True
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main():
    args = parser.parse_args()
    run_start_time = time.perf_counter()
    init_seeds(1, cuda_deterministic=True)
    os.makedirs(args.save_path, exist_ok=True)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    num_workers = 0 if os.name == "nt" else cfg.get('num_workers', 2)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    model = Prototyp_text(cfg)
    model_parameter_report = log_model_parameter_info(logger, model)

    text_head = None

    backbone_params = list(model.backbone.parameters())
    other_model_params = [param for name, param in model.named_parameters() if 'backbone' not in name]
    if text_head is not None:
        other_params = list(other_model_params) + list(text_head.parameters())
    else:
        other_params = list(other_model_params)

    optimizer = SGD(
        [
            {'params': backbone_params, 'lr': cfg['lr']},
            {'params': other_params, 'lr': cfg['lr'] * cfg['lr_multi']},
        ],
        lr=cfg['lr'],
        momentum=0.9,
        weight_decay=1e-4,
    )

    rank, word_size = 0, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if text_head is not None:
        text_head = text_head.to(device)


    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).to(device)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).to(device)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').to(device)

    is_supervised = cfg.get('supervised', False)
    if is_supervised:
        trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                                 cfg['window_size'], args.unlabeled_id_path)
        trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                                 cfg['window_size'], args.labeled_id_path, nsample=None)
    else:
        trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                                 cfg['window_size'], args.unlabeled_id_path)
        trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                                 cfg['window_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', id_path=cfg.get('val_id_path', None))


    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )

    if is_supervised:
        trainloader_u = None
    else:
        trainloader_u = DataLoader(
            trainset_u,
            batch_size=cfg['batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True
        )

    valloader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False
    )


    total_iters = len(trainloader_l) * cfg['epochs']
    previous_best = 0.0
    thresh_controller = ThreshController(nclass=cfg['nclass'], momentum=0.999, thresh_init=cfg['thresh_init'])

    for epoch in range(cfg['epochs']):
            
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        if text_head is not None:
            text_head.train()

        total_loss, total_loss_x, total_loss_s, total_loss_w_fp = 0.0, 0.0, 0.0, 0.0
        total_mask_ratio = 0.0

        if rank == 0:
            tbar = tqdm(total=len(trainloader_l))

        if is_supervised:
            for i, (img_x, mask_x) in enumerate(trainloader_l):
                img_x, mask_x = img_x.to(device), mask_x.to(device)
                model.train()
                res = model(img_x, need_fp=False, use_corr=False)
                pred_x = res['out']
                loss = criterion_l(pred_x, mask_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_loss_x += loss.item()
                iters = epoch * len(trainloader_l) + i
                lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
                if rank == 0:
                    tbar.set_description('Loss: {:.3f}'.format(total_loss / (i + 1)))
                    tbar.update(1)
        else:
            loader = zip(trainloader_l, trainloader_u)
            for i, ((img_x, mask_x),
                    (img_u_w, _, _, ignore_mask, _, _)) in enumerate(loader):

                img_x, mask_x = img_x.to(device), mask_x.to(device)
                img_u_w = img_u_w.to(device)
                ignore_mask = ignore_mask.to(device)

                model.train()

                res_x = model(img_x, need_fp=False)
                pred_x = res_x['out']

                res_u = model(img_u_w, need_fp=False)
                pred_u = res_u['out']
                pred_u_detached = pred_u.detach()
                conf_u = pred_u_detached.softmax(dim=1).max(dim=1)[0]
                mask_u = pred_u_detached.argmax(dim=1)

                thresh_controller.thresh_update(pred_u_detached, ignore_mask, update_g=True)
                thresh_global = thresh_controller.get_thresh_global(device=conf_u.device)

                conf_filter_u = ((conf_u >= thresh_global) & (ignore_mask != 255))

                loss_x = criterion_l(pred_x, mask_x)

                loss_u_s1 = criterion_u(pred_u, mask_u)
                loss_u_s1 = loss_u_s1 * conf_filter_u
                loss_u_s1 = torch.sum(loss_u_s1) / torch.sum(ignore_mask != 255).item()

                loss_corrmatch_core = (
                    0.5 * loss_x
                    + loss_u_s1 * 0.25
                )
                loss = loss_corrmatch_core

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_loss_x += loss_x.item()
                total_loss_s += loss_u_s1.item()
                total_mask_ratio += ((conf_u >= thresh_global) & (ignore_mask != 255)).sum().item() / \
                                    (ignore_mask != 255).sum().item()

                iters = epoch * len(trainloader_l) + i
                lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

                if rank == 0:
                    tbar.set_description(
                        ' Total loss: {:.3f}, Loss x: {:.3f}, Loss u: {:.3f}, Mask: {:.3f}'.format(
                            total_loss / (i + 1), total_loss_x / (i + 1),
                            total_loss_s / (i + 1), total_mask_ratio / (i + 1))
                    )
                    tbar.update(1)

        if rank == 0:
            tbar.close()

        if cfg['backbone'] in ('hibou', 'uni'):
            eval_mode = 'sliding_window'
        else:
            if cfg['dataset'] == 'cityscapes':
                eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
            else:
                eval_mode = 'original'
        torch.cuda.empty_cache()
        res_val = evaluate(model, valloader, eval_mode, cfg)

        mJaccard = res_val.get('mJaccard', res_val['mIOU'])
        mDice = res_val.get('mDice', None)
        iou_class = res_val.get('iou_class', None)
        dice_class = res_val.get('dice_class', None)

        best_metric_name = str(cfg.get('best_metric') or 'jaccard').strip().lower()
        if best_metric_name == 'dice':
            current_best_val = mDice if mDice is not None else mJaccard
        else:
            current_best_val = mJaccard
            if best_metric_name != 'jaccard':
                best_metric_name = 'jaccard'

        logger.info(f"VAL >>> mJaccard: {mJaccard:.2f}  mDice: {mDice:.2f}  (best_metric={best_metric_name})")

        if cfg.get('nclass', 0) == 2 and ('Dice_fg' in res_val) and ('Jaccard_fg' in res_val):
            logger.info(f"VAL (FG) >>> Jaccard_fg: {res_val['Jaccard_fg']:.2f}  Dice_fg: {res_val['Dice_fg']:.2f}")

        # (optional) print per-class
        if iou_class is not None:
            logger.info(f"VAL IoU per class: {iou_class}")
        if dice_class is not None:
            logger.info(f"VAL Dice per class: {dice_class}")


        if rank == 0:
            logger.info('***** Evaluation {} ***** >>>> meanJaccard: {:.4f} \n'.format(eval_mode, mJaccard))
            logger.info('***** ClassIOU ***** >>>> \n{}\n'.format(iou_class))

        best_ckpt_name = f"{cfg['backbone']}_{best_metric_name}4.pth"
        if current_best_val > previous_best and rank == 0:
            previous_best = current_best_val
            ckpt_path = os.path.join(args.save_path, best_ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            elapsed_seconds = time.perf_counter() - run_start_time
            parameter_path = save_model_parameter_info(
                model_parameter_report, ckpt_path, elapsed_seconds=elapsed_seconds
            )
            logger.info(f"***** New best ({best_metric_name}): {current_best_val:.3f} -> saved {best_ckpt_name} *****")
            logger.info("Elapsed time at checkpoint: %.3f seconds", elapsed_seconds)
            logger.info("Model parameter report saved to %s", parameter_path)
        # torch.distributed.barrier()
        torch.cuda.empty_cache()

    if rank == 0 and cfg.get('test_id_path'):
        best_metric_name = str(cfg.get('best_metric') or 'jaccard').strip().lower()
        if best_metric_name != 'dice':
            best_metric_name = 'jaccard'
        best_ckpt_name = f"{cfg['backbone']}_{best_metric_name}4.pth"
        ckpt_path = os.path.join(args.save_path, best_ckpt_name)
        if os.path.isfile(ckpt_path):
            logger.info('===========> Loading best checkpoint for test evaluation: %s', best_ckpt_name)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            testset = SemiDataset(cfg['dataset'], cfg['data_root'], 'test', id_path=cfg['test_id_path'])
            testloader = DataLoader(
                testset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
                drop_last=False
            )
            if cfg['backbone'] in ('hibou', 'uni'):
                eval_mode_test = 'sliding_window'
            else:
                eval_mode_test = 'center_crop' if cfg['dataset'] == 'cityscapes' else 'original'
            torch.cuda.empty_cache()
            res_test = evaluate(model, testloader, eval_mode_test, cfg)
            mJaccard_t = res_test.get('mJaccard', res_test['mIOU'])
            mDice_t = res_test.get('mDice', None)
            iou_class_t = res_test.get('iou_class', None)
            dice_class_t = res_test.get('dice_class', None)
            logger.info('TEST >>> mJaccard: {:.2f}  mDice: {:.2f}'.format(mJaccard_t, mDice_t or 0))
            if cfg.get('nclass', 0) == 2:
                logger.info('TEST (FG) >>> Jaccard_fg: {:.2f}  Dice_fg: {:.2f}'.format(
                    res_test.get('Jaccard_fg', 0), res_test.get('Dice_fg', 0)))
            score_path = os.path.join(args.save_path, 'test_scores.txt')
            with open(score_path, 'w') as f:
                f.write('split=test\n')
                f.write('mJaccard (mIOU)={:.4f}\n'.format(mJaccard_t))
                f.write('mDice={:.4f}\n'.format(mDice_t or 0))
                if iou_class_t is not None:
                    f.write('IoU_per_class={}\n'.format(iou_class_t.tolist()))
                if dice_class_t is not None:
                    f.write('Dice_per_class={}\n'.format(dice_class_t.tolist()))
                if cfg.get('nclass', 0) == 2:
                    f.write('Jaccard_fg={:.4f}\n'.format(res_test.get('Jaccard_fg', 0)))
                    f.write('Dice_fg={:.4f}\n'.format(res_test.get('Dice_fg', 0)))
            logger.info('Test scores saved to %s', score_path)
        else:
            logger.warning('Best checkpoint not found (%s), skip test evaluation.', ckpt_path)


if __name__ == '__main__':
    main()
