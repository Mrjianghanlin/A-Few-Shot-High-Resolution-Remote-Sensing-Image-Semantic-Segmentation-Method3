import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from dataset.semicd import SemiCDDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
import os




parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
# parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, default="splits/whu/5%/labeled.txt")
parser.add_argument('--unlabeled-id-path', type=str, default="splits/whu/5%/unlabeled.txt")
parser.add_argument('--save-path', type=str, default="exp/whu/unimatch/deeplabv3plus_r50/5%")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def evaluate(model, loader, cfg):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixel = AverageMeter()
    total_pixel = AverageMeter()

    with torch.no_grad():
        for imgA, imgB, mask, id in loader:
            imgA = imgA.cuda()
            imgB = imgB.cuda()

            pred = model(imgA, imgB).argmax(dim=1)

            intersection, union, target = intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

            correct_pixel.update((pred.cpu() == mask).sum().item())
            total_pixel.update(pred.numel())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    overall_acc = correct_pixel.sum / total_pixel.sum * 100.0

    return iou_class, overall_acc


def main():
    args = parser.parse_args()

    cfg = {
        'dataset': 'whu',
        'data_root': r'C:\Users\wang\Downloads\WHU-CD-256\WHU-CD-256',
        # 'data_root': r'/media/sj/Elements/wn/banjinadu/AdvSemiSeg-master/UniMatch-main/UniMatch-main/more-scenarios/remote-sensing/dataset/WHU-CD-256',
        'nclass': 2,
        'pretrained': True,
        'crop_size': 512,  # your desired crop size
        'epochs': 100,  # your desired crop size
        'batch_size': 2,  # your desired crop size
        'lr': 0.02,  # your desired crop size
        'replace_stride_with_dilation': [False, False, True],  # your desired crop size
        'dilations': [6, 12, 18],  # your desired crop size
        'lr_multi': 1.0,  # your desired crop size
        'conf_thresh': 0.95,  # your desired crop size
        'model': "pspnet",  # your desired crop size
        'backbone': "resnet50",  # your desired crop size
        # other configuration options
    }

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    os.makedirs(args.save_path, exist_ok=True)

    writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True
    rank=0

    # model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet}
    # assert cfg['model'] in model_zoo.keys()
    # model = model_zoo[cfg['model']](cfg)
    model=DeepLabV3Plus(cfg)

    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    # local_rank = int(os.environ["LOCAL_RANK"])
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model.cuda(local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
    #                                                   output_device=local_rank, find_unused_parameters=False)
    #
    # criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)
    # 将模型移动到GPU上（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)

    trainset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
    print("Number of data points in validation set:", len(trainset))
    valset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'val')
    print("Number of data points in validation set:", len(valset))

    trainsampler = SequentialSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=2, drop_last=True,
                             sampler=trainsampler)

    valsampler = torch.utils.data.SequentialSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False,
                           sampler=valsampler)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best_iou, previous_best_acc = 0.0, 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_iou = checkpoint['previous_best_iou']
        previous_best_acc = checkpoint['previous_best_acc']

        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            '===========> Epoch: {:}, LR: {:.5f}, Previous best Changed IoU: {:.2f}, Overall Accuracy: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best_iou, previous_best_acc))

        model.train()
        total_loss = AverageMeter()

        for i, (imgA, imgB, mask) in enumerate(trainloader):
            imgA, imgB, mask = imgA.to(device), imgB.to(device), mask.to(device)

            pred = model(imgA, imgB)

            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss.item(), iters)

            if (i % (max(2, len(trainloader) // 8)) == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        iou_class, overall_acc = evaluate(model, valloader, cfg)

            # 在训练

        if rank == 0:
            logger.info('***** Evaluation ***** >>>> Unchanged IoU: {:.2f}'.format(iou_class[0]))
            logger.info('***** Evaluation ***** >>>> Changed IoU: {:.2f}'.format(iou_class[1]))
            logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}\n'.format(overall_acc))
            
            writer.add_scalar('eval/unchanged_IoU', iou_class[0], epoch)
            writer.add_scalar('eval/changed_IoU', iou_class[1], epoch)
            writer.add_scalar('eval/overall_accuracy', overall_acc, epoch)

        is_best = iou_class[1] > previous_best_iou
        previous_best_iou = max(iou_class[1], previous_best_iou)
        if is_best:
            previous_best_acc = overall_acc
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best_iou': previous_best_iou,
                'previous_best_acc': previous_best_acc,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
