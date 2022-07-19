import os
import random
from tabnanny import check
import time
import glob
import sys
from collections import OrderedDict
from prettytable import PrettyTable
from tqdm import tqdm
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import s3dg
from args import get_args
from video_loader import VSum_DataLoader
from loss import MILNCELoss

from metrics import compute_metrics
from utils import (
    AllGather,
    get_cosine_schedule_with_warmup,
    setup_for_distributed,
    Logger,
    evaluate_summary,
    AverageMeter,
)

allgather = AllGather.apply


def main():
    args = get_args()
    if args.verbose:
        print(args)
    assert args.eval_video_root != "" or not (args.evaluate)
    assert args.video_root != ""
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)

    # create model

    if args.model_type == 1:
        model = s3dg.VSum(
            args.num_class,
            space_to_depth=False,
            word2vec_path=args.word2vec_path,
            init=args.weight_init,
            enc_layers=args.enc_layers,
            heads=args.heads,
            dropout=args.dropout,
        )
    elif args.model_type == 2:
        model = s3dg.VSum_MLP(
            args.num_class,
            space_to_depth=False,
            word2vec_path=args.word2vec_path,
            init=args.weight_init,
            dropout=args.dropout,
        )
    print("Created model")

    # load pretrained S3D weights
    if args.pretrain_cnn_path:
        net_data = torch.load(args.pretrain_cnn_path)
        model.base_model.load_state_dict(net_data)
    print("Loaded pretrained weights")

    for name, param in model.named_parameters():
        if "base" in name:
            param.requires_grad = False
        if "mixed_5" in name and args.finetune:
            param.requires_grad = True

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
            args.num_thread_reader = int(args.num_thread_reader / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    print("Finished loading distributed model")

    # Data loading code
    train_dataset = VSum_DataLoader(
        annt_path=args.annt_path,
        video_root=args.video_root,
        caption_root=args.caption_root,
        min_time=args.min_time,
        fps=args.fps,
        num_frames=args.num_frames,
        num_frames_per_segment=args.num_frames_per_segment,
        size=args.video_size,
        crop_only=args.crop_only,
        center_crop=args.centercrop,
        random_left_right_flip=args.random_flip,
        num_candidates=args.num_candidates,
        video_only=True,
    )
    # Test data loading code
    test_dataset = VSum_DataLoader(
        annt_path=args.eval_annt_path,
        video_root=args.eval_video_root,
        caption_root=args.caption_root,
        min_time=args.min_time,
        fps=args.fps,
        num_frames=args.num_frames,
        num_frames_per_segment=args.num_frames_per_segment,
        size=args.video_size,
        crop_only=args.crop_only,
        center_crop=args.centercrop,
        random_left_right_flip=args.random_flip,
        num_candidates=args.num_candidates,
        video_only=True,
        dataset="wikihow",
    )
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=False,
        num_workers=args.num_thread_reader,
        pin_memory=args.pin_memory,
        sampler=train_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_thread_reader,
        sampler=test_sampler,
    )

    # start a logger
    args.log_name = "{}_model_{}_bs_{}_lr_{}_nframes_{}_nfps_{}_nheads_{}_nenc_{}_dropout_{}_finetune_{}".format(
        args.log_name,
        args.model_type,
        args.batch_size,
        args.lrv,
        args.num_frames,
        args.num_frames_per_segment,
        args.heads,
        args.enc_layers,
        args.dropout,
        args.finetune,
    )
    tb_logdir = os.path.join(args.log_root, args.log_name)
    tb_logger = Logger(tb_logdir)
    if args.rank == 0:
        os.makedirs(tb_logdir, exist_ok=True)

    # define loss function (criterion) and optimizer
    # criterion = MILNCELoss()
    # criterion_c = nn.CrossEntropyLoss(
    #     weight=train_dataset.ce_weight.cuda(args.gpu), reduction="none"
    # )
    criterion_c = nn.MSELoss(reduction="none")
    # criterion_c = nn.CrossEntropyLoss(weight=None)
    criterion_r = nn.MSELoss()
    criterion_d = nn.CosineSimilarity(dim=1, eps=1e-6)

    if args.evaluate:
        print("starting eval...")
        evaluate(test_loader, model, epoch, tb_logger, criterion_c, args, "WikiHowTo")
        return

    vsum_params = []
    base_params = []
    for name, param in model.named_parameters():
        if "base" not in name:
            vsum_params.append(param)
        elif "mixed_5" in name and "base" in name:
            base_params.append(param)

    if args.optimizer == "adam":
        if args.finetune:
            optimizer = torch.optim.Adam(
                [
                    {"params": base_params, "lr": args.lrs},
                    {"params": vsum_params, "lr": args.lrv},
                ],
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), args.lrv, weight_decay=args.weight_decay
            )
    elif args.optimizer == "sgd":
        if args.finetune:
            optimizer = torch.optim.SGD(
                [
                    {"params": base_params, "lr": args.lrs},
                    {"params": vsum_params, "lr": args.lrv},
                ],
                momentum=args.momemtum,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.lrv,
                momentum=args.momemtum,
                weight_decay=args.weight_decay,
            )

    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer, args.warmup_steps, len(train_loader) * args.epochs
    # )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=1.0)
    checkpoint_dir = os.path.join(
        os.path.dirname(__file__), args.checkpoint_dir, args.log_name
    )

    if args.checkpoint_dir != "" and args.rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    # optionally resume from a checkpoint
    if args.resume:
        checkpoint_path = get_last_checkpoint(checkpoint_dir)
        if checkpoint_path:
            log("=> loading checkpoint '{}'".format(checkpoint_path), args)
            checkpoint = torch.load(checkpoint_path)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            log(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    checkpoint_path, checkpoint["epoch"]
                ),
                args,
            )
        else:
            log("=> no checkpoint found at '{}'".format(args.resume), args)

    if args.cudnn_benchmark:
        cudnn.benchmark = True
    total_batch_size = args.world_size * args.batch_size
    log(
        "Starting training loop for rank: {}, total batch size: {}".format(
            args.rank, total_batch_size
        ),
        args,
    )
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if (epoch + 1) % 2 == 0:
            evaluate(
                test_loader, model, epoch, tb_logger, criterion_c, args, "WikiHowTo"
            )
        # train for one epoch
        train(
            train_loader,
            model,
            criterion_c,
            optimizer,
            scheduler,
            epoch,
            train_dataset,
            tb_logger,
            args,
        )
        if args.rank == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                checkpoint_dir,
                epoch + 1,
            )


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    epoch,
    dataset,
    tb_logger,
    args,
):
    running_loss = 0.0
    s = time.time()
    for i_batch, sample_batch in enumerate(train_loader):
        s_step = time.time()
        batch_loss = TrainOneBatch(
            model, optimizer, scheduler, sample_batch, criterion, args, epoch
        )
        d_step = time.time() - s_step
        running_loss += batch_loss
        if (i_batch + 1) % args.log_freq == 0 and args.verbose and args.rank == 0:
            d = time.time() - s
            if args.finetune:
                current_lr = optimizer.param_groups[1]["lr"]
            else:
                current_lr = optimizer.param_groups[0]["lr"]
            log(
                "Epoch %d, Elapsed Time: %.3f, Epoch status: %.4f, Training loss: %.4f, Learning rate: %.6f"
                % (
                    epoch + 1,
                    d,
                    args.batch_size * args.world_size * float(i_batch) / len(dataset),
                    running_loss / args.log_freq,
                    current_lr,
                ),
                args,
            )
            # log training data into tensorboard
            if tb_logger is not None:
                logs = OrderedDict()
                logs["Train loss"] = running_loss / args.log_freq
                logs["Learning rate"] = current_lr
                # how many iterations we have trained
                iter_count = epoch * len(train_loader) + i_batch
                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, iter_count)
                tb_logger.flush()

            s = time.time()
            running_loss = 0.0


def TrainOneBatch(model, opt, scheduler, data, loss_fun, args, epoch):
    video = data["video"].float().cuda(args.gpu, non_blocking=args.pin_memory)
    label = data["label"].cuda(args.gpu, non_blocking=args.pin_memory).view(-1)
    label_scores = (
        data["label scores"].cuda(args.gpu, non_blocking=args.pin_memory).view(-1)
    )

    video = video / 255.0
    opt.zero_grad()
    with torch.set_grad_enabled(True):
        video_embd, score = model(video)
        if args.distributed:
            label = allgather(label, args)
            label_scores = allgather(label_scores, args)
            video_embd = allgather(video_embd, args)
            score = allgather(score, args)
        loss = loss_fun(score.view(-1), label_scores)
        # if args.rank == 0 and epoch > 5:
        #     # summary_frames = nn.functional.softmax(score.detach().cpu(), dim=1)[:, 1]
        #     summary_frames = nn.functional.softmax(score.view(-1).detach().cpu(), dim=0)
        #     summary_frames[summary_frames < threshold] = 0
        #     summary_frames[summary_frames > threshold] = 1
        # print("Gen {}, Labels {} :".format(summary_frames, label))
    # loss = loss / (int(args.num_frames / args.num_frames_per_segment))
    # loss.backward()
    # Since loss is a non-scalar, provide gradient as tensor of 1s
    gradient = torch.ones((loss.shape[0]), dtype=torch.long).cuda(
        args.gpu, non_blocking=args.pin_memory
    )
    loss.backward(gradient=gradient)
    loss = loss.mean()
    opt.step()
    scheduler.step()
    return loss.item()


def evaluate(test_loader, model, epoch, tb_logger, loss_fun, args, dataset_name):
    losses = AverageMeter()
    f_scores = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    model.eval()
    if args.rank == 0:
        log("Evaluating on {}".format(dataset_name), args)
        table = PrettyTable()
        table.title = "Eval result of epoch {}".format(epoch)
        table.field_names = ["F-score", "Precision", "Recall", "Loss"]
        table.float_format = "1.3"

    with torch.no_grad():
        for i_batch, data in enumerate(test_loader):
            label = data["label"].cuda().view(-1)
            video = data["video"].float().cuda()
            label_scores = data["label scores"].cuda().view(-1)
            video = video / 255.0
            video_embd, score = model(video)
            if args.distributed:
                label = allgather(label, args)
                label_scores = allgather(label_scores, args)
                video_embd = allgather(video_embd, args)
                score = allgather(score, args)

            if args.rank == 0:
                loss = loss_fun(score.view(-1), label_scores)
                # summary_frames = torch.argmax(score.data, 1)

                # score = nn.functional.log_softmax(score.view(-1).detach().cpu(), dim=0)
                # score = nn.functional.normalize(score.view(-1).detach().cpu(), dim=0)
                summary_ids = (
                    score.detach().cpu().view(-1).topk(int(0.50 * len(label)))[1]
                )
                summary = np.zeros(len(label))
                summary[summary_ids] = 1
                # threshold = 0.85 * summary_frames.max()
                # print("Thershold: ", threshold)
                # summary_frames[summary_frames < threshold] = 0
                # summary_frames[summary_frames > threshold] = 1

                # print(
                #     "Summary frames: ",
                #     summary,
                #     "Labels: ",
                #     label,
                #     "Scores: ",
                #     score.view(-1),
                #     "Label scores: ",
                #     label_scores,
                # )

                f_score, precision, recall = evaluate_summary(
                    summary, label.detach().cpu().numpy()
                )
                loss = loss.mean()
                losses.update(loss.item(), video_embd.shape[0])
                f_scores.update(f_score, video_embd.shape[0])
                precisions.update(precision, video_embd.shape[0])
                recalls.update(recall, video_embd.shape[0])

    loss = losses.avg
    f_score = f_scores.avg
    precision = precisions.avg
    recall = recalls.avg

    if args.rank == 0:
        log(
            "Epoch {} \t"
            "F-Score {} \t"
            "Precision {} \t"
            "Recall {} \t"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                epoch, f_score, precision, recall, loss=losses
            ),
            args,
        )
        table.add_row([f_score, precision, recall, loss])
        tqdm.write(str(table))

        if tb_logger is not None:
            # log training data into tensorboard
            logs = OrderedDict()
            logs["Val_IterLoss"] = losses.avg
            logs["F-Score"] = f_scores.avg
            logs["Precision"] = precisions.avg
            logs["Recall"] = recalls.avg

            # how many iterations we have validated
            for key, value in logs.items():
                tb_logger.log_scalar(value, key, epoch)

            tb_logger.flush()

    model.train()


def save_checkpoint(state, checkpoint_dir, epoch, n_ckpt=10):
    torch.save(
        state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch))
    )
    if epoch - n_ckpt >= 0:
        oldest_ckpt = os.path.join(
            checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch - n_ckpt)
        )
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)


def get_last_checkpoint(checkpoint_dir):
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, "epoch*.pth.tar"))
    if all_ckpt:
        all_ckpt = sorted(all_ckpt)
        return all_ckpt[-1]
    else:
        return ""


def log(output, args):
    print(output)
    with open(
        os.path.join(
            os.path.dirname(__file__), "vsum_ouptut_log", args.log_name + ".txt"
        ),
        "a",
    ) as f:
        f.write(output + "\n")


# gcn_params = []
# base_params = []
# for name, param in model.named_parameters():
#     if 'ga' in name or 'gcn' in name:
#         gcn_params.append(param)
#     else:
#         base_params.append(param)

# optimizer = optim.Adam([
#     {"params": base_params, "lr": args.learning_rate_lstm},
#     {"params": gcn_params, "lr":args.learning_rate_gcn},
# ], weight_decay=1e-6)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.4)


if __name__ == "__main__":
    main()
