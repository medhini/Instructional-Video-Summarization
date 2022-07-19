import subprocess
import numpy as np
import math
import torch
import os
import imageio
import math
import builtins
import datetime

import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def save_videos(
    output_video_folder, outfile, fps,
):
    try:
        subprocess.call(
            'ffmpeg -r {} -start_number "1" -i "{}" -c:v "libx264" -pix_fmt "yuv420p" -y "{}"'.format(
                fps, output_video_folder + "/%04d.png", outfile
            ),
            shell=True,
        )
        print("Written ", outfile)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            'command "{}" return with error (code {}): {}'.format(
                e.cmd, e.returncode, e.output
            )
        )

    # remove video file
    subprocess.call(["rm", "-r", output_video_folder])
    return


def video_from_summary(video_id, video_frames_dir, video_dir, out_dir, summary):
    reader = imageio.get_reader(os.path.join(video_dir, video_id + ".mp4"))
    fps = reader.get_meta_data()["fps"]

    # Save frames and create video
    os.makedirs(os.path.join(out_dir, video_id), exist_ok=True)

    # Fix issue with '(' in name
    if "(" in video_id:
        video_name = "'" + video_id + "'"
    else:
        video_name = video_id

    count = 0
    for idx, score in enumerate(summary):
        if score == 1:
            subprocess.call(
                "cp {} {}".format(
                    os.path.join(
                        video_frames_dir, video_name, "{:d}.png".format(idx + 1),
                    ),
                    os.path.join(out_dir, video_name, "{:d}.png".format(count + 1)),
                ),
                shell=True,
            )
            count += 1
    outfile = os.path.join(out_dir, video_name + ".mp4")

    try:
        subprocess.call(
            "ffmpeg -r {} -start_number 1 -i {} -c:v libx264 -crf 23 -pix_fmt yuv420p -y {}".format(
                str(fps), os.path.join(out_dir, video_name) + "/%d.png", outfile,
            ),
            shell=True,
        )
        print("Written ", outfile)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            'command "{}" return with error (code {}): {}'.format(
                e.cmd, e.returncode, e.output
            )
        )

    # Remove frames folder
    subprocess.call(["rm", "-r", os.path.join(out_dir, video_id)])

    return


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print
