import argparse
import os
import sys
import json
from xxlimited import new
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.io as io
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import glob

import s3dg

sys.path.append("../")
from utils import log_scores, Logger

parser = argparse.ArgumentParser(description="PyTorch ASR Video Segment MIL-NCE")

# Path related arguments
parser.add_argument(
    "--video_feats_dir",
    "-video_feats_dir",
    default="./datasets/wikihowto_milnce_feats_32/video/embedding",
    type=str,
    help="Path to video dataset",
)
parser.add_argument(
    "--video_dir",
    "-video_dir",
    default="./datasets/wikihowto_val",
    type=str,
    help="Path to video dataset",
)
parser.add_argument(
    "--video_frames_dir",
    "-video_frames_dir",
    default="./wikihow_frames",
    type=str,
    help="Path to video frames",
)
parser.add_argument(
    "--annt_dir",
    "-annt_dir",
    default="./datasets/wikihowto_annt.json",
    type=str,
    help="Path to video dataset",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="./exp_model_pretrainmilnce_trans_bs_1_lr_0.001_nframes_832_nfps_32/epoch0005.pth.tar",
    help="checkpoint model folder",
)
parser.add_argument(
    "--log_root", type=str, default="vsum_tboard_log", help="log dir root"
)
parser.add_argument(
    "--log_name", default="exp", help="name of the experiment for checkpoints and logs",
)
parser.add_argument(
    "--log_videos",
    dest="log_videos",
    action="store_true",
    help="Log top 10 and bottom 10 result videos",
)
parser.add_argument(
    "-out_dir",
    "--out_dir",
    default="./gen_summaries_milnce_wikihowto",
    type=str,
    help="folder for result videos",
)
parser.add_argument("--word2vec_path", type=str, default="data/word2vec.pth", help="")
parser.add_argument(
    "--pretrain_cnn_path",
    type=str,
    default="./pretrained_weights/s3d_howto100m.pth",
    help="",
)


def get_last_checkpoint(checkpoint_dir):
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, "epoch*.pth.tar"))
    if all_ckpt:
        all_ckpt = sorted(all_ckpt)
        return all_ckpt[-1]
    else:
        return ""


def rename_dict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def test(args):
    video_files = sorted(os.listdir(args.video_dir))

    # get logname from checkpoint dir
    if args.checkpoint_dir[-3:] == "tar":
        args.log_name = args.checkpoint_dir.split("/")[-2] + "_eval"
        checkpoint_path = args.checkpoint_dir
    else:
        args.log_name = args.checkpoint_dir.split("/")[-1] + "_eval"
        checkpoint_path = get_last_checkpoint(args.checkpoint_dir)

    # make out_dir
    args.out_dir = os.path.join(args.out_dir, args.log_name)
    os.makedirs(args.out_dir, exist_ok=True)

    # start a logger
    tb_logdir = os.path.join(args.log_root, args.log_name)
    os.makedirs(tb_logdir, exist_ok=True)
    tb_logger = Logger(tb_logdir)

    all_video_summaries = {}

    # create model
    model = s3dg.VSum(space_to_depth=False, word2vec_path=args.word2vec_path).cuda()
    model = model.eval()
    print("Created model")

    # load model checkpoint
    if checkpoint_path:
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        state_dict = rename_dict(checkpoint["state_dict"])
        model.load_state_dict(state_dict)
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_path, checkpoint["epoch"]
            )
        )

    with torch.no_grad():
        for itr, video_file in enumerate(video_files):
            video_id = video_file.split(".")[0]
            video_path = os.path.join(args.video_dir, video_file)

            print("Starting video: ", video_file)

            frames, _, meta = io.read_video(video_path)

            window_len = 32

            # Pad video with extra frames to ensure its divisible by window_len
            extra_frames = window_len - (len(frames) % window_len)
            frames = torch.cat((frames, frames[-extra_frames:]), dim=0)

            # [B, H, W, C] -> [B, C, H, W]
            frames = frames.permute(0, 3, 1, 2)

            n_segs = int(frames.shape[0] / window_len)

            print("Number of video segments: ", n_segs)
            video = []
            for frame in frames:
                # Transform video segments
                video.append(transforms.Resize((224, 224))(frame))
            video = torch.stack(video)

            # Transform video segments
            video = video / 255.0
            # [B, C, H, W] -> [B, T, C, H, W]
            video = video.view(n_segs, 32, 3, 224, 224)
            # [B, T, C, H, W] -> [B, C, T, H, W]
            video = video.permute(0, 2, 1, 3, 4)

            scores = []

            for seg in video:
                batch = seg.unsqueeze(0).cuda()
                _, score = model(batch)
                scores.append(score.view(-1))

            scores = torch.stack(scores)
            summary_frames = nn.functional.softmax(scores, dim=1)[:, 1]
            summary_frames[summary_frames > 0.5] = 1
            summary_frames = np.repeat(summary_frames.detach().cpu().numpy(), 32)
            print("Shape of summary frames:", summary_frames.shape)

            # Todo: Compute loss here

            # similarity_matrix = F.normalize(similarity_matrix.mean(axis=1), dim=0)
            # max_sim = float(similarity_matrix.max())

            # summary_frames = np.asarray(similarity_matrix)
            # summary_frames[summary_frames > max_sim * args.similarity_threshold] = 1.0
            # summary_frames[summary_frames != 1.0] = 0.0
            #

            video_id = video_id.split(".")[0]
            all_video_summaries[video_id] = {}
            all_video_summaries[video_id]["machine_summary"] = summary_frames.tolist()

    # Calculate scores and log videos
    log_scores(
        tb_logger,
        args.annt_dir,
        "/home/medhini/video_summarization/task_video_sum/datasets/how_to_summary_videos",
        args.video_frames_dir,
        args.out_dir,
        all_video_summaries,
        log_videos=args.log_videos,
    )

    return


if __name__ == "__main__":
    args = parser.parse_args()
    assert os.path.exists(args.video_dir), "No videos found at {}".format(
        args.video_dir
    )
    test(args)

