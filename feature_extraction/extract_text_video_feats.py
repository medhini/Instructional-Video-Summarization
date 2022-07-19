import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import math

from s3dg import S3D
from data import MIL_NCE_Extract_Feats

parser = argparse.ArgumentParser(description="MIL-NCE Feature Extraction")
parser.add_argument(
    "--window", "-w", default=1, type=int, help="Size of temporal window in seconds"
)
parser.add_argument(
    "--batch_size",
    "-bs",
    default=1,
    type=int,
    metavar="N",
    help="mini-batch size (default: 32)",
)
parser.add_argument(
    "--vdata",
    "-vdata",
    default="../datasets/pseudoGT_videos",
    type=str,
    help="Path to video dataset",
)

parser.add_argument(
    "--asr_path",
    "-asr_dir",
    default="../datasets/pseudoGT_asr",
    type=str,
    help="Path to asr",
)

parser.add_argument(
    "-rf",
    "--results_folder",
    default="../datasets/pseudoGT_milnce_feats_8fps",
    type=str,
    help="folder for result videos",
)

parser.add_argument(
    "--workers",
    "-j",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 8)",
)
parser.add_argument(
    "--dataset",
    "-dataset",
    default="crosstask",
    type=str,
    help="Dataset name",
)

def to_cuda(item):
    if isinstance(item[0], list):
        return [[x.cuda() for x in y] for y in item]
    elif isinstance(item, list):
        return [x.cuda() for x in item]
    return item.cuda()


def main(args):
    # create dataset
    dataset = MIL_NCE_Extract_Feats(args)

    # create training loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    # Instantiate the model
    net = S3D("../pretrained_weights/s3d_dict.npy", 512)

    # Load the model weights
    net.load_state_dict(torch.load("../pretrained_weights/s3d_howto100m.pth"))
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    # Evaluation mode
    net = net.eval()

    # Create results folders
    os.makedirs(os.path.join(args.results_folder, "video", "embedding"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "video", "mixed_5c"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "asr"), exist_ok=True)

    with torch.no_grad():
        for itr, batch_data in enumerate(dataloader):
            (video_segs, video_name) = batch_data

            # Compute MIL-NCE feats for video segments

            # [1, B, T, C, H, W] -> [B, C, T, H, W]
            video_segs = video_segs.squeeze(0).permute(0, 2, 1, 3, 4)
            batch_size = video_segs.shape[0]
            num_gpus = torch.cuda.device_count()
            gpu_bs = 30
            mini_batch_size = gpu_bs * num_gpus

            vid_feats = []
            vid_mixed_5c = []

            for j in range(math.ceil(batch_size / mini_batch_size)):
                this_batch = video_segs[j * mini_batch_size : (j + 1) * mini_batch_size]
                this_batch = to_cuda(this_batch)
                vid_feats.extend(net(this_batch)["video_embedding"].detach().cpu())
                vid_mixed_5c.extend(net(this_batch)["mixed_5c"].detach().cpu())
            vid_feats = torch.stack(vid_feats)
            vid_mixed_5c = torch.stack(vid_mixed_5c)
            print("Size of vid feats: ", vid_feats.shape)

            # Store MIL-NCE feats for each video
            torch.save(
                vid_feats,
                os.path.join(
                    args.results_folder, "video", "embedding", video_name[0] + ".pth"
                ),
            )
            torch.save(
                vid_mixed_5c,
                os.path.join(
                    args.results_folder, "video", "mixed_5c", video_name[0] + ".pth"
                ),
            )

            print("Saved feats for: ", video_name[0])

    asr_files = os.listdir(args.asr_path)
    with torch.no_grad():
        for asr_file in asr_files:
            video_name = asr_file.split(".")[0]
            asr_file = os.path.join(args.asr_path, asr_file)
            with open(asr_file, "r") as f:
                asr_sentences = f.read().splitlines()

                # Text inference
                asr_feats = net.text_module(asr_sentences)["text_embedding"].detach()

                print("Size of asr feats: ", asr_feats.shape)

                torch.save(
                    asr_feats,
                    os.path.join(args.results_folder, "asr", video_name + ".pth"),
                )
                print("Saved asr feats for: ", video_name)


if __name__ == "__main__":
    args = parser.parse_args()
    assert os.path.exists(args.vdata), "No videos found at {}".format(args.vdata)
    main(args)
