import argparse
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision.io as io
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import imageio
import copy

from types import SimpleNamespace


class MIL_NCE_Extract_Feats(Dataset):
    def __init__(self, args):
        self.v_path = args.vdata
        self.asr_path = args.asr_path
        self.window = args.window
        self.dataset = args.dataset

        self.asr_files = sorted(os.listdir(self.asr_path))

        if self.dataset == "wikihowto":
            all_vid_files = sorted(os.listdir(self.v_path))
            self.vid_files = []

            for file in all_vid_files:
                if file.split(".")[0] + ".en.txt" in self.asr_files:
                    self.vid_files.append(file)
        else:
            self.vid_files = sorted(os.listdir(self.v_path))
            # skip done, errors
            if os.path.exists("../datasets/pseudoGT_milnce_feats_8fps/video/embedding"):
                done = os.listdir(
                    "../datasets/pseudoGT_milnce_feats_8fps/video/embedding".format(
                        self.dataset
                    )
                )
                for f in done:
                    if f.split(".")[0] + ".mp4" in self.vid_files:
                        self.vid_files.remove(f.split(".")[0] + ".mp4")
                    if f.split(".")[0] + ".mkv" in self.vid_files:
                        self.vid_files.remove(f.split(".")[0] + ".mkv")
                    if f.split(".")[0] + ".webm" in self.vid_files:
                        self.vid_files.remove(f.split(".")[0] + ".webm")

        self.transform = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.vid_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.v_path, self.vid_files[idx])
        print("Starting video: ", self.vid_files[idx])

        assert os.path.exists(video_path), "No video found at {}".format(
            self.vid_files[idx]
        )

        video, _, meta = io.read_video(video_path, pts_unit="sec")

        # window_len = int(self.window * meta["video_fps"])
        window_len = 32

        # Pad video with extra frames to ensure its divisible by window_len
        extra_frames = window_len - (len(video) % window_len)
        video = torch.cat((video, video[-extra_frames:]), dim=0)

        n_segs = int(video.shape[0] / window_len)

        print("Number of video segments: ", n_segs)

        video = video.view(n_segs, 32, video.shape[1], video.shape[2], 3)
        video = video.permute(0, 1, 4, 2, 3)

        # Transform video segments
        video_segs = []
        for seg in video:
            # Resize and normalize to [0,1]
            video_segs.append(self.transform(seg) / 255.0)
        video_segs = torch.stack(video_segs)
        video_segs = video_segs.view(n_segs, 32, 3, 224, 224)

        return video_segs, self.vid_files[idx].split(".")[0]
