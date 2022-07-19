from cv2 import split
import torch
from torch.utils.data import Dataset
import torchvision.io as io
import torchvision.transforms as transforms
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import time
import re
import json
import torch
import imageio


class VSum_DataLoader(Dataset):
    """VSum_DataLoader Video loader."""

    def __init__(
        self,
        annt_path,
        video_root="",
        caption_root="",
        min_time=4.0,
        fps=8,
        num_frames=832,
        num_frames_per_segment=32,
        size=224,
        crop_only=False,
        center_crop=True,
        benchmark=False,
        token_to_word_path="data/dict.npy",
        max_words=20,
        num_candidates=1,
        random_left_right_flip=False,
        video_only=False,
        dataset="pseudo",
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        with open(annt_path) as f:
            self.annts = json.load(f)
        # self.csv = pd.read_csv(os.path.join(os.path.dirname(__file__), csv))
        self.video_root = video_root
        self.videos = [
            x
            for x in sorted(os.listdir(self.video_root))
            if x.split(".")[0] in self.annts.keys() and not x.startswith(".")
        ]
        self.caption_root = caption_root
        self.min_time = min_time
        self.size = size
        self.num_frames = num_frames
        self.num_frames_per_segment = num_frames_per_segment
        self.fps = fps
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.benchmark = benchmark
        self.max_words = max_words
        self.video_only = video_only
        token_to_word = np.load(
            os.path.join(os.path.dirname(__file__), token_to_word_path)
        )
        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
        self.num_candidates = num_candidates
        self.random_flip = random_left_right_flip
        self.dataset = dataset

        pos = 0
        neg = 0
        for v in self.videos:
            if self.dataset == "wikihow":
                labels = self.annts[v.split(".")[0]]
            else:
                labels = self.annts[v.split(".")[0]]["machine_summary"]
            pos += np.count_nonzero(np.asarray(labels))
            neg += len(labels) - np.count_nonzero(np.asarray(labels))

        print("Pos neg: ", pos, neg)
        self.ce_weight = torch.tensor(
            [(pos + neg) / neg, (pos + neg) / pos], dtype=torch.float32
        )

    def __len__(self):
        return len(self.videos)

    def _get_video(self, video_path, start_seek, time):
        # cmd = ffmpeg.input(video_path, ss=start_seek, t=time).filter(
        #         "fps", fps=self.fps
        #     )
        # if self.split == "train":
        #     if self.center_crop:
        #         aw, ah = 0.5, 0.5
        #     else:
        #         aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        #     if self.crop_only:
        #         cmd = cmd.crop(
        #             "(iw - {})*{}".format(self.size, aw),
        #             "(ih - {})*{}".format(self.size, ah),
        #             str(self.size),
        #             str(self.size),
        #         )
        #     else:
        #         cmd = cmd.crop(
        #             "(iw - min(iw,ih))*{}".format(aw),
        #             "(ih - min(iw,ih))*{}".format(ah),
        #             "min(iw,ih)",
        #             "min(iw,ih)",
        #         ).filter("scale", self.size, self.size)

        #     if (
        #         self.split == "train"
        #         and self.random_flip
        #         and random.uniform(0, 1) > 0.5
        #     ):
        #         cmd = cmd.hflip()
        # else:
        #     cmd = cmd.filter("scale", self.size, self.size)

        # out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
        #     capture_stdout=True, quiet=True
        # )

        # video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        # video = torch.from_numpy(video.copy())
        # video = video.view(-1, self.size, self.size, 3)

        # # [B, H, W, C] -> [C, B, H, W]
        # video = video.permute(3, 0, 1, 2)
        frames, _, _ = io.read_video(
            video_path, start_pts=start_seek, end_pts=start_seek + time, pts_unit="sec",
        )
        # [B, H, W, C] -> [B, C, H, W]
        frames = frames.permute(0, 3, 1, 2)
        video = []
        for frame in frames:
            # Transform video segments
            video.append(transforms.Resize((self.size, self.size))(frame))
        video = torch.stack(video)
        video = video.view(-1, 3, self.size, self.size)

        # [B, C, H, W] -> [C, B, H, W]
        video = video.permute(1, 0, 2, 3)

        if video.shape[1] < self.num_frames:
            zeros = torch.zeros(
                (3, self.num_frames - video.shape[1], self.size, self.size),
                dtype=torch.uint8,
            )
            video = torch.cat((video, zeros), axis=1)
        return video[:, : self.num_frames]

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [
            self.word_to_token[word] for word in words if word in self.word_to_token
        ]
        if words:
            we = self._zero_pad_tensor_token(torch.LongTensor(words), self.max_words)
            return we
        else:
            return torch.zeros(self.max_words, dtype=torch.long)

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = torch.zeros(size - len(tensor)).long()
            return torch.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    def _find_nearest_candidates(self, caption, ind):
        start, end = ind, ind
        diff = caption["end"][end] - caption["start"][start]
        n_candidate = 1
        while n_candidate < self.num_candidates:
            if start == 0:
                return 0
            elif end == len(caption) - 1:
                return start - (self.num_candidates - n_candidate)
            elif (
                caption["end"][end] - caption["start"][start - 1]
                < caption["end"][end + 1] - caption["start"][start]
            ):
                start -= 1
            else:
                end += 1
            n_candidate += 1
        return start

    def _get_text(self, caption):
        cap = pd.read_csv(caption)
        ind = random.randint(0, len(cap) - 1)
        if self.num_candidates == 1:
            words = self.words_to_ids(cap["text"].values[ind])
        else:
            words = torch.zeros(self.num_candidates, self.max_words, dtype=torch.long)
            cap_start = self._find_nearest_candidates(cap, ind)
            for i in range(self.num_candidates):
                words[i] = self.words_to_ids(
                    cap["text"].values[max(0, min(len(cap["text"]) - 1, cap_start + i))]
                )
        start, end = cap["start"].values[ind], cap["end"].values[ind]
        # TODO: May need to be improved for edge cases.
        if end - start < self.min_time:
            diff = self.min_time - end + start
            start = max(0, start - diff / 2)
            end = start + self.min_time
        return words, int(start), int(end)

    def __getitem__(self, idx):
        video_file = self.videos[idx]
        video_path = os.path.join(self.video_root, video_file)
        video_id = video_file.split(".")[0]

        if self.dataset == "wikihow":
            print("Video: ", video_file)
            # original labels are per frame for WikiHowTo
            orig_labels = torch.FloatTensor(self.annts[video_id])
            num_frames_in_video = len(orig_labels)

            # get original fps of video; retrieve video at original fps
            reader = io.VideoReader(os.path.join(video_path), "video")
            fps = reader.get_metadata()["video"]["fps"][0]
            self.fps = fps
            self.num_sec = self.num_frames / float(self.fps)
        else:
            labels = torch.LongTensor(self.annts[video_id]["machine_summary"])
            label_scores = (
                torch.FloatTensor(self.annts[video_id]["machine_summary_scores"]) / 0.1
            )
            num_frames_in_video = (
                len(labels) * self.num_frames_per_segment
            )  # labels are for each segment and each segment has 32 frames

        end = num_frames_in_video / self.fps  # num_frames/fps_of_video
        start_seek = random.randint(0, int(max(0, end - self.num_sec)))
        # start_seek = 0
        time = self.num_sec + 0.1
        video = self._get_video(video_path, start_seek=start_seek, time=time)

        n_segments = int(self.num_frames / self.num_frames_per_segment)
        start_frame = int(
            (start_seek * self.fps) / self.num_frames_per_segment
        )  # Get segment number from start second; (time * fps) / n_frames_per_seg

        if self.dataset == "wikihow":
            # size of labels should match n_segments in sampled video
            # for wikihowto labels are stored for each frame @ orig fps
            # first convert frame wise labels to segment wise labels
            # since there are 32 frames per segment, set labels to be max
            n_segments_whole_video = int(
                num_frames_in_video / self.num_frames_per_segment
            )
            label_scores = torch.zeros(n_segments_whole_video)
            labels = torch.zeros(n_segments_whole_video, dtype=torch.long)
            for x in range(n_segments_whole_video):
                count_1 = np.count_nonzero(
                    orig_labels[
                        x
                        * self.num_frames_per_segment : (x + 1)
                        * self.num_frames_per_segment
                    ]
                )
                count_0 = self.num_frames_per_segment - count_1
                if count_1 > count_0:
                    labels[x] = 1
                else:
                    labels[x] = 0
                label_scores[x] = orig_labels[
                    x
                    * self.num_frames_per_segment : (x + 1)
                    * self.num_frames_per_segment
                ].mean()

        # size of labels should match n_segments in sampled video
        labels = labels[start_frame : start_frame + n_segments]
        label_scores = label_scores[start_frame : start_frame + n_segments]

        # append 0s to labels
        if labels.shape[0] < n_segments:
            labels = torch.cat(
                (labels, torch.zeros((n_segments - len(labels)), dtype=torch.long))
            )
            label_scores = torch.cat(
                (label_scores, torch.zeros((n_segments - len(label_scores))))
            )

        labels = torch.LongTensor(labels)
        assert labels.shape[0] == n_segments
        assert label_scores.shape[0] == n_segments

        if self.video_only:
            return {"video": video, "label": labels, "label scores": label_scores}
        else:
            text, start, end = self._get_text(
                os.path.join(self.caption_root, video_id + ".csv")
            )
            return {"video": video, "text": text, "label": labels}
