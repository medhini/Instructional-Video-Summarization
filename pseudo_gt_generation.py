import argparse
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.io as io
import json

parser = argparse.ArgumentParser(description="PyTorch ASR Video Segment MIL-NCE")

# Path related arguments
parser.add_argument(
    "--video_feats_dir",
    "-video_feats_dir",
    default="./datasets/pseudoGT_milnce_feats_8fps/video/embedding",
    type=str,
    help="Path to video features",
)
parser.add_argument(
    "--video_dir",
    "-video_dir",
    default="./datasets/pseudoGT_videos",
    type=str,
    help="Path to video dataset",
)
parser.add_argument(
    "--asr_feats_dir",
    "-asr_feats_dir",
    default="./datasets/pseudoGT_milnce_feats_8fps/asr",
    type=str,
    help="Path to asr features",
)
parser.add_argument(
    "--annt_dir",
    "-annt_dir",
    default="./datasets/pseudoGT_task_annts.json",
    type=str,
    help="Path to pseudoGT task annotations",
)
parser.add_argument(
    "--logdir", default="./logs", help="folder to output tensorboard logs"
)
parser.add_argument(
    "--logname",
    default="pseudoGT summary generation algorithm",
    help="name of the experiment for checkpoints and logs",
)
parser.add_argument(
    "-out_dir",
    "--out_dir",
    default="./datasets/pseudogt_gen_summaries",
    type=str,
    help="folder for result videos",
)
parser.add_argument(
    "--log_videos", dest="log_videos", action="store_true", help="Logs videos"
)

# Model related arguments
parser.add_argument(
    "-vst",
    "--v_similarity_threshold",
    default=0.9,
    type=float,
    help="vsr similarity threshold",
)
parser.add_argument(
    "-th",
    "--threshold",
    default=0.85,
    type=float,
    help="cut off threshold",
)

def cluster_vid(args):
    all_video_summaries = {}
    # cluster into segments
    # for each task
    # for all videos in a given task;
    # for each segment in video, identify similarity between segment and segments in other video; keep score
    # if asr is available, compare ASR feats and video feats, assign scores
    # detect salient segments; retain only those
    # check overlap with ground truth annotations (precision, recall)
    # Make output directory

    os.makedirs(args.out_dir, exist_ok=True)

    tasks = {}
    with open(args.annt_dir) as f:
        videos = json.load(f)
        for video_id, task_name in videos.items():
            if video_id + ".pth" in os.listdir(args.video_feats_dir):
                if task_name not in tasks.keys():
                    tasks[task_name] = [video_id]
                else:
                    tasks[task_name].append(video_id)

    precisions = []
    recalls = []
    f_scores = []
    step_acc = []

    for task, videos in tasks.items():
        print("Starting task: ", task)
        video_feats = []  # segment feats stored video wise
        all_feats = []  # all segment feats
        video_segments = []  # segments for all videos
        for video_name in videos:
            video_seg_feats = []
            segment_count = 1
            segments = [(0, 0)]
            vid_feats = torch.load(
                os.path.join(args.video_feats_dir, video_name + ".pth")
            ).cuda()
            # vid_feats = F.normalize(vid_feats, dim=1)

            # Find max vid feature similarity
            vid_feat_sim = torch.matmul(vid_feats, vid_feats.t()).mean(axis=1)
            max_sim = vid_feat_sim.max()

            # With time as axis cluster the segments
            avg_feat = vid_feats[0]
            start_feat = vid_feats[0]
            moving_avg_count = 1
            for i in range(1, len(vid_feats)):
                sim = torch.matmul(vid_feats[i], start_feat.t())
                if sim > args.v_similarity_threshold * max_sim:
                    avg_feat += vid_feats[i]
                    start_feat = vid_feats[i]
                    moving_avg_count += 1
                else:
                    segment_count += 1
                    segments[len(segments) - 1] = (
                        segments[len(segments) - 1][0],
                        i - 1,
                    )
                    segments.append((i, i))
                    video_seg_feats.append(avg_feat / moving_avg_count)
                    all_feats.append(avg_feat)
                    avg_feat = vid_feats[i]
                    start_feat = vid_feats[i]
                    moving_avg_count = 1

            if moving_avg_count > 1:
                segments[len(segments) - 1] = (
                    segments[len(segments) - 1][0],
                    i,
                )
                video_seg_feats.append(avg_feat / moving_avg_count)
                all_feats.append(avg_feat)

            assert segment_count == len(segments)
            video_feats.append(torch.stack(video_seg_feats))
            video_segments.append(segments)

        all_feats = torch.stack(all_feats)
        # all_feats = F.normalize(all_feats, dim=1)

        for idx, video_seg_feats in enumerate(video_feats):
            video_name = videos[idx]
            print("Starting video :", video_name)
            segments = video_segments[idx]
            print("Number of segments: ", len(segments))
            seg_scores = []
            if len(video_seg_feats) == 0:
                continue

            # video_seg_feats = F.normalize(video_seg_feats, dim=1)

            asr_similarity_matrix = None
            # Compute ASR Similarity
            if os.path.exists(os.path.join(args.asr_feats_dir, video_name + ".pth")):
                asr_feats = torch.load(
                    os.path.join(args.asr_feats_dir, video_name + ".pth")
                ).cuda()
                print("ASR shape: ", asr_feats.shape)
                # asr_feats = F.normalize(asr_feats, dim=1)
                asr_similarity_matrix = (
                    torch.matmul(video_seg_feats, asr_feats.t()).detach().cpu()
                )
                asr_similarity_matrix = asr_similarity_matrix.mean(axis=1)

            v_similarity_matrix = (
                torch.matmul(video_seg_feats, all_feats.t()).detach().cpu()
            )
            v_similarity_matrix = v_similarity_matrix.mean(axis=1)

            # Combine both the similarity matrices
            if asr_similarity_matrix != None:
                scores = (v_similarity_matrix + asr_similarity_matrix) / 2
            else:
                scores = v_similarity_matrix

            scores = F.normalize(scores, dim=0)

            if os.path.exists(os.path.join(args.video_dir, video_name + ".mp4")):
                video_path = os.path.join(args.video_dir, video_name + ".mp4")
            elif os.path.exists(os.path.join(args.video_dir, video_name + ".webm")):
                video_path = os.path.join(args.video_dir, video_name + ".webm")
            elif os.path.exists(os.path.join(args.video_dir, video_name + ".mkv")):
                video_path = os.path.join(args.video_dir, video_name + ".mkv")
            else:
                print("Error: ", video_name)

            # reader = io.VideoReader(video_path)
            # fps = reader.get_meta_data()['video']["fps"]
            # duration = reader.get_meta_data()["duration"]
            # print("Frame rate: {} Duration: {}".format(fps, duration))

            if idx % 1000 == 0 and args.log_videos:
                try:
                    video, _, meta = io.read_video(
                        video_path,
                        pts_unit="sec",
                    )
                except:
                    video = None
                    print("Skipped logging.")

            # Save summary at 8 fps segment-wise; Each segment is 4 seconds (32 frames @ 8 fps);
            n_video_segments = segments[-1][1] + 1
            machine_summary = np.zeros(n_video_segments)
            machine_summary_scores = np.zeros(n_video_segments)
            print("Machine summary shape: ", machine_summary.shape)

            threshold = args.threshold * scores.max()
            print("Threshold: ", threshold)
            summary_video = []
            for itr, score in enumerate(scores):
                segment = segments[itr]
                if segment[1] >= segment[0]:
                    machine_summary_scores[segment[0] : (segment[1] + 1)] = score
                    if score >= threshold:
                        machine_summary[segment[0] : (segment[1] + 1)] = 1
                        if idx % 1000 == 0 and args.log_videos and video != None:
                            summary_video.append(
                                video[segment[0] * 32 : (segment[1] + 1) * 32]
                            )

            if idx % 1000 == 0 and args.log_videos and video != None:
                summary_video = torch.cat((summary_video), dim=0)
                print("Summary video shape: ", summary_video.shape)
                io.write_video(
                    os.path.join(
                        args.out_dir,
                        "{}_summary.mp4".format(video_name),
                    ),
                    summary_video,
                    meta["video_fps"],
                )
            all_video_summaries[video_name] = {}
            all_video_summaries[video_name][
                "machine_summary"
            ] = machine_summary.tolist()
            all_video_summaries[video_name][
                "machine_summary_scores"
            ] = machine_summary_scores.tolist()

    # Save results
    print("Total number of videos: ", len(all_video_summaries.keys()))
    with open(
        os.path.join(args.out_dir, "pseudoGT_summ_annts.json"),
        "w",
    ) as f:
        json.dump(all_video_summaries, f)
    print(
        "Generated summary JSON written to {}".format(
            os.path.join(args.out_dir, "pseudoGT_summ_annts.json")
        )
    )

    return


if __name__ == "__main__":
    args = parser.parse_args()
    cluster_vid(args)
