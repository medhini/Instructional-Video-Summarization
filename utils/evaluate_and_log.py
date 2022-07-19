import os
from utils import AverageMeter, video_from_summary
from collections import OrderedDict
from prettytable import PrettyTable
import json
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("../")
from visualization import generate_html


def evaluate_summary(machine_summary, gt_summary):
    """Compare machine summary with user summary (keyshot-based).
    Args:
    --------------------------------
    machine_summary and gt_summary should be binary vectors of ndarray type.
    """
    machine_summary = np.asarray(machine_summary, dtype=np.float32)
    gt_summary = np.asarray(gt_summary, dtype=np.float32)
    n_frames = gt_summary.shape[0]

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])

    if gt_summary.sum() == 0:
        return 1.0, 1.0, 1.0

    overlap_duration = (machine_summary * gt_summary).sum()
    precision = overlap_duration / (machine_summary.sum() + 1e-8)
    recall = overlap_duration / (gt_summary.sum() + 1e-8)
    if precision == 0 and recall == 0:
        f_score = 0.0
    else:
        f_score = (2 * precision * recall) / (precision + recall)

    return f_score, precision, recall


def log_scores(
    tb_logger,
    annt_dir,
    video_dir,
    video_frames_dir,
    out_dir,
    all_video_summaries,
    epoch=0,
    log_videos=False,
):
    # Compute scores
    f_scores = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    # Get video ids
    with open(annt_dir) as f:
        annts = json.load(f)

    # Write to table
    table = PrettyTable()
    table.title = "Eval result of epoch {}".format(epoch)
    table.field_names = ["ID", "F-score", "Precision", "Recall"]
    table.float_format = "1.3"

    remove_keys = []
    for video_id in all_video_summaries.keys():
        if video_id in annts.keys():
            # gt_summary = json.load(af)["annts"]
            gt_summary = annts[video_id]

            print(
                "GT summary shape, machine summary shape: ",
                len(gt_summary),
                len(all_video_summaries[video_id]["machine_summary"]),
            )

            # Trim machine summary
            all_video_summaries[video_id]["machine_summary"] = all_video_summaries[
                video_id
            ]["machine_summary"][: len(gt_summary)]

            f_score, prec, recall = evaluate_summary(
                all_video_summaries[video_id]["machine_summary"], gt_summary,
            )
            all_video_summaries[video_id]["score"] = f_score
            # table.add_row([video_id, f_score, prec, recall])
            f_scores.update(f_score)
            precisions.update(prec)
            recalls.update(recall)
            print("Fscore: ", f_score)
        else:
            remove_keys.append(video_id)

    # Remove keys for which annts are missing
    for key in remove_keys:
        all_video_summaries.pop(key)

    # Log to tensorboard
    logs = OrderedDict()
    logs["F-Score"] = f_scores.avg
    logs["Precision"] = precisions.avg
    logs["Recall"] = recalls.avg

    # Write logger
    for key, value in logs.items():
        tb_logger.log_scalar(value, key, epoch)
    tb_logger.flush()

    # Write table
    table.add_row(["mean", f_scores.avg, precisions.avg, recalls.avg])
    tqdm.write(str(table))

    # Sort scores and visualize
    sorted_ids = sorted(
        all_video_summaries, key=lambda x: (all_video_summaries[x]["score"])
    )

    # Top 10 and bottom 10 scoring videos
    top_bottom_10 = sorted_ids[-10:]
    top_bottom_10.extend(sorted_ids[:10])

    print("top bottom 10:", top_bottom_10)
    if video_frames_dir is not None and log_videos:
        for idx in top_bottom_10:
            video_from_summary(
                idx,
                video_frames_dir,
                video_dir,
                out_dir,
                all_video_summaries[idx]["machine_summary"],
            )
        # Write html file
        relative_results_path = os.path.join(
            "..", out_dir.split("/")[-2], out_dir.split("/")[-1]
        )
        generate_html(relative_results_path)

    all_video_summaries["Top Bottom 10"] = top_bottom_10

    # Add scores to dict
    all_video_summaries["Avg F-Score"] = f_scores.avg
    all_video_summaries["Avg Precision"] = precisions.avg
    all_video_summaries["Avg Recall"] = recalls.avg

    # Save results
    with open(os.path.join(out_dir, "results.json"), "w",) as f:
        json.dump(all_video_summaries, f)
    print(
        "Generated summary JSON written to {}".format(
            os.path.join(out_dir, "results.json")
        )
    )

    return

