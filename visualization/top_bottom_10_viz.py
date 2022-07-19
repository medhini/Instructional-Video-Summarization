import os
import argparse
import copy
import json
import csv
import sys


def generate_html(results_dir):
    files = os.listdir(results_dir)

    with open(os.path.join(results_dir, "results.json"), "r") as f:
        data = json.load(f)

    video_ids = {}
    for file in files:
        if file != "results.json":
            video_ids[file.split(".")[0]] = data[file.split(".")[0]]["score"]

    # Sort scores and visualize
    video_ids = dict(sorted(video_ids.items(), key=lambda x: x[1]))

    exp_name = results_dir.split("/")[-1]
    html_filename = "/home/medhini/video_summarization/task_video_sum/visualization/{}.html".format(
        exp_name
    )

    print("Writing to file {}".format(html_filename))

    with open(html_filename, "w") as f:
        # write meta
        f.write(
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset=\
                'UTF-8'>\n<script src=\"dump_display.js\"></script>\
                <title>Video Summarization</title>\n</head>\n<body>\n"
        )

        f.write(
            "<style>\nth {\nbackground: white;\n position: sticky;\n top: 0;\n \
                box-shadow: 0 2px 2px -1px rgba(0, 0, 0, 0.4);\n\
                z-index: 10;}\n</style>\n"
        )

        headline = "<h1> WikiHowTo Video Summaries</h1>"
        f.write(headline)

        table = "<font size='5'>\
                <table border='1', width=%d%%, style='table-layout: fixed, margin: 0px;'>\
                <thead>\n\
                    <th width='10%' style='text-align:center'>Video ID</th>\
                    <th width='40%' style='text-align:center'>Gen Summary</th>\
                    <th width='40%' style='text-align:center'>GT Summary</th>\
                    <th width='10%' style='text-align:center'>Score</th>\
                </thead>\
                <tbody>\n"

        f.write(table)

        count = 0

        for k, val in video_ids.items():
            gen_sum_path = os.path.join(results_dir, k + ".mp4")
            gt_sum_path = os.path.join("../datasets/how_to_summary_videos", k + ".mp4",)
            score = val

            if count == 0:
                td = '<tr><td colspan="4" style="text-align:center"> Bottom 11 </td></tr>\n'
                f.write(td)
            if count == 11:
                td = (
                    '<tr><td colspan="4" style="text-align:center"> Top 11 </td></tr>\n'
                )
                f.write(td)
            count += 1

            td = "<td>{}</td>".format(k)
            td += "<td width=20%% style='text-align:center'><video width='340' height='260' \
                    controls preload=\"metadata\">\n<source src='{}' type='video/mp4'>\n</video></td>".format(
                gen_sum_path
            )
            td += "<td width=20%% style='text-align:center'><video width='340' height='260' \
                    controls preload=\"metadata\">\n<source src='{}' type='video/mp4'>\n</video></td>".format(
                gt_sum_path
            )
            td += "<td>{}</td>".format(str(round(score, 2)))
            td = "<tr>{}</tr>\n".format(td)
            f.write(td)

        table_end = "</tbody></table></font>"
        f.write(table_end)

        table = "<font size='5'>\
                <table border='1', width=%d%%, style='table-layout: fixed, margin: 0px;'>\
                <thead>\n\
                    <th width='10%' style='text-align:center'>Avg F-Score</th>\
                    <th width='10%' style='text-align:center'>Avg Precision</th>\
                    <th width='10%' style='text-align:center'>Avg Recall</th>\
                </thead>\
                <tbody>\n"
        f.write(table)

        td = "<td>{}</td>".format(str(round(data["Avg F-Score"], 2)))
        td += "<td>{}</td>".format(str(round(data["Avg Precision"], 2)))
        td += "<td>{}</td>".format(str(round(data["Avg Recall"], 2)))
        td = "<tr>" + td + "</tr>\n"
        f.write(td)

        table_end = "</tbody></table></font>"
        f.write(table_end)

        f.write("</body>\n</html>\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:", sys.argv[0], "[relative results path]")
        sys.exit(0)

    generate_html(sys.argv[1])
