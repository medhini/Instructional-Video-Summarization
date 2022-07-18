import os
import json
import subprocess

WIKI_DIR = os.path.dirname(os.path.realpath(__file__))
files = os.listdir(WIKI_DIR + "/how_to_steps")

for file in files:
    with open(os.path.join(WIKI_DIR, "how_to_steps", file), "r") as f:
        if not os.path.exists(
            os.path.join(WIKI_DIR, "wikihow_videos/" + file.split(".")[0])
        ):
            print(os.path.join(WIKI_DIR, "wikihow_videos/" + file.split(".")[0]))
            data = json.load(f)
            os.makedirs(os.path.join(WIKI_DIR, "wikihow_videos/" + file.split(".")[0]))
            subprocess.call(
                "yt-dlp  -o '{}/wikihow_videos/{}/main_video.%(ext)s' --write-subs --write-auto-subs --geo-verification-proxy --geo-bypass '{}'".format(
                    WIKI_DIR,
                    file.split(".")[0],
                    data["video_url"],
                ),
                shell=True,
            )
            for key, val in data.items():
                if key != "video_url":
                    if data[key]["img"] != None:
                        subprocess.call(
                            "curl \"{}\" -o '{}/wikihow_videos/{}/{}.jpg'".format(
                                data[key]["img"], WIKI_DIR, file.split(".")[0], key
                            ),
                            shell=True,
                        )
                    if data[key]["vid"] != None:
                        try:
                            print(data[key]["vid"].encode("utf-8"))
                            subprocess.call(
                                "youtube-dl -f bestvideo+bestaudio/best -o '{}/wikihow_videos/{}/{}.mp4' -ci 'https://wikihow.com/video{}'".format(
                                    WIKI_DIR, file.split(".")[0], key, data[key]["vid"]
                                ),
                                shell=True,
                            )
                        except:
                            print(data[key]["vid"])
                            print("task name: {} ".format(file.split(".")[0]))
                            continue
                    if data[key]["img"] == None and data[key]["vid"] == None:
                        print(
                            "Missing task name: {} step: {}".format(
                                file.split(".")[0], key
                            )
                        )
