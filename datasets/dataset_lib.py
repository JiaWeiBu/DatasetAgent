# python version : 3.12.6

# Read from this folder
# read each item in the folder
# if the folder is a file then read the folder else do nothing
# then read the folder, inside the must contain only video files (mp4, mov)
# read the video files and extract the frames from the video using opencv
# save the images in the ./bin folder
# the images should be saved in the same folder as this folder and the name should be the same as the video file plus the frame number
# the images should be in png format
# images size should be as original

from os import listdir
from os.path import isfile, join
from classes.image_lib import ImageAgent
from numpy import ndarray

class DatasetAgent:
    def __init__(self):
        ...
    
    def VideoExtract(self, *, src_path : str = "./datasets", dst_path : str = "./bin") -> None:
        image_agent : ImageAgent = ImageAgent()

        # Read from this folder
        for week_folder in listdir(src_path):
            week_folder_path = join(src_path, week_folder)
            if isfile(week_folder_path):
                continue

            print(f"Reading {week_folder_path}")

            # read video files in the folder
            for video in listdir(week_folder_path):
                video_path = join(week_folder_path, video)
                if not video_path.endswith((".mp4", ".mov")):
                    "Warning: Invalid video file"
                    continue

                print(f"Reading {video_path}")

                frames : list[ndarray] = image_agent.LoadVideo(video_path)

                for i, frame in enumerate(frames):
                    image_agent.SaveImage(f"{dst_path}/{week_folder}/{video.replace(".mp4","").replace(".mov","")}/{i : 07d}.png", frame)

                print(f"Saved {video_path} in {dst_path}/{week_folder}")