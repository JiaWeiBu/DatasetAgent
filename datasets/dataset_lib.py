# python version : 3.12.6

from os import listdir
from os.path import isfile, join
from numpy import ndarray
from classes.image_lib import ImageAgent


class VideoDatasetAgent:
    """
    Agent for dataset operations.
    Used for extracting images from video files.
    """

    def __init__(self, vid_extensions : tuple[str, ...] = (".mp4", ".mov"), img_extensions : str = "png", frame_rate : int = 60) -> None:
        """
        Initialize the dataset agent.

        Args:
            vid_extensions (tuple[str, str]): Video file extensions to read.
            img_extensions (str): Image file extension to save.
            frame_rate (int): Frame rate for video extraction.

        :example:
        >>> dataset_agent : DatasetAgent = DatasetAgent()
        """
        self.image_agent_ : ImageAgent = ImageAgent()
        self.vid_extensions_ : tuple[str, ...] = vid_extensions
        self.img_extensions_ : str = img_extensions
        self.frame_rate_ : int = frame_rate
    
    def VideoExtract(self, *, src_path : str = "./datasets", dst_path : str = "./bin") -> None:
        """
        Extract images from video files in the dataset folder.

        Args:
            src_path (str): Path to the dataset folder.
            dst_path (str): Path to save the extracted images.
        
        :example:
        >>> dataset_agent : DatasetAgent = DatasetAgent()
        >>> dataset_agent.VideoExtract()
        """
        # Read from this folder
        for week_folder in listdir(src_path):
            week_folder_path = join(src_path, week_folder)
            if isfile(week_folder_path):
                continue

            print(f"Reading {week_folder_path}")

            # read video files in the folder
            for video in listdir(week_folder_path):
                video_path = join(week_folder_path, video)
                if not video_path.endswith(self.vid_extensions_):
                    print(f"Warning: Invalid video file on {video_path}")
                    continue

                print(f"Reading {video_path}")

                frames : list[ndarray] = self.image_agent_.LoadVideo(video_path, self.frame_rate_)

                self.SaveImages(frames, f"{dst_path}/{week_folder}/{self.StripExtension(video, self.vid_extensions_)}")

                print(f"Saved {video_path} in {dst_path}/{week_folder}")

    

    def SaveImages(self, frames : list[ndarray], dst_path : str):
        """
        Save the extracted images.

        Args:
            frames (list[ndarray]): Extracted images.
            dst_path (str): Path to save the images.

        :example:
        >>> dataset_agent : DatasetAgent = DatasetAgent()
        >>> dataset_agent.SaveImages(frames, dst_path)
        """
        for i, frame in enumerate(frames):
            self.image_agent_.SaveImage(f"{dst_path}/{i : 07d}.{self.img_extensions_}", frame)

    def StripExtension(self, path : str, extensions : tuple[str, ...]) -> str:
        """
        Strip the extension from the path.

        Args:
            path (str): Path to strip the extension from.
            extensions (tuple[str, ...]): Extensions to strip
        
        :example:
        >>> dataset_agent : DatasetAgent = DatasetAgent()
        >>> path = "file.txt"
        >>> extensions = (".txt", ".md")
        >>> print(dataset_agent.StripExtension(path, extensions)) # Output: "file"
        """

        for ext in extensions:
            if path.endswith(ext):
                return path.replace(ext, "")
        return path