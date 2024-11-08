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

    def __init__(self, vid_extensions : tuple[str, ...] | str = (".mp4", ".mov"), img_extensions : str = "png", frame_rate : int = 60) -> None:
        """
        Initialize the dataset agent.

        Args:
            vid_extensions (tuple[str, ...] | str): Video file extensions to read.
            img_extensions (str): Image file extension to save.
            frame_rate (int): Frame rate for video extraction.

        :example:
        >>> dataset_agent : VideoDatasetAgent = VideoDatasetAgent()
        """
        self.image_agent_ : ImageAgent = ImageAgent()
        self.vid_extensions_ : tuple[str, ...] | str = vid_extensions
        self.img_extensions_ : str = img_extensions
        self.frame_rate_ : int = frame_rate
    
    def VideoExtract(self, *, src_path : str = "./datasets", dst_path : str = "./bin") -> None:
        """
        Extract images from video files in the dataset folder.

        Video Source Structure:
        - datasets
            - week1
                - video1.mp4

        Image Destination Structure:
        - bin
            - week1
                - video1
                    - 0000000.png
                    - 0000001.png
                    - ...

        Args:
            src_path (str): Path to the dataset folder.
            dst_path (str): Path to save the extracted images.
        
        :example:
        >>> dataset_agent : VideoDatasetAgent = VideoDatasetAgent()
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
        >>> dataset_agent : VideoDatasetAgent = VideoDatasetAgent()
        >>> dataset_agent.SaveImages(frames, dst_path)
        """
        for i, frame in enumerate(frames):
            self.image_agent_.SaveImage(f"{dst_path}/{i : 07d}.{self.img_extensions_}", frame)

    def StripExtension(self, path : str, extensions : tuple[str, ...] | str) -> str:
        """
        Strip the extension from the path.

        Args:
            path (str): Path to strip the extension from.
            extensions (tuple[str, ...]): Extensions to strip
        
        :example:
        >>> dataset_agent : VideoDatasetAgent = VideoDatasetAgent()
        >>> path = "file.txt"
        >>> extensions = (".txt", ".md")
        >>> print(dataset_agent.StripExtension(path, extensions)) # Output: "file"
        """
        # Type String Case
        if isinstance(extensions, str):
            return path.replace(extensions, "")

        # Type Tuple Case
        for ext in extensions:
            if path.endswith(ext):
                return path.replace(ext, "")
        return path