# python version : 3.12.6
from enum import Enum, unique
from os import listdir
from os.path import isfile, join
from numpy import ndarray
from classes.image_lib import ImageAgent 


class ImageDatasetAgent:
    """
    Agent for dataset operations.
    Used for cropping plant from the images for dataset creation.

    Enum:
        ImageAngleEnum: Enum for different angles of the plant images.

    Attributes:
        image_agent_ (ImageAgent): Image agent for image operations.
        img_extensions_ (tuple[str, ...] | str): Image file extensions to read.
        angle_dict_ (dict[str, ImageAngle]): Dictionary for image angles.
    
    Methods:
        PlantExtract: Extract plant images from the dataset folder.
        GetImageAngle: Get the image angle from the image path.
    
    :example:
    >>> dataset_agent : ImageDatasetAgent = ImageDatasetAgent()
    """

    @unique
    class ImageAngleEnum(Enum):
        """
        Enum for different angles of the plant images.

        60DEGREES : Side view of the plant at 60 degrees
        SIDE : Side view of the plant
        TOP : Top view of the plant
        UNKNOWN : Unknown angle of the plant
        """
        degrees60_ = "60degrees"
        side_ = "side"
        top_ = "top"
        unknown_ = "unknown"

    def __init__(self, img_extensions : tuple[str, ...] | str = (".png")) -> None:
        """
        Initialize the dataset agent.

        Args:
            img_extensions (tuple[str, ...] | str): Image file extensions to read.

        :example:
        dataset_agent : ImageDatasetAgent = ImageDatasetAgent()
        """
        self.image_agent_ : ImageAgent = ImageAgent()
        self.img_extensions_ : tuple[str, ...] | str = img_extensions
        self.angle_dict_ : dict[str, ImageDatasetAgent.ImageAngleEnum] = {
            "60degrees" : ImageDatasetAgent.ImageAngleEnum.degrees60_,
            "side" : ImageDatasetAgent.ImageAngleEnum.side_,
            "top" : ImageDatasetAgent.ImageAngleEnum.top_
        }

    def PlantExtract(self, *, src_path : str = "./bin", dst_path : str = "./bin/bin", filter_angle : list[ImageAngleEnum] = [ImageAngleEnum.top_]) -> None:
        """
        (NOT COMPLETED)
        Extract plant images from the dataset folder using image processing.
        Crop the plant roi from the source image and save it to the destination folder.

        Image Source Structure:
        - bin
            - week1
                - angle1.png
                    - 0000000.png
                    - 0000001.png
                    - ...

        Image Destination Structure:
        - bin
            - bin
                - week1
                    - angle1
                        - 0000000.png
                        - 0000001.png
                        - ...

        Args:
            src_path (str): Path to the dataset folder.
            dst_path (str): Path to save the extracted images.
        
        :example:
        >>> dataset_agent : ImageDatasetAgent = ImageDatasetAgent()
        >>> dataset_agent.PlantExtract()
        """

        # Read from this folder
        for week_folder in listdir(src_path):
            week_folder_path = join(src_path, week_folder)
            if isfile(week_folder_path):
                continue

            print(f"Reading {week_folder_path}")

            # Read angle from this folder
            for angle_folder in listdir(week_folder_path):
                angle_folder_path = join(week_folder_path, angle_folder)
                if isfile(angle_folder_path):
                    continue
                
                image_angle : ImageDatasetAgent.ImageAngleEnum = ImageDatasetAgent.ImageAngleEnum(angle_folder)

                if image_angle not in filter_angle:
                    print(f"Skipping {angle_folder_path} as {image_angle.value}")
                    continue

                print(f"Reading {angle_folder_path} as {image_angle.value}")

                # read image files in the folder
                for images in listdir(angle_folder_path):
                    image_path = join(angle_folder_path, images)
                    if not image_path.endswith(self.img_extensions_):
                        print(f"Warning: Invalid image file on {image_path}")
                        continue

                    print(f"Reading {image_path}")

                    image : ndarray = self.image_agent_.LoadImage(image_path, ImageAgent.ColorModeEnum.rgb_)

                    # Crop the plant from the image
                    

    def GetImageAngle(self, path : str) -> ImageDatasetAgent.ImageAngleEnum:
        """
        Get the image angle from the image path.

        Args:
            path (str): Path to the image file.

        Returns:
            ImageAngle: Image angle enum value.

        :example:
        >>> dataset_agent : ImageDatasetAgent = ImageDatasetAgent()
        >>> angle : ImageAngle = dataset_agent.ImageAngle("top/image.png") # Output: ImageAngle.top_
        """

        # if path contains angle_dict_ key, return the corresponding value
        # the path may include other characters Eg: SideView-232455252-Week1
        # so we need to lower and find the key in the path

        path_lower = path.lower()
        for angle, angle_enum in self.angle_dict_.items():
            # dont look for unknown
            if angle_enum == ImageDatasetAgent.ImageAngleEnum.unknown_:
                continue

            if angle in path_lower:
                return angle_enum
        return ImageDatasetAgent.ImageAngleEnum.unknown_