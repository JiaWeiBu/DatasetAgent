# python version : 3.12.6

from enum import Enum, unique
from os.path import exists, dirname
from os import makedirs
from cv2 import imread, imwrite, resize, cvtColor, VideoCapture, inRange, findContours, boundingRect, COLOR_RGB2GRAY, COLOR_GRAY2RGB, COLOR_RGB2HSV, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4, IMREAD_COLOR, IMREAD_GRAYSCALE, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, contourArea
from numpy import ndarray, array
from classes.util_lib import Size, Rect

class ImageAgent:
    """
    Agent for loading, saving, resizing, converting color, and cropping images.

    Enum:
        ColorModeEnum: Enum for different color modes.
        ColorConversionEnum: Enum for different color conversions.
        ImageInterpolationEnum: Enum for different interpolation methods for image resizing.

    Methods:
        LoadImage: Load image from file.
        SaveImage: Save image to file.
        ResizeImage: Resize image.
        ConvertColor: Convert color of image.
        CropImage: Crop image.
        LoadVideo: Load video from file.
        FindPlantMask: Find plant mask in the image using color range.
        FindPlantContour: Find plant contour in the mask.

    :example:
    >>> image_agent: ImageAgent = ImageAgent()
    """

    @unique
    class ColorModeEnum(Enum):
        """
        Enum for different color modes.

        rgb_ : RGB color mode.
        grayscale_ : Grayscale color mode.
        """
        rgb_ = IMREAD_COLOR
        grayscale_ = IMREAD_GRAYSCALE

    @unique
    class ColorConversionEnum(Enum):
        """
        Enum for different color conversions.

        rgb2gray_ : RGB to Grayscale conversion.
        gray2rgb_ : Grayscale to RGB conversion.
        rgb2hsv_ : RGB to HSV conversion.
        """
        rgb2gray_ = COLOR_RGB2GRAY
        gray2rgb_ = COLOR_GRAY2RGB
        rgb2hsv_ = COLOR_RGB2HSV

    @unique
    class ImageInterpolationEnum(Enum):
        """
        Enum for different interpolation methods for image resizing.

        nearest_ : Nearest neighbor interpolation.
        linear_ : Linear interpolation.
        cubic_ : Cubic interpolation.
        """
        nearest_ = INTER_NEAREST
        linear_ = INTER_LINEAR
        cubic_ = INTER_CUBIC
        lanczos4_ = INTER_LANCZOS4

    def __init__(self):
        pass

    def LoadImage(self, path: str, color_mode: ColorModeEnum) -> ndarray:
        """
        Load image from file.

        Args:
            path (str): Path to the image file.
            color_mode (ColorModeEnum): Color mode of the image.

        Returns:
            ndarray: Loaded image.

        :example:
        >>> image_agent: ImageAgent = ImageAgent()
        >>> image: ndarray = image_agent.LoadImage("path/to/image.jpg", ImageAgent.ColorModeEnum.rgb_)
        """
        assert exists(path), "File not found"
        image: ndarray = imread(path, color_mode.value)
        return image

    def SaveImage(self, path: str, image: ndarray) -> None:
        """
        Save image to file.

        Args:
            path (str): Path to save the image.
            image (ndarray): Image data.

        :example:
        >>> image_agent: ImageAgent = ImageAgent()
        >>> image: ndarray = image_agent.LoadImage("path/to/image.jpg", ImageAgent.ColorModeEnum.rgb_)
        >>> image_agent.SaveImage("path/to/save/image.jpg", image)
        """
        assert path.endswith((".jpg", ".png")), "Invalid file format"

        # Ensure the directory exists
        directory = dirname(path)
        if not exists(directory):
            makedirs(directory)

        # Save the image
        if not imwrite(path, image):
            raise IOError(f"Failed to save image to {path}")

    def ResizeImage(self, image: ndarray, size: Size[int], interpolation: ImageInterpolationEnum = ImageInterpolationEnum.linear_) -> ndarray:
        """
        Resize image.

        Args:
            image (ndarray): Image data.
            size (Size[int]): Size to resize the image.
            interpolation (ImageInterpolation): Interpolation method.

        Returns:
            ndarray: Resized image.

        :example:
        >>> image_agent: ImageAgent = ImageAgent()
        >>> image: ndarray = image_agent.LoadImage("path/to/image.jpg", ImageAgent.ColorModeEnum.rgb_)
        >>> resized_image: ndarray = image_agent.ResizeImage(image, Size(100, 100), ImageAgent.ImageInterpolation.nearest_)
        """
        assert size.width_ > 0 and size.height_ > 0, "Invalid size"
        return resize(image, (size.width_, size.height_), interpolation=interpolation.value)

    def ConvertColor(self, image: ndarray, conversion: ColorConversionEnum) -> ndarray:
        """
        Convert color of image.

        Args:
            image (ndarray): Image data.
            conversion (ColorConversionEnum): Color conversion method.

        Returns:
            ndarray: Converted image.

        :example:
        >>> image_agent: ImageAgent = ImageAgent()
        >>> image: ndarray = image_agent.LoadImage("path/to/image.jpg", ImageAgent.ColorModeEnum.rgb_)
        >>> converted_image: ndarray = image_agent.ConvertColor(image, ImageAgent.ColorConversionEnum.rgb2gray_)
        """
        return cvtColor(image, conversion.value)

    def CropImage(self, image: ndarray, rect: Rect[int]) -> ndarray:
        """
        Crop image.

        Args:
            image (ndarray): Image data.
            rect (Rect[int]): Rectangle to crop the image.

        Returns:
            ndarray: Cropped image.

        :example:
        >>> image_agent: ImageAgent = ImageAgent()
        >>> image: ndarray = image_agent.LoadImage("path/to/image.jpg", ImageAgent.ColorModeEnum.rgb_)
        >>> cropped_image: ndarray = image_agent.CropImage(image, Rect(10, 10, 50, 50))
        """
        assert rect.size_.width_ > 0 and rect.size_.height_ > 0, "Invalid rectangle size"
        return image[rect.point_.y_:rect.point_.y_ + rect.size_.height_, rect.point_.x_:rect.point_.x_ + rect.size_.width_]

    def LoadVideo(self, path: str, frame_rate: int) -> list[ndarray]:
        """
        Load video from file.

        Args:
            path (str): Path to the video file.
            frame_rate (int): Frame rate of the video.

        Returns:
            list[ndarray]: List of frames.

        :example:
        >>> image_agent: ImageAgent = ImageAgent()
        >>> frames: list[ndarray] = image_agent.LoadVideo("path/to/video.mp4", 30)
        """
        assert exists(path), "File not found"
        assert path.endswith((".mp4", ".mov")), "Invalid file format"

        video = VideoCapture(path)
        frames = []
        count: int = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if count % frame_rate == 0:
                frames.append(frame)
            count += 1

        video.release()
        return frames

    def FindPlantMask(self, image: ndarray, lower_color: list[int] = [35, 40, 40], upper_color: list[int] = [85, 255, 255]) -> ndarray:
        """
        Find plant mask in the image using color range.

        Args:
            image (ndarray): Image data.
            lower_color (list[int]): Lower color range.
            upper_color (list[int]): Upper color range.

        Returns:
            ndarray: Plant mask.

        :example:
        >>> image_agent: ImageAgent = ImageAgent()
        >>> image: ndarray = image_agent.LoadImage("path/to/image.jpg", ImageAgent.ColorModeEnum.rgb_)
        >>> mask: ndarray = image_agent.FindPlantMask(image)
        """
        hsv = cvtColor(image, self.ColorConversionEnum.rgb2hsv_.value)
        np_lower_color = array(lower_color)
        np_upper_color = array(upper_color)
        mask = inRange(hsv, np_lower_color, np_upper_color)
        return mask

    def FindPlantContour(self, mask: ndarray) -> list[Rect[int]] | None:
        """
        Find plant contour in the mask.

        Args:
            mask (ndarray): Plant mask.

        Returns:
            list[Rect[int]] | None: List of bounding boxes of the plant contours or None if no contours are found.

        :example:
        >>> image_agent: ImageAgent = ImageAgent()
        >>> image: ndarray = image_agent.LoadImage("path/to/image.jpg", ImageAgent.ColorModeEnum.rgb_)
        >>> mask: ndarray = image_agent.FindPlantMask(image)
        >>> plant_contours: list[Rect[int]] = image_agent.FindPlantContour(mask)
        """
        contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contours = sorted(contours, key=contourArea, reverse=True)[:5]
            plant_contours: list[Rect[int]] = []
            for contour in largest_contours:
                x, y, w, h = boundingRect(contour)
                plant_contours.append(Rect(x, y, w, h))
            return plant_contours
        else:
            return None