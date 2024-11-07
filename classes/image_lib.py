# python version : 3.12.6 

from classes.enum import ColorMode, ImageInterpolation, ColorConversion
from cv2 import imread, imwrite, resize, cvtColor, VideoCapture, imshow, waitKey, destroyAllWindows
from numpy import ndarray
from classes.util_lib import Size, Rect
from os.path import exists, dirname
from os import makedirs

class ImageAgent:
    def __init__(self):
        ...
    
    def LoadImage(self, path: str, color_mode: ColorMode) -> ndarray:
        """
        Load image from file.

        Args:
            path (str): Path to the image file.
            color_mode (ColorMode): Color mode of the image.

        Returns:
            ndarray: Image data.

        :example:
        >>> image_agent : ImageAgent = ImageAgent()
        >>> image : ndarray = image_agent.LoadImage("path/to/image.jpg", ColorMode.rgb
        """
        assert exists(path), "File not found"

        image : ndarray = imread(path, color_mode.value)
        return image

    def SaveImage(self, path: str, image: ndarray) -> None:
        """
        Save image to file.

        Args:
            path (str): Path to save the image.
            image (ndarray): Image data.

        :example:
        >>> image_agent : ImageAgent = ImageAgent()
        >>> image : ndarray = image_agent.LoadImage("path/to/image.jpg", ColorMode.rgb)
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

    def ResizeImage(self, image: ndarray, size: Size[int], interpolation: ImageInterpolation = ImageInterpolation.linear_) -> ndarray:
        """
        Resize image.

        Args:
            image (ndarray): Image data.
            size (Size): Size to resize the image.
            interpolation (int): Interpolation method.

        Returns:
            ndarray: Resized image.

        :example:
        >>> image_agent : ImageAgent = ImageAgent()
        >>> image : ndarray = image_agent.LoadImage("path/to/image.jpg", ColorMode.rgb)
        >>> resized_image : ndarray = image_agent.ResizeImage(image, Size(100, 100), ImageInterpolation.nearest)
        """
        assert size.width_ > 0 and size.height_ > 0, "Invalid size"

        return resize(image, (size.width_, size.height_), interpolation=interpolation.value)

    def ConvertColor(self, image: ndarray, conversion: ColorConversion) -> ndarray:
        """
        Convert color of image.

        Args:
            image (ndarray): Image data.
            conversion (ColorConversion): Color conversion method.

        Returns:
            ndarray: Image with converted color.

        :example:
        >>> image_agent : ImageAgent = ImageAgent()
        >>> image : ndarray = image_agent.LoadImage("path/to/image.jpg", ColorMode.rgb)
        >>> converted_image : ndarray = image_agent.ConvertColor(image, ColorConversion.rgb2gray)
        """
        return cvtColor(image, conversion.value)

    def CropImage(self, image: ndarray, rect : Rect[int]) -> ndarray:
        """
        Crop image.

        Args:
            image (ndarray): Image data.
            rect (Rect): Rectangle to crop the image.

        Returns:
            ndarray: Cropped image.

        :example:
        >>> image_agent : ImageAgent = ImageAgent()
        >>> image : ndarray = image_agent.LoadImage("path/to/image.jpg", ColorMode.rgb)
        >>> cropped_image : ndarray = image_agent.CropImage(image, Rect(0, 0, 100, 100))
        """
        assert rect.point_.x_ >= 0 and rect.point_.y_ >= 0, "Invalid rectangle point"
        assert rect.size_.width_ > 0 and rect.size_.height_ > 0, "Invalid rectangle size"

        return image[rect.point_.y_:rect.point_.y_+rect.size_.height_, rect.point_.x_:rect.point_.x_+rect.size_.width_]

    def LoadVideo(self, path: str) -> list[ndarray]:
        """
        Load video from file.

        Args:
            path (str): Path to the video file.

        Returns:
            list[ndarray]: List of frames.

        :example:
        >>> image_agent : ImageAgent = ImageAgent()
        >>> frames : list[ndarray] = image_agent.LoadVideo("path/to/video.mp4")
        """
        # verify the path is a video file
        assert exists(path), "File not found"
        assert path.endswith((".mp4", ".mov")), "Invalid file format"

        video = VideoCapture(path)
        
        # Extract frames from video
        # frames = []
        # while video.isOpened():
        #     ret, frame = video.read()
        #     if not ret:
        #         break
        #     frames.append(frame)
        frames : list[ndarray] = [frame for ret, frame in iter(lambda: video.read(), (False, None)) if ret]

        video.release()
        return frames
