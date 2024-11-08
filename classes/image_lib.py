# python version : 3.12.6 

from os.path import exists, dirname
from os import makedirs, listdir
from cv2 import imread, imwrite, resize, cvtColor, VideoCapture, imshow, waitKey, destroyAllWindows, inRange, findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, boundingRect, contourArea
from numpy import ndarray, array

from classes.enum import ColorMode, ImageInterpolation, ColorConversion
from classes.util_lib import Size, Rect

class ImageAgent:
    """
    Agent for image operations.
    Used for loading, saving, resizing, converting color, and cropping images.
    """

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
        assert exists(dirname(path)), f"{dirname(path)} not found"

        # list dir in path
        print(listdir(dirname(path)))

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

    def LoadVideo(self, path: str, frame_rate : int) -> list[ndarray]:
        """
        Load video from file.

        Args:
            path (str): Path to the video file.
            frame_rate (int): Frame rate of the video.

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
        frames = []
        count : int = 0
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
        >>> image_agent : ImageAgent = ImageAgent()
        >>> image : ndarray = image_agent.LoadImage("path/to/image.jpg", ColorMode.rgb)
        >>> mask : ndarray = image_agent.FindPlantMask(image)
        """
        # Convert image to HSV color space
        hsv = cvtColor(image, ColorConversion.rgb2hsv_.value)

        np_lower_color = array(lower_color)
        np_upper_color = array(upper_color)

        # Create a mask for the green color (plant)
        mask = inRange(hsv, np_lower_color, np_upper_color)

        return mask

    def FindPlantContour(self, mask: ndarray) -> list[Rect[int]] | None:
        """
        Find plant contour in the mask.

        Args:
            mask (ndarray): Plant mask.
        
        Returns:
            Rect: Bounding box of the plant contour.

        :example:
        >>> image_agent : ImageAgent = ImageAgent()
        >>> image : ndarray = image_agent.LoadImage("path/to/image.jpg", ColorMode.rgb)
        >>> mask : ndarray = image_agent.FindPlantMask(image)
        >>> plant_contour : Rect = image_agent.FindPlantContour(mask)
        """
        contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

        if contours:
            # largest_contour = max(contours, key=contourArea)
            # x, y, w, h = boundingRect(largest_contour)
            # return [Rect(x, y, w, h)]

            # Get 5 largest contours
            largest_contours = sorted(contours, key=contourArea, reverse=True)[:5]
            plant_contours : list[Rect[int]] = []
            for contour in largest_contours:
                x, y, w, h = boundingRect(contour)
                plant_contours.append(Rect(x, y, w, h))
            return plant_contours
        else:
            return None