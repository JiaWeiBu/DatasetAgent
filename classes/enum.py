# python version : 3.12.6 
from enum import Enum, unique
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4
from cv2 import COLOR_RGB2GRAY, COLOR_GRAY2RGB

# Enum for color mode
@unique
class ColorMode(Enum):
    """
    Enum for different color modes for image processing.

    RGB : Red-Green-Blue color mode
    GRAYSCALE : Grayscale color mode
    """
    rgb_ = IMREAD_COLOR
    grayscale_ = IMREAD_GRAYSCALE

# Enum for color conversion
@unique
class ColorConversion(Enum):
    """
    Enum for different color conversion methods for image processing.

    RGB2GRAY : Convert RGB to Grayscale
    GRAY2RGB : Convert Grayscale to RGB
    """
    rgb2gray_ = COLOR_RGB2GRAY
    gray2rgb_ = COLOR_GRAY2RGB

@unique
class ImageInterpolation(Enum):
    """
    Enum for different interpolation methods for image resizing.

    NEAREST : Nearest-neighbor interpolation
    LINEAR : Bilinear interpolation
    CUBIC : Bicubic interpolation
    LANCZOS4 : Lanczos interpolation
    """
    nearest_ = INTER_NEAREST
    linear_ = INTER_LINEAR
    cubic_ = INTER_CUBIC
    lanczos4_ = INTER_LANCZOS4