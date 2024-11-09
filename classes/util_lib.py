# python version : 3.12.6 

from typing import TypeVar, Generic, Callable
# Deprecated Decorator
def Deprecated(message: str) -> Callable:
    """
    Decorator to mark a function as deprecated

    :example:
    >>> @deprecated("This function is deprecated")
    >>> def old_function() -> None:
    >>>     pass
    """
    def decorator(func) -> Callable:
        def wrapper(*args, **kwargs) -> Callable:
            print(f"WARNING: {func.__name__} is deprecated. {message}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Unused Function
def Unused(*args) -> None:
    """
    Pass any unused variable to make code editor think the variable is used

    :example:
    >>> Unused(1)
    """
    pass

# Size, Point, Rect classes
"""
Size, Point, and Rect classes for 2D dimensions

:methods:
Add : Adds a value to the object
Sub : Subtracts a value from the object
Mul : Multiplies a value with the object
TrueDiv : Divides the object by a value
FloorDiv : Divides the object by a value and returns the floor value
Mod : Returns the modulus of the object and a value
Pow : Returns the object raised to the power of a value
__add__ : Adds two objects together
__sub__ : Subtracts two objects together
__mul__ : Multiplies two objects together
__truediv__ : Divides two objects together
__floordiv__ : Divides two objects together and returns the floor value
__mod__ : Returns the modulus of two objects
__pow__ : Returns the object raised to the power of another object
__eq__ : Checks if two objects are equal
__ne__ : Checks if two objects are not equal
__lt__ : Checks if one object is less than another
__le__ : Checks if one object is less than or equal to another
__gt__ : Checks if one object is greater than another
__ge__ : Checks if one object is greater than or equal to another
"""

T = TypeVar('T', int, float)

class Size(Generic[T]):
    """
    Size class for 2D dimensions
    """

    def __init__(self, width: T, height: T) -> None:
        """
        Constructor for Size class
        
        :example:
        >>> square: Size[int] = Size[int](10, 10)
        """
        self.width_: T = width
        self.height_: T = height
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Size object

        :example:
        >>> square: Size[int] = Size[int](10, 10)
        >>> print(square) # Size(width=10, height=10)
        """
        self.ClassValidator()
        return f"Size(width={self.width_}, height={self.height_})"

    def Add(self, value: T) -> 'Size':
        """
        Adds a value to the Size object

        :example:
        >>> square: Size[int] = Size[int](10, 10)
        >>> print(square.Add(5)) # Size(width=15, height=15)
        """
        self.ClassValidator()
        return Size(self.width_ + value, self.height_ + value)

    def Sub(self, value: T) -> 'Size':
        """
        Subtracts a value from the Size object

        :example:
        >>> square: Size[int] = Size[int](10, 10)
        >>> print(square.Sub(5)) # Size(width=5, height=5)
        """
        self.ClassValidator()
        return Size(self.width_ - value, self.height_ - value)

    def Mul(self, value: T) -> 'Size':
        """
        Multiplies a value with the Size object

        :example:
        >>> square: Size[int] = Size[int](10, 10)
        >>> print(square.Mul(5)) # Size(width=50, height=50)
        """
        self.ClassValidator()
        return Size(self.width_ * value, self.height_ * value)
    
    def TrueDiv(self, value: T) -> 'Size':
        """
        Divides the Size object by a value

        :example:
        >>> square: Size[int] = Size[int](10, 10)
        >>> print(square.TrueDiv(5)) # Size(width=2.0, height=2.0)
        """
        self.ClassValidator()
        return Size(self.width_ / value, self.height_ / value)

    def FloorDiv(self, value: T) -> 'Size':
        """
        Divides the Size object by a value and returns the floor value

        :example:
        >>> square: Size[int] = Size[int](10, 10)
        >>> print(square.FloorDiv(5)) # Size(width=2, height=2)
        """
        self.ClassValidator()
        return Size(self.width_ // value, self.height_ // value)

    def Mod(self, value: T) -> 'Size':
        """
        Returns the modulus of the Size object and a value

        :example:
        >>> square: Size[int] = Size[int](10, 10)
        >>> print(square.Mod(5))
        """
        self.ClassValidator()
        return Size(self.width_ % value, self.height_ % value)
    
    def Pow(self, value: T) -> 'Size':
        """
        Returns the Size object raised to the power of a value

        :example:
        >>> square: Size[int] = Size[int](10, 10)
        >>> print(square.Pow(2)) # Size(width=100, height=100)
        """
        self.ClassValidator()
        return Size(self.width_ ** value, self.height_ ** value)

    def __add__(self, other: 'Size') -> 'Size':
        """
        Adds two Size objects together

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](20, 20)
        >>> print(square1 + square2) # Size(width=30, height=30)
        """
        self.OtherValidator(other)
        return Size(self.width_ + other.width_, self.height_ + other.height_)

    def __sub__(self, other: 'Size') -> 'Size':
        """
        Subtracts two Size objects together

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](20, 20)
        >>> print(square1 - square2) # Size(width=-10, height=-10)
        """
        self.OtherValidator(other)
        return Size(self.width_ - other.width_, self.height_ - other.height_)

    def __mul__(self, other: 'Size') -> 'Size':
        """
        Multiplies two Size objects together

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](20, 20)
        >>> print(square1 * square2) # Size(width=200, height=200)
        """
        self.OtherValidator(other)
        return Size(self.width_ * other.width_, self.height_ * other.height_)

    def __truediv__(self, other: 'Size') -> 'Size':
        """
        Divides two Size objects together

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](20, 20)
        >>> print(square1 / square2) # Size(width=0.5, height=0.5)
        """
        self.OtherValidator(other)
        return Size(self.width_ / other.width_, self.height_ / other.height_)

    def __floordiv__(self, other: 'Size') -> 'Size':
        """
        Divides two Size objects together and returns the floor value

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](20, 20)
        >>> print(square1 // square2)
        """
        self.OtherValidator(other)
        return Size(self.width_ // other.width_, self.height_ // other.height_)

    def __mod__(self, other: 'Size') -> 'Size':
        """
        Returns the modulus of two Size objects

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](20, 20)
        >>> print(square1 % square2)
        """
        self.OtherValidator(other)
        return Size(self.width_ % other.width_, self.height_ % other.height_)

    def __pow__(self, other: 'Size') -> 'Size':
        """
        Returns the Size object raised to the power of another Size object

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](2, 2)
        >>> print(square1 ** square2) # Size(width=100, height=100)
        """
        self.OtherValidator(other)
        return Size(self.width_ ** other.width_, self.height_ ** other.height_)

    def __eq__(self, other: object) -> bool:
        """
        Checks if two Size objects are equal

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](10, 10)
        >>> rint(square1 == square2) # True
        """
        assert isinstance(other, Size), "other must be of type Size"
        self.OtherValidator(other)
        return self.width_ == other.width_ and self.height_ == other.height_

    def __ne__(self, other: object) -> bool:
        """
        Checks if two Size objects are not equal

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](20, 20)
        >>> print(square1 != square2) # True
        """
        assert isinstance(other, Size), "other must be of type Size"
        self.OtherValidator(other)
        return self.width_ != other.width_ or self.height_ != other.height_

    def __lt__(self, other: 'Size') -> bool:
        """
        Checks if one Size object is less than another

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](20, 20)
        >>> print(square1 < square2) # True
        """
        self.OtherValidator(other)
        return self.width_ < other.width_ and self.height_ < other.height_

    def __le__(self, other: 'Size') -> bool:
        """
        Checks if one Size object is less than or equal to another

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](20, 20)
        >>> print(square1 <= square2) # True
        """
        self.OtherValidator(other)
        return self.width_ <= other.width_ and self.height_ <= other.height_

    def __gt__(self, other: 'Size') -> bool:
        """
        Checks if one Size object is greater than another

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](20, 20)
        >>> print(square1 > square2) # False
        """
        self.OtherValidator(other)
        return self.width_ > other.width_ and self.height_ > other.height_

    def __ge__(self, other: 'Size') -> bool:
        """
        Checks if one Size object is greater than or equal to another

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](20, 20)
        >>> print(square1 >= square2) # False
        """
        assert isinstance(other, Size), "other must be of type Size"
        return self.width_ >= other.width_ and self.height_ >= other.height_

    def ClassValidator(self) -> None:
        """
        Validates the class type

        :example:
        >>> square: Size[int] = Size[int](10, 10)
        >>> square.ClassValidator()
        """
        assert isinstance(self.width_, type(self.width_)), f"width_ must be of type {type(self.width_)}"
        assert isinstance(self.height_, type(self.height_)), f"height_ must be of type {type(self.height_)}"

    def OtherValidator(self, other: 'Size') -> None:
        """
        Validates the other type

        :example:
        >>> square1: Size[int] = Size[int](10, 10)
        >>> square2: Size[int] = Size[int](20, 20)
        >>> square1.OtherValidator(square2)
        """
        assert isinstance(other, Size), "other must be of type Size"
        assert isinstance(other.width_, type(self.width_)), f"width_ must be of type {type(self.width_)}"
        assert isinstance(other.height_, type(self.height_)), f"height_ must be of type {type(self.height_)}"

class Point(Generic[T]):
    """
    Point class for 2D coordinates
    """

    def __init__(self, x: T, y: T) -> None:
        """
        Constructor for Point class
        
        :example:
        >>> point: Point[int] = Point[int](10, 10)
        """
        self.x_: T = x
        self.y_: T = y
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Point object

        :example:
        >>> point: Point[int] = Point[int](10, 10)
        >>> print(point) # Point(x=10, y=10)
        """
        self.ClassValidator()
        return f"Point(x={self.x_}, y={self.y_})"

    def Add(self, value: T) -> 'Point':
        """
        Adds a value to the Point object

        :example:
        >>> point: Point[int] = Point[int](10, 10)
        >>> print(point.Add(5)) # Point(x=15, y=15)
        """
        self.ClassValidator()
        return Point(self.x_ + value, self.y_ + value)

    def Sub(self, value: T) -> 'Point':
        """
        Subtracts a value from the Point object

        :example:
        >>> point: Point[int] = Point[int](10, 10)
        >>> print(point.Sub(5)) # Point(x=5, y=5)
        """
        self.ClassValidator()
        return Point(self.x_ - value, self.y_ - value)

    def Mul(self, value: T) -> 'Point':
        """
        Multiplies a value with the Point object

        :example:
        >>> point: Point[int] = Point[int](10, 10)
        >>> print(point.Mul(5)) # Point(x=50, y=50)
        """
        self.ClassValidator()
        return Point(self.x_ * value, self.y_ * value)
    
    def TrueDiv(self, value: T) -> 'Point':
        """
        Divides the Point object by a value

        :example:
        >>> point: Point[int] = Point[int](10, 10)
        >>> print(point.TrueDiv(5)) # Point(x=2.0, y=2.0)
        """
        self.ClassValidator()
        return Point(self.x_ / value, self.y_ / value)

    def FloorDiv(self, value: T) -> 'Point':
        """
        Divides the Point object by a value and returns the floor value

        :example:
        >>> point: Point[int] = Point[int](10, 10)
        >>> print(point.FloorDiv(5)) # Point(x=2, y=2)
        """
        self.ClassValidator()
        return Point(self.x_ // value, self.y_ // value)

    def Mod(self, value: T) -> 'Point':
        """
        Returns the modulus of the Point object and a value

        :example:
        >>> point: Point[int] = Point[int](10, 10)
        >>> print(point.Mod(5)) # Point(x=0, y=0)
        """
        self.ClassValidator()
        return Point(self.x_ % value, self.y_ % value)
    
    def Pow(self, value: T) -> 'Point':
        """
        Returns the Point object raised to the power of a value

        :example:
        >>> point: Point[int] = Point[int](10, 10)
        >>> print(point.Pow(2)) # Point(x=100, y=100)
        """
        self.ClassValidator()
        return Point(self.x_ ** value, self.y_ ** value)

    def __add__(self, other: 'Point') -> 'Point':
        """
        Adds two Point objects together

        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](20, 20)
        >>> print(point1 + point2) # Point(x=30, y=30)
        """
        self.OtherValidator(other)
        return Point(self.x_ + other.x_, self.y_ + other.y_)

    def __sub__(self, other: 'Point') -> 'Point':
        """
        Subtracts two Point objects together

        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](20, 20)
        >>> print(point1 - point2) # Point(x=-10, y=-10)
        """
        self.OtherValidator(other)
        return Point(self.x_ - other.x_, self.y_ - other.y_)

    def __mul__(self, other: 'Point') -> 'Point':
        """
        Multiplies two Point objects together

        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](20, 20)
        >>> print(point1 * point2) # Point(x=200, y=200)
        """
        self.OtherValidator(other)
        return Point(self.x_ * other.x_, self.y_ * other.y_)

    def __truediv__(self, other: 'Point') -> 'Point':
        """
        Divides two Point objects together

        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](20, 20)
        >>> print(point1 / point2) # Point(x=0.5, y=0.5)
        """
        self.OtherValidator(other)
        return Point(self.x_ / other.x_, self.y_ / other.y_)
    
    def __floordiv__(self, other: 'Point') -> 'Point':
        """
        Divides two Point objects together and returns the floor value

        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](20, 20)
        >>> print(point1 // point2) # Point(x=0, y=0)
        """
        self.OtherValidator(other)
        return Point(self.x_ // other.x_, self.y_ // other.y_)

    def __mod__(self, other: 'Point') -> 'Point':
        """
        Returns the modulus of two Point objects

        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](20, 20)
        >>> print(point1 % point2) # Point(x=10, y=10)
        """
        self.OtherValidator(other)
        return Point(self.x_ % other.x_, self.y_ % other.y_)
    
    def __pow__(self, other: 'Point') -> 'Point':
        """
        Returns the Point object raised to the power of another Point object

        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](2, 2)
        >>> print(point1 ** point2) # Point(x=100, y=100)
        """
        self.OtherValidator(other)
        return Point(self.x_ ** other.x_, self.y_ ** other.y_)

    def __eq__(self, other: object) -> bool:
        """
        Checks if two Point objects are equal

        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](10, 10)
        >>> print(point1 == point2) # True
        """
        assert isinstance(other, Point), "other must be of type Point"
        self.OtherValidator(other)
        return self.x_ == other.x_ and self.y_ == other.y_
    
    def __ne__(self, other: object) -> bool:
        """
        Checks if two Point objects are not equal

        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](20, 20)
        >>> print(point1 != point2) # True
        """
        assert isinstance(other, Point), "other must be of type Point"
        self.OtherValidator(other)
        return self.x_ != other.x_ or self.y_ != other.y_

    def __lt__(self, other: 'Point') -> bool:
        """
        Checks if one Point object is less than another

        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](20, 20)
        >>> print(point1 < point2) # True
        """
        self.OtherValidator(other)
        return self.x_ < other.x_ and self.y_ < other.y_

    def __le__(self, other: 'Point') -> bool:
        """
        Checks if one Point object is less than or equal to another

        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](20, 20)
        >>> print(point1 <= point2) # True
        """
        self.OtherValidator(other)
        return self.x_ <= other.x_ and self.y_ <= other.y_

    def __gt__(self, other: 'Point') -> bool:
        """
        Checks if one Point object is greater than another

        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](20, 20)
        >>> print(point1 > point2) # False
        """
        self.OtherValidator(other)
        return self.x_ > other.x_ and self.y_ > other.y_

    def __ge__(self, other: 'Point') -> bool:
        """
        Checks if one Point object is greater than or equal to another

        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](20, 20)
        >>> print(point1 >= point2) # False
        """
        assert isinstance(other, Point), "other must be of type Point"
        return self.x_ >= other.x_ and self.y_ >= other.y_

    def ClassValidator(self) -> None:
        """
        Validates the class type

        :example:
        >>> point: Point[int] = Point[int](10, 10)
        >>> point.ClassValidator()
        """
        assert isinstance(self.x_, type(self.x_)), f"x_ must be of type {type(self.x_)}"
        assert isinstance(self.y_, type(self.y_)), f"y_ must be of type {type(self.y_)}"

    def OtherValidator(self, other: 'Point') -> None:
        """
        Validates the other type
        
        :example:
        >>> point1: Point[int] = Point[int](10, 10)
        >>> point2: Point[int] = Point[int](20, 20)
        >>> point1.OtherValidator(point2)
        """
        assert isinstance(other, Point), "other must be of type Point"
        assert isinstance(other.x_, type(self.x_)), f"x_ must be of type {type(self.x_)}"
        assert isinstance(other.y_, type(self.y_)), f"y_ must be of type {type(self.y_)}"

class Rect(Generic[T]):
    """
    Rect class for 2D rectangle
    """

    def __init__(self, width: T, height: T, x: T, y: T) -> None:
        """
        Constructor for Rect class
        
        :example:
        >>> rect: Rect[int] = Rect[int](10, 10, 5, 5)
        """
        self.size_: Size[T] = Size[T](width, height)
        self.point_: Point[T] = Point[T](x, y)
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Rect object

        :example:
        >>> rect: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> print(rect) # Rect(size=Size(width=10, height=10), point=Point(x=5, y=5))
        """
        self.ClassValidator()
        return f"Rect(size={self.size_}, point={self.point_})"

    def Add(self, value: T) -> 'Rect':
        """
        Adds a value to the Rect object

        :example:
        >>> rect: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> print(rect.Add(5)) # Rect(size=Size(width=15, height=15), point=Point(x=10, y=10))
        """
        self.ClassValidator()
        new_size = self.size_.Add(value)
        new_point = self.point_.Add(value)
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def Sub(self, value: T) -> 'Rect':
        """
        Subtracts a value from the Rect object

        :example:
        >>> rect: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> print(rect.Sub(5)) # Rect(size=Size(width=5, height=5), point=Point(x=0, y=0))
        """
        self.ClassValidator()
        new_size = self.size_.Sub(value)
        new_point = self.point_.Sub(value)
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def Mul(self, value: T) -> 'Rect':
        """
        Multiplies a value with the Rect object

        :example:
        >>> rect: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> print(rect.Mul(5)) # Rect(size=Size(width=50, height=50), point=Point(x=25, y=25))
        """
        self.ClassValidator()
        new_size = self.size_.Mul(value)
        new_point = self.point_.Mul(value)
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def TrueDiv(self, value: T) -> 'Rect':
        """
        Divides the Rect object by a value

        :example:
        >>> rect: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> print(rect.TrueDiv(5)) # Rect(size=Size(width=2.0, height=2.0), point=Point(x=1.0, y=1.0))
        """
        self.ClassValidator()
        new_size = self.size_.TrueDiv(value)
        new_point = self.point_.TrueDiv(value)
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def FloorDiv(self, value: T) -> 'Rect':
        """
        Divides the Rect object by a value and returns the floor value

        :example:
        >>> rect: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> print(rect.FloorDiv(5)) # Rect(size=Size(width=2, height=2), point=Point(x=1, y=1))
        """
        self.ClassValidator()
        new_size = self.size_.FloorDiv(value)
        new_point = self.point_.FloorDiv(value)
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def Mod(self, value: T) -> 'Rect':
        """
        Returns the modulus of the Rect object and a value

        :example:
        >>> rect: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> print(rect.Mod(5))
        """
        self.ClassValidator()
        new_size = self.size_.Mod(value)
        new_point = self.point_.Mod(value)
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def Pow(self, value: T) -> 'Rect':
        """
        Returns the Rect object raised to the power of a value

        :example:
        >>> rect: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> print(rect.Pow(2)) # Rect(size=Size(width=100, height=100), point=Point(x=25, y=25))
        """
        self.ClassValidator()
        new_size = self.size_.Pow(value)
        new_point = self.point_.Pow(value)
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def __add__(self, other: 'Rect') -> 'Rect':
        """
        Adds two Rect objects together

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](20, 20, 10, 10)
        >>> print(rect1 + rect2) # Rect(size=Size(width=30, height=30), point=Point(x=15, y=15))
        """
        self.OtherValidator(other)
        new_size = self.size_ + other.size_
        new_point = self.point_ + other.point_
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def __sub__(self, other: 'Rect') -> 'Rect':
        """
        Subtracts two Rect objects together

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](20, 20, 10, 10)
        >>> print(rect1 - rect2) # Rect(size=Size(width=-10, height=-10), point=Point(x=-5, y=-5))
        """
        self.OtherValidator(other)
        new_size = self.size_ - other.size_
        new_point = self.point_ - other.point_
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def __mul__(self, other: 'Rect') -> 'Rect':
        """
        Multiplies two Rect objects together

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](20, 20, 10, 10)
        >>> print(rect1 * rect2) # Rect(size=Size(width=200, height=200), point=Point(x=50, y=50))
        """
        self.OtherValidator(other)
        new_size = self.size_ * other.size_
        new_point = self.point_ * other.point_
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def __truediv__(self, other: 'Rect') -> 'Rect':
        """
        Divides two Rect objects together

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](20, 20, 10, 10)
        >>> print(rect1 / rect2) # Rect(size=Size(width=0.5, height=0.5), point=Point(x=0.5, y=0.5))
        """
        self.OtherValidator(other)
        new_size = self.size_ / other.size_
        new_point = self.point_ / other.point_
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def __floordiv__(self, other: 'Rect') -> 'Rect':
        """
        Divides two Rect objects together and returns the floor value

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](20, 20, 10, 10)
        >>> print(rect1 // rect2)
        """
        self.OtherValidator(other)
        new_size = self.size_ // other.size_
        new_point = self.point_ // other.point_
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def __mod__(self, other: 'Rect') -> 'Rect':
        """
        Returns the modulus of two Rect objects

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](20, 20, 10, 10)
        >>> print(rect1 % rect2)
        """
        self.OtherValidator(other)
        new_size = self.size_ % other.size_
        new_point = self.point_ % other.point_
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def __pow__(self, other: 'Rect') -> 'Rect':
        """
        Returns the Rect object raised to the power of another Rect object

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](2, 2, 2, 2)
        >>> print(rect1 ** rect2) # Rect(size=Size(width=100, height=100), point=Point(x=25, y=25))
        """
        self.OtherValidator(other)
        new_size = self.size_ ** other.size_
        new_point = self.point_ ** other.point_
        return Rect(new_size.width_, new_size.height_, new_point.x_, new_point.y_)

    def __eq__(self, other: object) -> bool:
        """
        Checks if two Rect objects are equal

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> print(rect1 == rect2) # True
        """
        assert isinstance(other, Rect), "other must be of type Rect"
        self.OtherValidator(other)
        return self.size_ == other.size_ and self.point_ == other.point_

    def __ne__(self, other: object) -> bool:
        """
        Checks if two Rect objects are not equal

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](20, 20, 10, 10)
        >>> print(rect1 != rect2) # True
        """
        assert isinstance(other, Rect), "other must be of type Rect"
        self.OtherValidator(other)
        return self.size_ != other.size_ or self.point_ != other.point_

    def __lt__(self, other: 'Rect') -> bool:
        """
        Checks if one Rect object is less than another

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](20, 20, 10, 10)
        >>> print(rect1 < rect2) # True
        """
        self.OtherValidator(other)
        return self.size_ < other.size_ and self.point_ < other.point_

    def __le__(self, other: 'Rect') -> bool:
        """
        Checks if one Rect object is less than or equal to another

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](20, 20, 10, 10)
        >>> print(rect1 <= rect2) # True
        """
        self.OtherValidator(other)
        return self.size_ <= other.size_ and self.point_ <= other.point_

    def __gt__(self, other: 'Rect') -> bool:
        """
        Checks if one Rect object is greater than another

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](20, 20, 10, 10)
        >>> print(rect1 > rect2) # False
        """
        self.OtherValidator(other)
        return self.size_ > other.size_ and self.point_ > other.point_

    def __ge__(self, other: 'Rect') -> bool:
        """
        Checks if one Rect object is greater than or equal to another

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](20, 20, 10, 10)
        >>> print(rect1 >= rect2) # False
        """
        assert isinstance(other, Rect), "other must be of type Rect"
        return self.size_ >= other.size_ and self.point_ >= other.point_

    def ClassValidator(self) -> None:
        """
        Validates the class attributes

        :example:
        >>> rect: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect.ClassValidator()
        """
        self.size_.ClassValidator()
        self.point_.ClassValidator()

    def OtherValidator(self, other: 'Rect') -> None:
        """
        Validates the other class attributes

        :example:
        >>> rect1: Rect[int] = Rect[int](10, 10, 5, 5)
        >>> rect2: Rect[int] = Rect[int](20, 20, 10, 10)
        >>> rect1.OtherValidator(rect2)
        """
        assert isinstance(other, Rect), "other must be of type Rect"
        self.size_.OtherValidator(other.size_)
        self.point_.OtherValidator(other.point_)