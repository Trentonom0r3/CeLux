# ffmpy.pyi

from typing import Any, Optional, List, Dict, Union
import numpy as np
import torch

class Frame:
    def getData(self, plane: int) -> bytes:
        """
        Get data for a specific plane.

        :param plane: Plane index (e.g., 0 for Y, 1 for UV).
        :return: Raw bytes of the plane data.
        """
        ...

    def getLineSize(self, plane: int) -> int:
        """
        Get the line size for a specific plane.

        :param plane: Plane index.
        :return: Line size in bytes.
        """
        ...

    def getWidth(self) -> int:
        """
        Get the width of the frame.

        :return: Width in pixels.
        """
        ...

    def getHeight(self) -> int:
        """
        Get the height of the frame.

        :return: Height in pixels.
        """
        ...

class VideoReader:
    def __init__(
        self, 
        filePath: str, 
        useHardware: bool = True, 
        hwType: str = "cuda", 
        as_numpy: bool = False
    ) -> None:
        """
        Initialize a VideoReader instance.

        :param filePath: Path to the video file.
        :param useHardware: Whether to use hardware acceleration.
        :param hwType: Type of hardware acceleration (e.g., "cuda").
        :param as_numpy: Whether to return frames as NumPy arrays.
        """
        ...

    def readFrame(self) -> Union[torch.Tensor, np.ndarray, None]:
        """
        Read and return the next video frame.

        :return: A video frame as a Frame object, a PyTorch tensor, a NumPy array, or None if no frame is read.
        """
        ...

    def seek(self, frame_number: int) -> None:
        """
        Seek to a specific frame number in the video.

        :param frame_number: The frame number to seek to.
        """
        ...

    def supportedCodecs(self) -> List[str]:
        """
        Get a list of supported codecs.

        :return: A list of codec names.
        """
        ...
    
    def getProperties(self) -> Dict[str, Union[int, float]]:
        """
        Get properties of the video.

        :return: A dictionary containing video properties (e.g., width, height, fps, duration, totalframes, pixelFormat).
        """
        ...

    def __len__(self) -> int:
        """
        Get the total number of frames in the video.

        :return: Total frame count.
        """
        ...

    def __iter__(self) -> 'VideoReader':
        """
        Return the iterator object itself.

        :return: The VideoReader instance.
        """
        ...

    def __next__(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Return the next frame in the video.

        :return: The next video frame as a Frame object, a PyTorch tensor, or a NumPy array.
        :raises StopIteration: When there are no more frames.
        """
        ...

    def __enter__(self) -> 'VideoReader':
        """
        Enter the runtime context related to the VideoReader object.

        :return: The VideoReader instance.
        """
        ...

    def __exit__(
        self, 
        exc_type: Optional[type], 
        exc_value: Optional[BaseException], 
        traceback: Optional[Any]
    ) -> bool:
        """
        Exit the runtime context and handle exceptions.

        :param exc_type: Exception type.
        :param exc_value: Exception value.
        :param traceback: Traceback object.
        :return: False to indicate that exceptions should not be suppressed.
        """
        ...
