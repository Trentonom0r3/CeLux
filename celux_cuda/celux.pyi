from typing import List, Optional, Any, Union, Tuple
import torch
from enum import Enum
        
class LogLevel(Enum):
    trace = 0
    debug = 1
    info = 2
    warn = 3
    error = 4
    critical = 5
    off = 6
    
def set_log_level(level: LogLevel) -> None:
    """
    Set the logging level for CeLux.

    Args:
        level (LogLevel): The logging level to set.
    """
    ...

class VideoReader:
    def __init__(self, input_path: str, device: str = "cuda",
stream : torch.Stream = None) -> None:
        """
        Initialize the VideoReader object.

        Args:
            input_path (str): Path to the video file.
            device (str): Device to be used. Default is "cuda".
            stream (torch.Stream): CUDA stream to use for reading frames. Default is None.
        """
        ...

    def __call__(self, frame_range: Union[Tuple[int, int], List[int]]) -> 'VideoReader':
        """
        Set the frame range for the video reader.

        Allows you to specify a range of frames to read from the video.

        Example:
            with VideoReader('input.mp4')([10, 20]) as reader:
                for frame in reader:
                    print(f"Processing {frame}")

        Args:
            frame_range (Union[Tuple[int, int], List[int]]): A tuple or list containing the start and end frame indices.

        Returns:
            VideoReader: The video reader object itself.

        Raises:
            ValueError: If `frame_range` is not a tuple or list of two integers.
        """
        ...

    def read_frame(self) -> torch.Tensor:
        """
        Read a frame from the video.

        Returns:
            Union[torch.Tensor: The frame data, as a torch.Tensor.
        """
        ...

    def seek(self, timestamp: float) -> bool:
        """
        Seek to a specific timestamp in the video.

        Args:
            timestamp (float): Timestamp in seconds.

        Returns:
            bool: True if seek was successful, otherwise False.
        """
        ...

    def supported_codecs(self) -> List[str]:
        """
        Get a list of supported video codecs.

        Returns:
            List[str]: List of supported codec names.
        """
        ...

    def get_properties(self) -> dict:
        """
        Get properties of the video.

        Returns:
            A dictionary containing specific video properties.
            Contains the following:
            - width: Width of the video.
            - height: Height of the video.
            - fps: Frames per second of the video.
            - duration: Duration of the video in seconds.
            - total_frames: Total number of frames in the video.
            - codec: Codec used for the video.
            - pixel_format: Pixel format of the video.
            - bit_depth: Bit depth of the video.
        """
        ...

    def __len__(self) -> int:
        """
        Get the total number of frames in the video.

        Returns:
            int: Number of frames.
        """
        ...

    def __iter__(self) -> 'VideoReader':
        """
        Get the iterator object for the video reader.

        Returns:
            VideoReader: The video reader object itself.
        """
        ...

    def __next__(self) -> torch.Tensor:
        """
        Get the next frame in the video.

        Returns:
          torch.Tensor: The next frame as a torch.Tensor.
        
        Raises:
            StopIteration: When no more frames are available.
        """
        ...

    def __enter__(self) -> 'VideoReader':
        """
        Enter the context manager.

        Returns:
            VideoReader: The video reader object itself.
        """
        ...

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[Any]) -> bool:
        """
        Exit the context manager.

        Args:
            exc_type (Optional[type]): The exception type, if any.
            exc_value (Optional[BaseException]): The exception value, if any.
            traceback (Optional[Any]): The traceback object, if any.

        Returns:
            bool: False to propagate exceptions, True to suppress them.
        """
        ...


    def reset(self) -> None:
        """
        Reset the video reader to the beginning of the video.
        """
        ...
