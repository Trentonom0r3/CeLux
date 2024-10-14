from typing import List, Dict, Optional, Any, Union, TypedDict, Tuple
import torch
import numpy

class VideoProperties(TypedDict):
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int
    pixel_format: str
    has_audio: bool

class VideoReader:
    def __init__(self, input_path: str, as_numpy: bool = False, d_type: str = "uint8") -> None:
        """
        Initialize the VideoReader object.

        Args:
            input_path (str): Path to the video file.
            as_numpy (bool): Whether to return frames as NumPy arrays. Default is False.
            d_type (str): Data type of the frames (e.g., "uint8"). Default is "uint8".
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

    def read_frame(self) -> Union['torch.Tensor', 'numpy.ndarray']:
        """
        Read a frame from the video.

        Returns:
            Union[torch.Tensor, numpy.ndarray]: The frame data, either as a torch.Tensor or numpy.ndarray.
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

    def get_properties(self) -> VideoProperties:
        """
        Get properties of the video.

        Returns:
            VideoProperties: A dictionary containing specific video properties.
            Contains the following:
            - width: Width of the video.
            - height: Height of the video.
            - fps: Frames per second of the video.
            - duration: Duration of the video in seconds.
            - total_frames: Total number of frames in the video.
            - pixel_format: Pixel format of the video.
            - has_audio: Whether the video has an audio stream.
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

    def __next__(self) -> Union['torch.Tensor', 'numpy.ndarray']:
        """
        Get the next frame in the video.

        Returns:
            Union[torch.Tensor, numpy.ndarray]: The next frame as a torch.Tensor or numpy.ndarray.
        
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

    def sync(self) -> None:
        """
        Synchronize the video reader (if required for hardware operations).
        """
        ...

    def reset(self) -> None:
        """
        Reset the video reader to the beginning of the video.
        """
        ...

class VideoWriter:
    def __init__(self, file_path: str, width: int, height: int, fps: float, 
                 as_numpy: bool = False, dtype: str = "uint8") -> None:
        """
        Initialize the VideoWriter object.

        Args:
            file_path (str): Path to the output video file.
            width (int): Width of the video frames.
            height (int): Height of the video frames.
            fps (float): Frames per second for the output video.
            as_numpy (bool): Whether to expect frames as NumPy arrays. Default is False.
            dtype (str): Data type of the frames (e.g., "uint8"). Default is "uint8".
        """
        ...

    def write_frame(self, frame: Union['torch.Tensor', 'numpy.ndarray']) -> bool:
        """
        Write a frame to the video.

        Args:
            frame (Union[torch.Tensor, numpy.ndarray]): The frame data to write.

        Returns:
            bool: True if the frame was written successfully, otherwise False.

        Raises:
            ValueError: If the frame dimensions or data type do not match the writer's configuration.
        """
        ...

    def supported_codecs(self) -> List[str]:
        """
        Get a list of supported video codecs for writing.

        Returns:
            List[str]: List of supported codec names.
        """
        ...

    def __call__(self, frame: Union['torch.Tensor', 'numpy.ndarray']) -> bool:
        """
        Write a frame to the video.

        This method allows you to write frames using a function call syntax.

        Example:
            writer = VideoWriter('output.mp4', width=1920, height=1080, fps=30)
            writer(frame)

        Args:
            frame (Union[torch.Tensor, numpy.ndarray]): The frame data to write.

        Returns:
            bool: True if the frame was written successfully, otherwise False.
        """
        ...

    def __enter__(self) -> 'VideoWriter':
        """
        Enter the context manager.

        Returns:
            VideoWriter: The video writer object itself.
        """
        ...

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[Any]) -> bool:
        """
        Exit the context manager, finalizing and closing the video file.

        Ensures that all resources are properly released and the video file is finalized.

        Args:
            exc_type (Optional[type]): The exception type, if any.
            exc_value (Optional[BaseException]): The exception value, if any.
            traceback (Optional[Any]): The traceback object, if any.

        Returns:
            bool: False to propagate exceptions, True to suppress them.
        """
        ...

    def close(self) -> None:
        """
        Close the video writer and finalize the output video file.

        This method is called automatically when exiting a context manager, but can be called manually if needed.
        """
        ...
