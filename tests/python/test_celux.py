import unittest
import celux
import sys


class TestVideoReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Extract the video path from the command-line arguments once for all tests."""
        # Ensure the video path is provided as an argument
        if not hasattr(cls, "video_path"):
            print("Usage: python test_ffmpy.py <video_path>")
            sys.exit(1)

        print(f"Using video path: {cls.video_path}")

    def setUp(self):
        """Set up a VideoReader instance before each test."""
        self.reader = celux.VideoReader(self.video_path, as_numpy=True, d_type="uint8")

    def test_initialization(self):
        """Test that the VideoReader initializes correctly."""
        self.assertIsInstance(self.reader, celux.VideoReader)

    def test_get_properties(self):
        """Test that get_properties returns the expected dictionary structure."""
        properties = self.reader.get_properties()
        self.assertIsInstance(properties, dict)
        self.assertIn("width", properties)
        self.assertIn("height", properties)
        self.assertIn("fps", properties)
        self.assertIn("duration", properties)
        self.assertIn("total_frames", properties)
        self.assertIn("pixel_format", properties)

    def test_seek(self):
        """Test seeking to a valid timestamp."""
        self.assertTrue(self.reader.seek(1.5))

    def test_seek_invalid(self):
        """Test seeking to an invalid timestamp."""
        self.assertFalse(self.reader.seek(-10))

    def test_iteration(self):
        """Test iteration through frames."""
        count = 0
        for frame in self.reader:
            self.assertIsNotNone(frame)
            count += 1
            if count >= 5:
                break

    def test_context_manager(self):
        """Test using VideoReader with a context manager."""
        with celux.VideoReader(self.video_path) as reader:
            properties = reader.get_properties()
            self.assertIn("width", properties)

    def test_next_frame(self):
        """Test getting the next frame."""
        self.assertIsNotNone(next(self.reader))

    def test_len(self):
        """Test the length of the video."""
        self.assertGreater(len(self.reader), 0)

    def tearDown(self):
        """Clean up after each test."""
        self.reader = None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ffmpy.py <video_path>")
        sys.exit(1)

    # Extract the video path and set it as a class attribute
    TestVideoReader.video_path = sys.argv.pop(1)

    # Run the tests
    unittest.main(argv=sys.argv)
