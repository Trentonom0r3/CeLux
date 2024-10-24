
import logging
import torch
import sys
import os
import cv2


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import celux_cuda as cx

#video_url = r"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
video_pathstr = r"C:\Users\tjerf\source\repos\FrameSmith\Input.mp4"
#video_pathstr = os.path.join(os.getcwd(), "ForBiggerBlazes.mp4")

cx.set_log_level(cx.LogLevel.debug)
 
class test:
    def __init__(self, video_path):
        self.video_path = video_path
        
    def run(self):
        try:
            frameCount = 0
            video = cv2.VideoCapture( self.video_path)
            totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            print("CV2 Metadata:")
            print(f"Total Frames: {totalFrames}")
            with cx.VideoReader(
                self.video_path , d_type="uint8", device="cuda", stream=torch.cuda.Stream()
            ) as reader:
                print(reader.get_properties())
                input("Press Enter to continue.")
                for frame in reader:
                    
                    #me(frame)
                    frameCount += 1
                   # print(frameCount)
            print("Frames: ", frameCount)
        except StopIteration:
            print("End of video")
        except Exception as e:
            logging.error(f"Error processing video: {e}")
            raise
        
if __name__ == "__main__":
    test(video_pathstr).run()
    