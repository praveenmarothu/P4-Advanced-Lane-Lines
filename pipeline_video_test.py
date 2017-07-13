from pipeline import pipeline
from pipeline_vars import PipelineVars
from camera import Camera
from moviepy.editor import VideoFileClip

def process_frame(frame):
    camera = Camera.get_calibrated_camera()
    pv = pipeline(frame,camera)  #type: PipelineVars
    return pv.img_processed

camera = Camera.get_calibrated_camera()
video = VideoFileClip("project_video.mp4")
annotated_video = video.fl_image(process_frame)
annotated_video.write_videofile("out.mp4", audio=False)

print("Processed to : out.mp4")