from matplotlib import image as mpimg, pyplot as plt
from pipeline import pipeline
from plotter import plot_images
from pipeline_vars import PipelineVars
from camera import Camera
import numpy as np

img = mpimg.imread('test_images/test2.jpg')
camera = Camera.get_calibrated_camera()
pv = pipeline(img,camera,debug=True)  #type: PipelineVars

images = [ [pv.image, "Original Image" ],
           [pv.img_schannel_binary, "S-Channel",'gray'],
           [pv.img_sobel_binary, "Absolute Sobel",'gray'],
           [pv.img_sobel_magnitude, "Sobel Magnitude",'gray'],
           [pv.img_sobel_direction, "Sobel Direction",'gray'],
           [pv.img_combined_binary, "Combined",'gray'],
           [pv.img_warped_binary, "Warped",'gray'],
           [pv.img_sliding_window, "Sliding Window"],
           [pv.img_processed, "With Lanes"]
           ]

plot_images(images,3,3,"output_images/pipeline_test2_output.png")
plt.show()
