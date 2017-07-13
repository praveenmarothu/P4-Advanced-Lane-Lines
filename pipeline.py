from image_processor import ImageProcessor as IP
from lane_detector import LaneDetector as LD
from pipeline_vars import PipelineVars

def pipeline(image,camera,debug=False):

    pv= PipelineVars(image,camera.camera_matrix,camera.dist_coeff)

    IP.undistort(pv)
    IP.generate_colormap_images(pv)
    IP.process_schannel(pv)
    IP.process_sobel_absolute(pv)
    IP.process_sobel_direction(pv)
    IP.process_sobel_magnitude(pv)
    IP.process_combined(pv)
    IP.perspective_transform(pv)

    LD.find_lane_points(pv)
    LD.generate_plot_points(pv)
    LD.calculate_curve(pv)
    LD.calculate_vehicle_offset(pv)

    IP.draw_overlay(pv)

    if debug :
        IP.draw_sliding_windows(pv)


    return pv