import collections

class PipelineVars(object):


    def __init__(self,img,camera_matrix,dist_coeff):

        self.image = img
        self.camera_matrix = camera_matrix
        self.dist_coeff = dist_coeff

        self.img_gray = None
        self.img_hls = None

        #Processed by Image Processor
        self.img_undistorted = None
        self.img_sobel_binary = None
        self.img_sobel_magnitude = None
        self.img_sobel_direction = None
        self.img_schannel_binary = None
        self.img_combined_binary = None
        self.img_warped_binary = None

        self.transform_matrix = None
        self.inverse_matrix = None

        self.img_sliding_window = None
        self.color_warp = None
        self.color_unwarped = None
        self.img_processed = None

        #Processed by Lane Detector
        self.left_points_x = None
        self.left_points_y = None
        self.left_window_positions = None
        self.right_points_x = None
        self.right_points_y = None
        self.right_window_positions = None

        self.left_poly = None
        self.left_plot_x = None
        self.left_plot_y = None
        self.right_poly = None
        self.right_plot_x = None
        self.right_plot_y = None

        self.curve_radius = None
        self.vehicle_offset = None


