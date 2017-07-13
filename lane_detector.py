import numpy as np
from pipeline_vars import PipelineVars
from global_vars import GlobalVars as GV

class LaneDetector(object):


    @staticmethod
    def find_lane_points(pv):
        """
        :type pv:PipelineVars
        """
        #Positions of all white pixels
        nz_points = pv.img_warped_binary.nonzero()

        if LaneDetector.fit_lane_points(nz_points):
            return

        #Get the base X position of right and left lanes
        histogram = np.sum(pv.img_warped_binary[pv.img_warped_binary.shape[0]//2:,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        left_base_x = np.argmax( histogram[100:midpoint] ) + 100
        right_base_x = np.argmax(histogram[midpoint:-100]) + midpoint

        #Get the positions of all white pixels in left lane and right lane
        pv.left_window_positions,left_points_x,left_points_y = LaneDetector.sliding_window(nz_points,left_base_x)
        pv.right_window_positions,right_points_x,right_points_y = LaneDetector.sliding_window(nz_points,right_base_x)

        GV.global_left_points_x.append(left_points_x)
        GV.global_left_points_y.append(left_points_y)
        GV.global_right_points_x.append(right_points_x)
        GV.global_right_points_y.append(right_points_y)


    @staticmethod
    def fit_lane_points(nz_points):

        if GV.global_left_poly is None:
            return False

        margin = 100
        minpoints = 10
        Y=np.array(nz_points[0])
        X=np.array(nz_points[1])

        poly = GV.global_left_poly
        left_fit_x = poly[0]*(Y**2) + poly[1]*Y + poly[2]
        poly = GV.global_right_poly
        right_fit_x = poly[0]*(Y**2) + poly[1]*Y + poly[2]

        left_indices = (X > (left_fit_x - margin)) & (X < (left_fit_x + margin))
        right_indices = (X > (right_fit_x - margin)) & (X < (right_fit_x + margin))

        if( len(left_indices) < minpoints or len(right_indices) < minpoints):
            return False

        if(len(X[left_indices]) > 5):
            GV.global_left_points_x.append(X[left_indices])
            GV.global_left_points_y.append(Y[left_indices])
            GV.global_left_poly = np.polyfit(X[left_indices],Y[left_indices],2)

        if(len(X[right_indices]) > 5):
            GV.global_right_points_x.append(X[right_indices])
            GV.global_right_points_y.append(Y[right_indices])
            GV.global_right_poly = np.polyfit(X[right_indices],Y[right_indices],2)

        return True



    @staticmethod
    def generate_plot_points(pv):
        """
        @type pv: PipelineVars
        """
        image_height = pv.img_undistorted.shape[0]
        pv.left_points_x = np.concatenate (GV.global_left_points_x)
        pv.left_points_y = np.concatenate (GV.global_left_points_y)
        pv.right_points_x = np.concatenate(GV.global_right_points_x)
        pv.right_points_y = np.concatenate(GV.global_right_points_y)

        ploty = np.linspace(0, image_height-1, image_height )

        pv.left_poly  = np.polyfit(pv.left_points_y, pv.left_points_x, 2)
        pv.right_poly  = np.polyfit(pv.right_points_y, pv.right_points_x, 2)

        pv.left_plot_x = pv.left_poly[0]*ploty**2 + pv.left_poly[1]*ploty + pv.left_poly[2]
        pv.left_plot_y = ploty

        pv.right_plot_x = pv.right_poly[0]*ploty**2 + pv.right_poly[1]*ploty + pv.right_poly[2]
        pv.right_plot_y = ploty

        GV.global_left_poly=pv.left_poly
        GV.global_right_poly=pv.right_poly

    @staticmethod
    def sliding_window(nz_points,base_x):

        image_height = 720
        nwindows = 9
        window_height = image_height//nwindows
        margin = 100
        minpix = 50
        Y=np.array(nz_points[0])
        X=np.array(nz_points[1])
        points_x=[]
        points_y=[]
        window_positions=[]


        for window in range(nwindows):

            x1 = base_x - margin
            x2 = base_x + margin
            y1 = image_height - (window+1)*window_height
            y2 = image_height - window*window_height

            indices = ((Y >= y1) & (Y < y2) & (X >= x1) & (X < x2)).nonzero()[0]

            points_x.append(X[indices])
            points_y.append(Y[indices])
            window_positions.append([x1,y1,x2,y2])

            if len(indices) > minpix:
                base_x = np.int(np.mean(X[indices]))

        points_x = np.concatenate(points_x)
        points_y = np.concatenate(points_y)


        return window_positions,points_x , points_y


    @staticmethod
    def calculate_curve(pv):
        """
        @type pv: PipelineVars
        """
        y_eval = 719  # 720p video/image, so last (lowest on screen) y index is 719
        y_meters_per_pix = 30/720 # meters per pixel in y dimension
        x_meters_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space

        left_poly  = np.polyfit(pv.left_points_y *y_meters_per_pix, pv.left_points_x*x_meters_per_pix, 2)
        right_poly = np.polyfit(pv.right_points_y*y_meters_per_pix, pv.right_points_x*x_meters_per_pix, 2)

        left_curve_radius = ((1 + (2*left_poly[0]*y_eval*y_meters_per_pix + left_poly[1])**2)**1.5) / np.absolute(2*left_poly[0])
        right_curve_radius = ((1 + (2*right_poly[0]*y_eval*y_meters_per_pix + right_poly[1])**2)**1.5) / np.absolute(2*right_poly[0])

        pv.curve_radius = (left_curve_radius + right_curve_radius)/2



    @staticmethod
    def calculate_vehicle_offset(pv):
        """
        @type pv:PipelineVars
        """

        # Calculate vehicle center offset in pixels
        bottom_y = 719
        bottom_x_left = pv.left_poly[0]*(bottom_y**2) + pv.left_poly[1]*bottom_y + pv.left_poly[2]
        bottom_x_right = pv.right_poly[0]*(bottom_y**2) + pv.right_poly[1]*bottom_y + pv.right_poly[2]

        pv.vehicle_offset = pv.img_undistorted.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

        # Convert pixel offset to meters
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        pv.vehicle_offset *= xm_per_pix

