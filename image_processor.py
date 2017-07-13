import numpy as np
import cv2
from pipeline_vars import PipelineVars


class ImageProcessor(object):

    @staticmethod
    def undistort(pv):
        """
        :type pv:PipelineVars
        """
        pv.img_undistorted = cv2.undistort(pv.image, pv.camera_matrix, pv.dist_coeff, None, pv.camera_matrix)

    @staticmethod
    def generate_colormap_images(pv):
        """
        :type pv:PipelineVars
        """
        pv.img_gray = cv2.cvtColor(pv.img_undistorted,cv2.COLOR_RGB2GRAY)
        pv.img_hls = cv2.cvtColor(pv.img_undistorted,cv2.COLOR_RGB2HLS)

    @staticmethod
    def process_sobel_absolute(pv):
        """
        :type pv:PipelineVars
        """
        abs_thresh = (50, 255)
        # Absolute Sobel Binary
        abs_sobel = np.absolute(cv2.Sobel(pv.img_gray, cv2.CV_64F, 1, 0))
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= abs_thresh[0]) & (scaled_sobel <= abs_thresh[1])] = 1
        pv.img_sobel_binary = sobel_binary

    @staticmethod
    def process_sobel_magnitude(pv):
        """
        :type pv:PipelineVars
        """
        mag_kernel = 3
        mag_thresh = (50, 255)

        sobelx = cv2.Sobel(pv.img_gray, cv2.CV_64F, 1, 0, ksize=mag_kernel)
        sobely = cv2.Sobel(pv.img_gray, cv2.CV_64F, 0, 1, ksize=mag_kernel)
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        sobel_magnitude = np.zeros_like(gradmag)
        sobel_magnitude[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        pv.img_sobel_magnitude = sobel_magnitude

    @staticmethod
    def process_sobel_direction(pv):
        """
        :type pv:PipelineVars
        """
        dir_kernel = 15
        dir_thresh = (0.7, 1.3)

        sobelx = cv2.Sobel(pv.img_gray, cv2.CV_64F, 1, 0, ksize=dir_kernel)
        sobely = cv2.Sobel(pv.img_gray, cv2.CV_64F, 0, 1, ksize=dir_kernel)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        sobel_direction = np.zeros_like(absgraddir)
        sobel_direction[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1
        pv.img_sobel_direction = sobel_direction

    @staticmethod
    def process_schannel(pv):
        """
        :type pv:PipelineVars
        """
        thresh = (170, 255)
        s_channel = pv.img_hls[:, :, 2]
        pv.img_schannel_binary = np.zeros_like(s_channel)
        pv.img_schannel_binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    def process_combined(pv):
        """
        @type pv:PipelineVars
        """
        pv.img_combined_binary = np.zeros_like(pv.img_sobel_binary)
        pv.img_combined_binary[( pv.img_sobel_binary == 1 | ((pv.img_sobel_magnitude == 1) & (pv.img_sobel_direction == 1))) | pv.img_schannel_binary == 1] = 1

    @staticmethod
    def perspective_transform(pv):
        """
        :type pv:PipelineVars
        """
        img_size = (pv.img_combined_binary.shape[1], pv.img_combined_binary.shape[0])
        src = np.float32([[200, 720],[1100, 720],[595, 450],[685, 450]])
        dst = np.float32([[300, 720],[980 , 720],[300, 0  ],[980, 0  ]])

        pv.transform_matrix= cv2.getPerspectiveTransform(src, dst)
        pv.inverse_matrix = cv2.getPerspectiveTransform(dst, src)
        pv.img_warped_binary = cv2.warpPerspective(pv.img_combined_binary, pv.transform_matrix, img_size, flags=cv2.INTER_LINEAR)



    @staticmethod
    def draw_overlay(pv):
        """
        @type pv: PipelineVars
        """
        pv.color_warp = np.zeros_like(pv.img_undistorted, dtype='uint8')  # NOTE: Hard-coded image dimensions

        pts_left = np.array([np.transpose(np.vstack([pv.left_plot_x,pv.left_plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([pv.right_plot_x, pv.right_plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(pv.color_warp, np.int_([pts]), (0,255, 0))

        pv.color_unwarped = cv2.warpPerspective(pv.color_warp, pv.inverse_matrix, (pv.img_undistorted.shape[1], pv.img_undistorted.shape[0]))
        pv.img_processed = cv2.addWeighted(pv.img_undistorted, 1, pv.color_unwarped, 0.3, 0)

        label_str = 'Radius of curvature: %.1f m' % pv.curve_radius
        cv2.putText(pv.img_processed, label_str, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)

        label_str = 'Vehicle offset from lane center: %.1f m' % pv.vehicle_offset
        cv2.putText(pv.img_processed, label_str, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)

    @staticmethod
    def draw_sliding_windows(pv):
        """
        :type pv:PipelineVars
        """
        #Draw Sliding windows for left and right path
        img = (np.dstack((pv.img_warped_binary, pv.img_warped_binary, pv.img_warped_binary))*255).astype('uint8')
        img = ImageProcessor.draw_lane_windows(img,pv.left_window_positions,pv.left_points_x,pv.left_points_y,pv.left_plot_x,pv.left_plot_y,[255,0,0])
        img = ImageProcessor.draw_lane_windows(img,pv.right_window_positions,pv.right_points_x,pv.right_points_y,pv.right_plot_x,pv.right_plot_y,[255,0,0])

        pv.img_sliding_window = img


    @staticmethod
    def draw_lane_windows(img,window_positions,points_x,points_y,plot_x,plot_y,color):

        img[points_y,points_x] = [255,0,0]

        for wp in window_positions:
            cv2.rectangle(img,(wp[0],wp[1]),(wp[2],wp[3]),(0,255,0), 3)

        poly_pts=np.dstack( (plot_x,plot_y) ).astype(np.int32)
        cv2.polylines(img,poly_pts,False,(255,255,0),4)

        img_overlay = np.zeros_like(img)
        margin = 100
        path_pts1 = np.array([np.transpose(np.vstack([plot_x-margin, plot_y]))])
        path_pts2 = np.array([np.flipud(np.transpose(np.vstack([plot_x+margin, plot_y])))])
        path_pts =  np.hstack((path_pts1, path_pts2)).astype(np.int32)

        cv2.fillPoly(img_overlay, path_pts, (0,255, 0))
        img = cv2.addWeighted(img, 1, img_overlay, 0.3, 0)

        return img