"""This is .py file for performing road lane detection"""

import numpy as np
import cv2

# function for rgb channel color threshold
def rgb_select(img, r_thresh, g_thresh, b_thresh):
    r_channel = img[:,:,0]
    g_channel=img[:,:,1]
    b_channel = img[:,:,2]
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel > r_thresh[0]) & (r_channel <= r_thresh[1])] = 1
    
    g_binary = np.zeros_like(g_channel)
    g_binary[(r_channel > g_thresh[0]) & (r_channel <= g_thresh[1])] = 1
    
    b_binary = np.zeros_like(b_channel)
    b_binary[(r_channel > b_thresh[0]) & (r_channel <= b_thresh[1])] = 1
    
    combined = np.zeros_like(r_channel)
    combined[((r_binary == 1) & (g_binary == 1) & (b_binary == 1))] = 1
    return combined

# function for directional gradient threshold
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

# function for sblv channels threshold
def color_thresh(image, s_thresh, l_thresh, b_thresh, v_thresh):
    luv= cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    lab=cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    s_channel = hsv[:,:,1]
    b_channel=lab[:,:,2]
    l_channel = luv[:,:,0]
    v_channel= hsv[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    b_binary = np.zeros_like(b_channel)
    b_binary[(s_channel > b_thresh[0]) & (s_channel <= b_thresh[1])] = 1
    l_binary = np.zeros_like(l_channel)
    l_binary[(s_channel > l_thresh[0]) & (s_channel <= l_thresh[1])] = 1
    v_binary = np.zeros_like(v_channel)
    v_binary[(s_channel > v_thresh[0]) & (s_channel <= v_thresh[1])] = 1
    combined = np.zeros_like(s_channel)
    combined[((s_binary == 1) & (b_binary == 1) & (l_binary == 1) & (v_binary == 1))] = 1
    
    return combined

# function combining all thresholds
def color_gradient_threshold(image_undistorted):
    ksize = 15
    hsv = cv2.cvtColor(image_undistorted,cv2.COLOR_RGB2HSV)
    s_channel = hsv[:,:,1]
    
    gradx=abs_sobel_thresh(image_undistorted,orient='x',sobel_kernel=ksize,thresh=(50,80))
    grady=abs_sobel_thresh(image_undistorted,orient='y',sobel_kernel=ksize,thresh=(50,90))
    c_binary=color_thresh(image_undistorted,s_thresh=(70,100),l_thresh=(60,255),b_thresh=(50,255),v_thresh=(150,255))
    rgb_binary=rgb_select(image_undistorted,r_thresh=(225,255),g_thresh=(225,255),b_thresh=(0,255))
    combined_binary = np.zeros_like(s_channel)
    combined_binary[((gradx == 1) | (grady == 1) | (c_binary == 1) | (rgb_binary==1))] = 255
    color_binary = combined_binary
    return color_binary, combined_binary

# function for perspective transform
def perspective_transform(masked_image):
    top_left = [905, 884]
    top_right = [1215, 884]
    bottom_right = [1500, 1080]
    bottom_left = [500, 1080]
    top_left_dst = [500,0]
    top_right_dst = [1500,0]
    bottom_right_dst = [1500,1080]
    bottom_left_dst = [500,1080]

    img_size = (masked_image.shape[1], masked_image.shape[0])
    src = np.float32([top_left,top_right, bottom_right, bottom_left] )
    dst = np.float32([top_left_dst, top_right_dst, bottom_right_dst, bottom_left_dst])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped  = cv2.warpPerspective(masked_image, M, img_size)
    return warped, Minv, M

# function for lane detection
def finding_line(warped):
    histogram2 = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((warped, warped, warped))*255
    midpoint = np.int(histogram2.shape[0]/2)

    leftx_base = np.argmax(histogram2[:midpoint])
    rightx_base = np.argmax(histogram2[midpoint:])+midpoint
    nwindows = 3
    window_height = np.int(warped.shape[0]/nwindows)
    nonzero = warped.nonzero()   
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 40
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        win_y_low = warped.shape[0]-(window+1)*window_height
        win_y_high = warped.shape[0]-window*window_height
        win_xleft_low = leftx_current-margin
        win_xleft_high = leftx_current+margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
  
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    print(left_lane_inds)
    print(right_lane_inds)

    return left_fitx, right_fitx,out_img, left_fit, right_fit,left_lane_inds,right_lane_inds,ploty

def sliding_window(binary_warped,left_fit,right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))  
    return left_fitx, right_fitx, left_line_pts,right_line_pts, window_img, out_img,left_lane_inds, right_lane_inds, ploty

def region_of_interest(color_binary, vertices):
    mask = np.zeros_like(color_binary)   
    
    if len(color_binary.shape) > 2:
        channel_count = color_binary.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
           
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(color_binary, mask)
    return masked_image

def apply_region_of_interest_mask(color_binary):
    x_factor = 200
    y_factor = 330
    vertices = np.array([[
                        (480,color_binary.shape[0]),
                    (((color_binary.shape[1]/2)- x_factor+180), (color_binary.shape[0]/2)+ y_factor), 
                     (((color_binary.shape[1]/2) + x_factor+50), (color_binary.shape[0]/2)+ y_factor), 
                     (color_binary.shape[1]-400,color_binary.shape[0])]], dtype=np.int32)

    return region_of_interest(color_binary, vertices)

def main_pipline(image, location):
    
    #1 color and gradient detection
    color_binary, combined_binary = color_gradient_threshold(image)

    #3 interest region (with road lane in)
    masked = apply_region_of_interest_mask(color_binary)

    #4 perspective transformation
    warped_0, Minv, M = perspective_transform(masked)
    
    #5 sliding windows road lane detection
    left_fitx, right_fitx, out_img,left_fit, right_fit,left_lane_inds,right_lane_inds, ploty = finding_line(warped_0)
    
    #6 draw road lane on image
    warp_zero = np.zeros_like(warped_0).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    location = np.concatenate(location)
    location = np.array([location])
    location_warped = cv2.perspectiveTransform(location, M)
    count = 0
    for i in range(location_warped.shape[1]):
        if (int(location_warped[0][i][1]) > len(pts[0]) or (len(pts[0])-(int(location_warped[0][i][1])+1) > len(pts[0]))):
            continue   
        elif (location_warped[0][i][0] > (pts[0][int(location_warped[0][i][1])][0] - 550)) and (location_warped[0][i][0] < (pts[0][len(pts[0])-(int(location_warped[0][i][1])+1)][0] + 550)):
            count += 1
        else:
            continue
    if count != 0:
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))
    else:
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result
