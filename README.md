## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted/undistorted.png "Undistorted"
[image2]: ./output_images/undistorted/test2.jpg "Road Transformed"
[color_original]: ./output_images/color_spaces/01-original.png "Original"
[color_rgb]: ./output_images/color_spaces/02-rgb.png "RGB"
[color_hsv]: ./output_images/color_spaces/03-hsv.png "HSV"
[color_hls]: ./output_images/color_spaces/04-hls.png "HLS"
[color_yuv]: ./output_images/color_spaces/05-yuv.png "YUV"
[color_luv]: ./output_images/color_spaces/06-luv.png "LUV"
[color_lab]: ./output_images/color_spaces/07-lab.png "LAB"
[color_xyz]: ./output_images/color_spaces/08-xyz.png "XYZ"
[color_yellow]: ./output_images/color_spaces/09-yellow-options.png "Yellow Options"
[color_white]: ./output_images/color_spaces/10-white-options.png "White Options"
[color_threshold]: ./output_images/color_spaces/11-combined-threshold.png "Threshold"
[bev]: ./output_images/perspective_transform/01-birds-eye-view.png "Birds Eye View"
[polyfit1]: ./output_images/polyfit/01-sliding-window.png/ "Sliding Window"
[polyfit2]: ./output_images/polyfit/02-polyfit-tracking.png/ "Polyfit Tracking"
[draw_lane]: ./output_images/perspective_transform/03-invert-back.png "Draw Lane"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./project_submission.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I applied cv2.undistort() to all images after getting the matrix for camera distortion correction. Below is the example when I apply the distortion correction to one of the test images:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

First, I explored all possible color spaces to all images to isolate yellow lines and white lines. I took test5.jpg image as example because it has yellow lines, white lines, and different lighting conditions. Below are the original image and different color spaces.

![alt text][color_original]

RGB
![alt text][color_rgb]

HSV
![alt text][color_HSV]

HLS
![alt text][color_hls]

YUV
![alt text][color_yuv]

LUV
![alt text][color_luv]

LAB
![alt text][color_lab]

XYZ
![alt text][color_xyz]

Based on those images, I decided to compare several options for yellow and white lines detection.

Yellow Options
![alt text][color_yellow]

White Options
![alt text][color_white]

I used B channel from LAB color space to isolate yellow line and S channel from HLS to isolate white color. After applying color thresholding for yellow color, I applied sobel x gradient to detect white lines. Combining them is the final step. Below is the visualitation of thresholding contribution after applying to all test images. Green color is to detect yellow line and blue color is to detect white line. The right side is combined binary threshold.
![alt text][color_threshold]

The code is titled "Thresholding" in the Jupyter notebook and the function is `calc_thresh()` which takes an image (`img`)as its input, as well as `sobelx_thresh` and `color_thresh` for threshold values.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in "Birds Eye View" section of the IPython notebook.  The `warp()` function takes as inputs an image (`img`). I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])
dst = np.float32([(280,0),
                  (w-280,0),
                  (280,h),
                  (w-280,h)])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 464      | 280, 0        | 
| 707, 464      | 1000, 0       |
| 258, 682      | 280, 720      |
| 1049, 682     | 1000, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][bev]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I created `sliding_window_fit()` and `polyfit_tracking()` to identify lane lines and fit second order polynomial for both left and right lines. It is put under "Detection using sliding window" and "Polyfit based on previous frame" section in Jupyter Notebook.

`sliding_window_fit()` takes a warped binary image as in input. It will calculate the histogram of the bottom second third of the image. Detection for yellow line in the left side quite good so that I decided to just get the local maxima from the left half of the image. However, for the right side, I put a sliding window to detect the base line which is put under "finding rightx base" section in the function. It calculates the number of detected pixel in y direction within the window. It saves the value in a temporary array and return back the index of maximum value. This is more robust compared to finding local maxima which may lead to detect noise. The function then identifies nine windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below. This effectively "follows" the lane lines up to the top of the binary image. I collect pixels that are inside the window and use `np.polyfit()` to get a second order polynomial equation.

![alt text][polyfit1]

Once I get the result from the previous frame, I applied `polyfit_tracking()` which takes an binary warped image as an input, as well as previous fit for left and right side. This function will search nonzero pixel within 80 margin of each polynomial fit. It will speed up the process because I don't need to blindly search from beginning.

![alt text][polyfit2]



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curcature measurement is written under "Determine the curvature of the lane" section in Jupyter Notebook

The radius of curvature is based upon [this website](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) and calculated in the code cell titled "Radius of Curvature and Distance from Lane Center Calculation" using this line of code (altered for clarity):
```python
curve_radius = ((1 + (2*fit[0]*y_0*y_meters_per_pixel + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```

`y_meters_per_pixel` is determined by eyeballing the image. Below are the constants I used for calculation.

```
ym_per_pix = 30/720
xm_per_pix = 3.7/700
```

`calc_curvature` also calculates the position of the vehicle with respect to center. Assuming the camera is in the middle of the vehicle, the position can be estimated by substracting car position, which is `binary_warped.shape[1]/2`, and  lane centre position, `(r_fit_x + l_fit_x)/2`. The result of the substraction will be scaled by `xm_per_pixel` to get meter as the unit.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in "Warp the detected lane boundaries back onto the original image" section in the function `draw_lane()`. This function needs 5 inputs: `original_img, binary_img, l_fit, r_fit, Minv`.

`Minv` contains inverse matrix to transform warped image back to the original image. Applying `cv2.addWeighted()` will overlay it and make it look nice.

After drawing the map, I put the curvature and vehicle position information on the top left of the image by calling `put_curvature()` function.

Here is an example of my result on a test image:

![alt text][draw_lane]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problem is that my pipeline is not really robust to overcome extreme lighting conditions, shadows, and bumpy roads. I think the better approach is using deep learning for image segmentation. From there we can get the lines and use it to determine the lane.

On top of that, I am not sure if birds eye view is the best method. I would be happy if you can share some papers or research related to this.