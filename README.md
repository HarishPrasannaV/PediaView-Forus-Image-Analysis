# PediaView Image Analysis Software
Pediview is a device developed for screening pediatric children for rential abnormalities and refractive errors using red reflex images of their eyes
## Image analysis software for PediaView developed for Forus Health
- The Image analysis software uses the color and IR images of the pupil captured by the device to screen for refractive error and other ocular/retinal disorders
- The color image is used for getting the red reflex of the eye and the IR image is used for pupil detection.

### Initial Results:

**Color Image:**
- Used for image analysis including estimation of refractive error
- Intensity of digital potentiometer 0xBF
- Color gain parameter in picamer2 (1.5,3.0)
- Exposure time 67ms
<img width="624" alt="image" src="https://github.com/user-attachments/assets/da163f7c-75a4-4fbb-9144-5ac7dc1f1c4f">



**IR Image:**
- Used for pupil detection
<img width="617" alt="image" src="https://github.com/user-attachments/assets/e6a9eae7-e477-4047-b351-d6f06e2c4f8f">

## Image Analysis Algorithms:
**Libraries Used:**
- Numpy
- Matplotlib
- OpenCV
- Sci-Kit Learn

### Pupil Detection:
#### Image Preprocessing:
- The pre-processing is handled by the ‘preprocessImage(image’ function.
- The image is converted to greyscale if the input is not already
- Contrast of the image is enhanced by using the histogram equalizer in openCV: cv2.equalizeHist(image)
- The image is blurred to remove noise: cv2.medianBlur(equalizedImage, 15)
<img width="626" alt="image" src="https://github.com/user-attachments/assets/3690144e-2aee-401d-b83c-4bc57606a8df">

(Blurred and Equalized Image)

- A threshold filter is applied to create a binary image from which the pupil can be detected: cv2.threshold(blurredImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
<img width="629" alt="image" src="https://github.com/user-attachments/assets/748082a0-b24b-4fc9-ac81-32ce43430ec5">

(Thresholded Binary Image from which the pupil can be detected)


#### Pupil(Contour) Detection and Image Cropping:
- The contour detection and highlighting is handled by detectContours(thresholdedImage, circularityThreshold=0.8, minArea=200) and highlightContours(image, contours)function
- Contours are detected by cv2.findContours(thresholdedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)and filtered by how circular they are, the ratio of the perimeter of the contour to it’s area gives a parameter caller ‘circularity’ using which the pupils are detected, the threshold circularity is set to 0.8
- For the detected contour a circle of similar size which completely encloses the contour is found using cv2.minEnclosingCircle(contour), This is done to eliminate jagged edges and ensure complete enclosure of pupil
- The highlightContours(image, contours)highlights the contours on the IR and color image
<img width="622" alt="image" src="https://github.com/user-attachments/assets/e7bd0195-b8bd-43e7-b0c5-ede1ce9244a3">

(The Pupils highlighted on the IR Image)
- The contours detected from the IR image is overlaid on top of the color image to detect the red reflex
<img width="622" alt="image" src="https://github.com/user-attachments/assets/5b92e28b-aec6-4338-9f4d-a3542450ff31">

- The contours and the red reflex may not exactly align due to camera movement and patient movement, but this leads to minimal error in image analysis and can largely be ignored
- The contour is then used as a mask and the red reflex is cropped out for further analysis by cropContours(image, contours)function
<img width="614" alt="image" src="https://github.com/user-attachments/assets/059fef5d-1bf2-4270-bf29-fa5b0438bc23">



### Image Analysis:
#### Image Intensity Analysis:
- The calculateColorAndIntensity(croppedImages)function give the average intensities of the colors Red, Blue and Green in the pupils
<img width="623" alt="image" src="https://github.com/user-attachments/assets/54372613-02c3-4172-942d-1f1a4288b939">


#### Gradient Analysis and Flagging for Refractive Error:
- Ideally an IR image take with one IR LED on should be user for gradient analysis but since individual IR control is not possible in this setup, we just use the grayscale version of the color image
- The plotIntensityProfiles3D(croppedImages)function takes the cropped images of the pupils and first converts it to grayscale if it’s not already.
- Then the intensities of the pupils are plotted on a 3D graph using matplotlib
- A plane is fit on the the 3D intensity graph using fitPlane(cropped)
- The magnitude and the sloped of the filled plane is calculated and then used for analysis
<img width="577" alt="image" src="https://github.com/user-attachments/assets/1e62267b-67d2-4307-b367-454a7c88392c">
<img width="549" alt="image" src="https://github.com/user-attachments/assets/486b8831-0929-4a2b-9002-cd7a90f33ec0">


<img width="619" alt="image" src="https://github.com/user-attachments/assets/4f41703b-2265-4b0f-a56f-715c7c34b141">

- For the LED configuration of the device, an upward slope along x-axis indicates myopia and a downward slope indicates hyperopia
- The refractiveError(slopes, a_dir, thresh=0.4)uses the magnitude and direction of the slope of the pupil to flag for myopia or hyperopia, the threshold for begin flagged for refractive error is having a slope of greater than 0.4

#### Flagging for Ocular Disorders:
- The isWhite(image, threshold=200)and isRed(image, threshold=100)functions are used to detect the color of the pupils.
- The ocularDisorders(croppedImages)use the above functions to flag for ocular disorders based on presence of white reflex or absence of red reflex.
#### Flagging for Amblyopia:
- The amblyopia(slopes,thresh=0.6)function calculates the difference between the slopes of the pupils, if the difference is greater than 0.6(the threshold) then the subject is flagged for amblyopia

<img width="515" alt="image" src="https://github.com/user-attachments/assets/dffc29f5-f2d3-4457-b978-70c70dddd646">




