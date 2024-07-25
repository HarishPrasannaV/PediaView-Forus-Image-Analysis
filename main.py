import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import LinearRegression

# Generating binary image
def preprocessImage(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalizedImage = cv2.equalizeHist(image)
    blurredImage = cv2.medianBlur(equalizedImage, 15)
    _, thresholdedImage = cv2.threshold(blurredImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholdedImage, blurredImage

# Detecting contours from the binary image and filtering based on circularity
def detectContours(thresholdedImage, circularityThreshold=0.8, minArea=200):
    contours, _ = cv2.findContours(thresholdedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filteredContours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity >= circularityThreshold and area >= minArea:
            # Calculate the minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            # Increase the radius slightly (e.g., by 10%)
            radius *= 1
            # Create a new contour that is a circle with the increased radius
            circleContour = np.array([[[int(x + radius * np.cos(theta)), int(y + radius * np.sin(theta))]] for theta in np.linspace(0, 2 * np.pi, 100)], dtype=np.int32)
            filteredContours.append((circleContour, area))
    return filteredContours

# Highlighting the contours on the image
def highlightContours(image, contours):
    if len(image.shape) == 2:
        outputImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        outputImage = image.copy()
    for contour, area in contours:
        cv2.drawContours(outputImage, [contour], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            textX = cX
            textY = cY - 10
            cv2.putText(outputImage, f'{int(area)}', (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return outputImage

# Cropping out the pupils with contours
def cropContours(image, contours):
    croppedImages = []
    for contour, area in contours:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        maskedImage = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(contour)
        cropped = maskedImage[y:y + h, x:x + w]
        croppedImages.append(cropped)
    return croppedImages


# Calculating the intensities of R,G & B in pupils
def calculateColorAndIntensity(croppedImages):
    colorAndIntensity = []
    for cropped in croppedImages:
        if len(cropped.shape) == 3:
            meanIntensity = cv2.mean(cropped)[:3]  # Get mean intensity for B, G, R channels
        else:
            meanIntensity = (cv2.mean(cropped)[0],) * 3  # For grayscale images, repeat intensity for B, G, R

        colorAndIntensity.append(meanIntensity)
    return colorAndIntensity

# Using linear regression to fit a plane to the 3D graph of the pupil intensities
def fitPlane(cropped):
    if cropped.ndim == 3:  # Convert color image to grayscale
        z = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        z = cropped

    # Create meshgrid for x and y coordinates
    x = np.arange(z.shape[1])
    y = np.arange(z.shape[0])
    x, y = np.meshgrid(x, y)

    # Flatten the arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Fit a linear model to the data
    A = np.vstack([x_flat, y_flat, np.ones_like(x_flat)]).T
    model = LinearRegression()
    model.fit(A, z_flat)

    # Extract coefficients
    a, b = model.coef_[0], model.coef_[1]
    c = model.intercept_

    # Create the fitted plane
    z_fit = a * x + b * y + c

    return z_fit, a, b


# Converting pupils to greyscale and plotting the 3d intensity graph
def plotIntensityProfiles3D(croppedImages):
    slopes=[]
    a_dir=[]
    b_dir=[]
    for i, cropped in enumerate(croppedImages):
        if cropped.size == 0:
            continue

        fig = plt.figure()

        # Plot original data
        ax = fig.add_subplot(121, projection='3d')
        x = np.arange(cropped.shape[1])
        y = np.arange(cropped.shape[0])
        x, y = np.meshgrid(x, y)
        if cropped.ndim == 2:  # Grayscale image
            z = cropped
        elif cropped.ndim == 3:  # Color image
            z = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        ax.plot_surface(x, y, z, cmap='gray')
        ax.set_title(f'Intensity Profile of Pupil {i + 1}')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Intensity')

        # Fit a plane and plot it
        z_fit, a, b = fitPlane(cropped)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(x, y, z, cmap='gray', alpha=0.5)
        ax2.plot_surface(x, y, z_fit, color='blue', alpha=0.5)
        ax2.set_title(f'Plane Fit on Pupil {i + 1}')
        ax2.set_xlabel('X axis')
        ax2.set_ylabel('Y axis')
        ax2.set_zlabel('Intensity')

        plt.show()

        slope = np.sqrt(a ** 2 + b ** 2)
        direction_x = "upwards" if a > 0 else "downwards"
        print(f'Slope of Pupil {i + 1}: {slope:.4f}')
        print(f'Direction of slope along X axis: {direction_x}')
        slopes.append(slope)
        a_dir.append(a)
    return slopes,a_dir

# Refractive error analysis with slope of the plane and it's direction
def refractiveError(slopes, a_dir, thresh=0.4):
    result = False
    type = ''
    a = max(a_dir)
    for i in slopes:
        if i > thresh:
            result = True
    if a <0:
        type = "Myopia"
    else:
        type = "Hyperopia"

    return result,type

def amblyopia(slopes,thresh=0.6):
    diff = abs(slopes[0] - slopes[1])
    if diff > thresh:
        return True
    else:
        return False


# Analysis for ocular disorders based on the colour of pupils
def isWhite(image, threshold=200):
    meanColor = cv2.mean(image)[:3]
    return all(channel > threshold for channel in meanColor)
def isRed(image, threshold=100):
    meanColor = cv2.mean(image)[:3]
    return meanColor[2] > threshold
def ocularDisorders(croppedImages):
    white_detected = 0
    red_detected = 0
    for cropped in croppedImages:
        if isWhite(cropped):
            white_detected = white_detected + 1
        elif isRed(cropped):
            red_detected = red_detected + 1

    if white_detected == 0 and red_detected == 2:
        print("No Ocular Disorders Detected")
    elif white_detected !=0:
        print("Flagged for Ocular Disorders")
    elif red_detected < 2 and white_detected == 0:
        print("Flagged for Amblyopia")


# Enter the path to the IR and Colour Images here
pupil_image = cv2.imread('images/lir.jpg', cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread('images/lcr.jpg', cv2.IMREAD_COLOR)

if pupil_image is None or color_image is None:
    raise FileNotFoundError("One or both image files were not found.")


thresholdedImage,blurred = preprocessImage(pupil_image)
contours = detectContours(thresholdedImage, circularityThreshold=0.8, minArea=5000)
outputImage = highlightContours(pupil_image, contours)
cOut = highlightContours(color_image, contours)
c_croppedImages = cropContours(pupil_image, contours)
g_croppedImages = cropContours(color_image, contours)

print("Intensities of colours in pupils:")
colorAndIntensity = calculateColorAndIntensity(g_croppedImages)
for i, (blue, green, red) in enumerate(colorAndIntensity):
    print(f'Pupil {i + 1}: Blue- {blue}, Green- {green}, Red- {red}')
print("\n")

cv2.imshow('Contours Highlighted on Grayscale', outputImage)
cv2.imshow('Contours Highlighted on RGB', cOut)
cv2.imshow('Blob', thresholdedImage)
cv2.imshow('Blurred', blurred)

for i, cropped in enumerate(g_croppedImages):
    windowName = f'Cropped Image {i + 1}'
    cv2.imshow(windowName, cropped)

#Refractive Error
print("Gradient analysis of pupils:")
slopes,a_dir = plotIntensityProfiles3D(g_croppedImages)
result,type = refractiveError(slopes,a_dir, thresh=0.4)

print("\n")

print("Results:")
if result:
    print(f"Flagged for {type}")
else:
    print("No refractive error")

#Amblyopia

if amblyopia(slopes, thresh = 0.6):
    print("Flagged for Amblyopia")
else:
    print("No Amblyopia")

#OcularDisorders

ocularDisorders(g_croppedImages)


cv2.waitKey(0)
cv2.destroyAllWindows()
