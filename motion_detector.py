"""
Raspberry Pi Motion Detector

"""

# import OpenCV, time, and Pandas library
import cv2, time, pandas
# import datetime class from datetime library
from datetime import datetime

# Initialize MOG2 background subtractor
backg_subtract = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# List when any moving object appear
# Tracks whether motion was detected from the previous frame to the current frame
motion_list = [ None, None ]

# Records and stores timestamps of movement. Start and end times
time = []

# Creates empty table with a start column and end column
# Each row represents a motion event
dataframe = pandas.DataFrame(columns = ["Start", "End"])

# opens first available camera of device.
# returns video object to read frames from
video = cv2.VideoCapture(0)

# Infinite while loop to treat stack of image as video
while True:

    # obtain next frame (image) from video
    # check is true if frame capture successful
    # frame is the captured image, which is a numpy array of the pixel data
    check, frame = video.read()

    # set motion to 0 (no motion detected) at start of each loop
    motion = 0


    # IMAGE PROCESSING
    # Convert frame's color image from RGB to create new grayscale frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply blur to grey frame 
    # prevents irrelevant pixel changes like noise from interfering
    gray = cv2.GaussianBlur(gray, (21, 21), 0)


    # Apply MOG2 to get foreground mask (moving objects) by subtracting background from current frame
    diff_frame = backg_subtract.apply(gray)

    # any pixel difference greater than 30 is considered motion and set to white, otherwise black
    # create new threshold frame with only black and white pixels
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in gaps, making contours easier to find
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

    # Finding outlines of moving object (white blobs)
    contours,_ = cv2.findContours(thresh_frame.copy(), 
                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loops through every detected contour (potential moving object)
    for contour in contours:

        # filters any contours too small to be considered motion (noise)
        if cv2.contourArea(contour) < 500:
            continue

        # contour large enough to be considered motion was found, set motion to 1
        motion = 1

        # get a rectangle that fits around the contour (moving object)
        (x, y, w, h) = cv2.boundingRect(contour)

        # creates a rectangle around the contour (moving object) in the original color frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    #MOTION LOGGING
    # Appending current frame's motion status
    motion_list.append(motion)

    #trims motion list to last two entries to save memory (previous and current frame)
    motion_list = motion_list[-2:]

    # if previous frame had no motion and current frame has motion, then motion started, so record start time
    if motion_list[-1] == 1 and motion_list[-2] == 0:
        time.append(datetime.now())

    # if previous frame had motion and current frame has no motion, then motion ended, so record end time
    if motion_list[-1] == 0 and motion_list[-2] == 1:
        time.append(datetime.now())

    # DISPLAYING FRAMES
    # Displaying image in gray_scale
    cv2.imshow("Gray Frame", gray)

    # Displaying the difference in currentframe to the staticframe(very first_frame)
    cv2.imshow("Difference Frame", diff_frame)

    # Displaying the black and white image
    # (if intensity difference greater than 30 it appears white)
    cv2.imshow("Threshold Frame", thresh_frame)

    # Displaying color frame with contour of motion of object
    cv2.imshow("Color Frame", frame)


    key = cv2.waitKey(1)

    # if q entered whole process will stop
    if key == ord('q'):
        # if something is moving then it append the end time of movement
        if motion == 1:
            time.append(datetime.now())
        break

# SAVING RESULTS
# Appending time of motion in DataFrame
rows = []
for i in range(0, len(time) - 1, 2):
    rows.append({"Start": time[i], "End": time[i + 1]})
dataframe = pandas.DataFrame(rows, columns=["Start", "End"])

# Filter out motion events shorter than 1 second (noise)
dataframe = dataframe[(pandas.to_datetime(dataframe["End"]) - pandas.to_datetime(dataframe["Start"])).dt.total_seconds() > 1]


# Creates a CSV file to save time movement data
dataframe.to_csv("Time_of_movements.csv")

# Release the video object to free up resources
video.release()

# Destroying all the windows
cv2.destroyAllWindows()