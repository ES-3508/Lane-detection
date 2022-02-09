
import cv2
import numpy as np
import datetime

#Region for line detection
region_of_interest_vertices = [
    (320,570),
    (480,570),
    (440, 0)
]

#function to get slope and intecept of two points
def C_m(x1,y1,x2,y2):
    
    m=(y2-y1)/(x2-x1)
    c=(y1*x2-y2*x1)/(x2-x1)
    return m,c
    
    
#Create mask for interest region
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def drow_the_lines(img, lines):
    
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        cv2.line(blank_image, (lines[0,0,0],lines[0,0,1]), (lines[0,0,2],lines[0,0,3]), (0, 255, 0), thickness=2)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img
    
    
    

def linedetect(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 200, 250)
    cropped_image = region_of_interest(canny_image,
                np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                        rho=6,
                        theta=np.pi/180,
                        threshold=220,
                        lines=np.array([]),
                        minLineLength=300,
                        maxLineGap=100)
    return lines
   

#To get center line points
image = cv2.imread("Image path")
lines=linedetect(image)
x_1, y_1, x_2, y_2=lines[0,0,0],lines[0,0,1],lines[0,0,2],lines[0,0,3]


# load a video
video = cv2.VideoCapture('video path')

# You can set custom kernel size if you want.
kernel = None

# Initialize the background object.
backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

while True:
    
    # Read a new frame.
    ret, frame = video.read()

    # Check if frame is not read correctly.
    if not ret:
        
        # Break the loop.

        break
    
    # Apply the background object on the frame to get the segmented mask. 
    fgmask = backgroundObject.apply(frame)
    
    # Perform thresholding to get rid of the shadows.
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
   
    
    # Apply some morphological operations to make sure you have a good mask
    fgmask = cv2.erode(fgmask, kernel, iterations = 1)
    fgmask = cv2.dilate(fgmask, kernel, iterations = 2)
    
    # Detect contours in the frame.
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the frame to draw bounding boxes around the detected cars.
    frameCopy = frame.copy()
    
    # loop over each contour found in the frame.
    for cnt in contours:
        
        # Consider large moving part to reducwe noise
        if cv2.contourArea(cnt) > 400:
            
            # RO get vehicle bounding box coordinates from the contour.
            x, y, width, height = cv2.boundingRect(cnt)
            
            # Draw a bounding box around the car.

            if lines is not None:
                x_1, y_1, x_2, y_2=lines[0,0,0],lines[0,0,1],lines[0,0,2],lines[0,0,3]
            
            
            
            if((x_1-x_2)!=0):
                m,c=C_m(x_1, y_1, x_2, y_2)
            

            r=m*(x+width-10)+c
            l=y+height+10
            
            
            #Checj the cross line or not 
            if(l>r):
                r=m*(x+10)+c
                l=y-10
                
                if(l<r):
                    
                    cv2.rectangle(frameCopy, (x , y), (x + width, y + height),(0, 0,255), 2)
                    cv2.putText(frameCopy, 'Car Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
                    current_time = datetime.datetime.now()
                    txt='capture/'+ str(current_time.hour) + str(current_time.minute) +str(current_time.second) +'.jpg'
                    cv2.imwrite(txt, frameCopy)
                    
    # Extract the foreground from the frame using the segmented mask.
    foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)
    
    
    
    try:
        #call line detection function and draw lines
        lines=linedetect(image)
        #print(lines[0,0,0])
        framen = drow_the_lines(frameCopy, lines)
        
    except :
        framen = frame.copy()
    
    cv2.imshow('Detection', framen)
    # Stack the original frame, extracted foreground, and annotated frame. 
    #stacked = np.hstack((frame, foregroundPart, frameCopyn))

   


    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xff
    
    # Check if 'q' key is pressed.
    if k == ord('q'):
        
        # Break the loop.
        break

# Release the VideoCapture Object.
video.release()

# Close the windows.q
cv2.destroyAllWindows()






