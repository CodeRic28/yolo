import numpy as np
import argparse
import time
import cv2
import os
import imutils
from imutils.video import VideoStream
from imutils.video import FPS


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


# Initialize a dictionary to keep track of stationary objects
stationary = {}

# Initialize tracker using OpenCV tracking algorithm
tracker = cv2.TrackerKCF_create()


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
time.sleep(2.0)
writer = None
(W, H) = (None, None)

# Initialize the frames per second throughput estimator
fps = FPS().start()

try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("{} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("could not determine # of frames in video")
    total = -1

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    frame = frame[1] if args.get("video", False) else frame

    # frame = imutils.resize(frame, width=500)
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    detections = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # confidences, and class IDs


                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

                detection_info = {'box': [x, y, int(width), int(height)],
                                  'confidence': float(confidence),
                                  'classID': classID}
                detections.append(detection_info)
    boxes = [d['box'] for d in detections]
    confidences = [d['confidence'] for d in detections]

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # If the object is not in the stationary dictionary, add it
            if i not in stationary:
                stationary[i] = {'last_seen': time.time(), 'position': detections[i]['box'], 'classID': detections[i]['classID']}
            else:
                del stationary[i]
            # else:
            #     # If the object has not moved much, update the last seen time
            #     if np.linalg.norm(np.array(boxes[i][:2]) - np.array(stationary[i]['position'][:2])) < 20:
            #         stationary[i]['last_seen'] = time.time()
            #         stationary[i]['position'] = boxes[i]
            #     # If the object has moved, remove it from the stationary dictionary
            #     else:
            #         del stationary[i]


        # Iterate over stationary objects and check if they have been there
        # for more than 2 seconds, if so, draw their bounding box
        for i in list(stationary.keys()):
            if time.time() - stationary[i]['last_seen'] > 1:
                # draw bounding box and text
                (x, y, w, h) = stationary[i]['position']
                classID = stationary[i]['classID']
                color = [int(c) for c in COLORS[classID]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # If the object has not been seen for 2 seconds, remove it
            elif time.time() - stationary[i]['last_seen'] > 2:
                del stationary[i]

        # extract the bounding box coordinates
        # (x, y) = (boxes[i][0], boxes[i][1])
        # (w, h) = (boxes[i][2], boxes[i][3])
        # # draw a bounding box rectangle and label on the frame
        # color = [int(c) for c in COLORS[classIDs[i]]]
        # color = [int(c) for c in COLORS[stationary[i]['classID']]]
        # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        # text = "{}: {:.4f}".format(LABELS[stationary[i]['classID']], confidences[i])
        # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)
        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
    # write the output frame to disk
    writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()