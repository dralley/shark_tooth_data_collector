#!/usr/bin/env python3

from __future__ import division, print_function

import os
import cv2
import math
import json
import argparse
import numpy as np


DEBUG = False
VERBOSE = False


class Colors:

    """ Enumeration of colors to use for drawing / debugging purposes
    """
    BLACK = (0, 0, 0)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)


class ToothPic:

    """ Lame pun...
    """

    def __init__(self, src):
        # File pathname
        self.path = src
        # Image
        self.image = cv2.imread(src)
        # Name of the file, ignoring the extension and the full path
        self.name = src.split("/")[-1].split(".")[0]
        # Name of the processed picture (set once the image is saved)
        self.fname = None
        # Debugging images
        self.debug_imgs = {"debug": self.image.copy()}
        # Pixels per millimeter as determined by the scale in the image
        self.scale = self._get_scale()
        # Size of the tooth
        self.measurement = self.measure()
        # Key, crop and resize the image
        self.scaled = self._key_and_resize()

        if DEBUG:
            for name, img in self.debug_imgs.items():
                cv2.imshow(name, img)

            cv2.waitKey(-1)
            cv2.destroyAllWindows()

    def _get_tooth_contour(self):
        """ Get the tooth contour
        """
        hsv_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv_img)

        # Threshold (binary inverse) the image to kill anything brighter than a threshold of 225
        ret, thresh = cv2.threshold(val, 225, 255, cv2.THRESH_BINARY_INV)

        img = thresh.copy()

        # Erode + Dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.erode(img, kernel)
        img = cv2.dilate(img, kernel)

        # Find contours in the image
        _, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # The largest contour is the tooth
        tooth_contour = max(contours, key=lambda x: cv2.contourArea(x))

        if DEBUG and VERBOSE and len(contours) > 0:

            self.debug_imgs["tooth"] = np.zeros(img.shape, np.uint8)
            cv2.drawContours(self.debug_imgs["tooth"], [tooth_contour], 0, Colors.WHITE, -1)

        return tooth_contour

    def _get_scale(self):
        """ Return the number of pixels per millimeter
        """
        hsv_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv_img)

        # Threshold (binary inverse) the image to kill anything with saturation less than 160
        ret, thresh = cv2.threshold(sat, 160, 255, cv2.THRESH_BINARY)

        img = thresh

        # Erode + Dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.erode(img, kernel)
        img = cv2.dilate(img, kernel)

        # Find contours in the image
        _, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pixels = 104  # Seems to be a good default value

        # Some pictures don't have the scale...
        if len(contours) > 0:
            # The largest contour is the tooth
            scale_contour = max(contours, key=lambda x: cv2.contourArea(x))

            # Get the bounding rectangle of the scale contour
            (x, y, w, h) = cv2.boundingRect(scale_contour)

            # Get whichever measurement (width, height) of the bounding rectangle is larger.  Offset is to
            # better approximate the center of the handles, as opposed to the edge
            pixels = max(w, h) - 6

        if DEBUG:

            if len(contours) > 0:
                # Draw the left and right points onto the debug image, again using the offsets
                left = (int(x) + 3, int(y) + int(h / 2))
                right = (int(x + w) - 3, int(y) + int(h / 2))

                cv2.circle(self.debug_imgs["debug"], left, 4, Colors.BLACK, -1)
                cv2.circle(self.debug_imgs["debug"], right, 4, Colors.BLACK, -1)

                if VERBOSE:
                    self.debug_imgs["scale"] = img
                    # cv2.drawContours(self.debug_imgs["scale"], [scale_contour], 0, Colors.BLACK, -1)

        return pixels

    def _key_and_resize(self, resize_amt=4):
        """ Cut out the non-tooth parts of the image and make them transparent.  Then, crop and scale down the picture.
        """
        tooth_contour = self._get_tooth_contour()

        # Draw a white mask in the shape of the tooth contour
        rows, cols, channels = self.image.shape
        mask = np.zeros((rows, cols, 1), np.uint8)
        cv2.drawContours(mask, [tooth_contour], 0, 255, -1)

        # Erode + Dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel)

        # Mask the image to cut out everything not within the bounds of the tooth contour
        tooth_img = cv2.bitwise_and(self.image, self.image, mask=mask)

        # The alpha channel == the mask: everything set to 0 is transparent
        alpha = mask

        # Split up the original image into RGB channels, and then merge them back with alpha
        r, g, b = cv2.split(tooth_img)
        tooth_img = cv2.merge((r, g, b, alpha))

        # Get the bounding rectangle for the tooth contour
        (x, y, w, h) = cv2.boundingRect(tooth_contour)

        # Get 2 opposite corners of the rectangle.  P1 is bottom left, P2 is top right.
        # Use 5 pixel offsets for some extra room at the edges
        point1 = (int(x - 5), int(y - 5))
        point2 = (int(x + w + 5), int(y + h + 5))

        x1, y1 = point1
        x2, y2 = point2

        # roi is the cropped image
        roi = tooth_img.copy()[y1:y2, x1:x2]
        height, width = roi.shape[:2]

        scaled = cv2.resize(roi.copy(), (width // resize_amt, height // resize_amt))

        if DEBUG and VERBOSE:

            self.debug_imgs["crop"] = self.image.copy()
            cv2.rectangle(self.debug_imgs["crop"], point1, point2, Colors.GREEN, 3)

            self.debug_imgs["scaled"] = scaled

        return scaled

    def measure(self):
        """ Measure the tooth using the scale in the picture
        """
        tooth_contour = self._get_tooth_contour()

        def tooth_round(pix_diam, scale, alpha=.2):
            """ If the difference between the actual tooth diameter (in mm) and a smaller tooth measurement is less than
            alpha x 1mm, round down to the smaller measurement.  Otherwise, round upward.
            """
            raw_diam = pix_diam / self.scale

            lower_bound = int(math.floor(raw_diam))
            upper_bound = int(math.ceil(raw_diam))

            if raw_diam - lower_bound < alpha:
                return lower_bound

            return upper_bound

        # Find the minimum enclosing circle
        (x, y), pix_radius = cv2.minEnclosingCircle(tooth_contour)

        # Cast the center and radius to integers
        center = (int(x), int(y))
        pix_radius = int(pix_radius * .97)  # account for an innacuracy in the minEnclosingCircle algorithm
        pix_diam = 2 * pix_radius

        measurement = tooth_round(pix_diam, self.scale)

        if DEBUG:

            # cv2.line(self.image, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (0, 0, 255), 4)

            if VERBOSE:
                # Draw the outline of the tooth contour
                cv2.drawContours(self.debug_imgs["debug"], [tooth_contour], 0, Colors.RED, 3)

                # Draw all measurement circles
                for i in range(0, 15):
                    cv2.circle(self.debug_imgs["debug"], center, int(i / 2 * self.scale), Colors.BLUE, 4)

            # Write the measurement to the center of the image
            cv2.putText(self.debug_imgs["debug"], str(measurement), (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, Colors.GREEN, 3)

            # Draw the minimum enclosing circle
            cv2.circle(self.debug_imgs["debug"], center, pix_radius, Colors.RED, 4)

            # Draw the *actual* circle that it fits into
            cv2.circle(self.debug_imgs["debug"], center, int(measurement / 2 * self.scale), Colors.GREEN, 4)

            # Draw the circle 1 size smaller than the actual size
            cv2.circle(self.debug_imgs["debug"], center, int((measurement - 1) / 2 * self.scale), Colors.BLUE, 4)

        return measurement

    def save(self, path, extension=".png"):
        """ Write the image to a new directory using the original filename
        """
        self.fname = os.path.join(path, self.name + extension)
        cv2.imwrite(self.fname, self.scaled)

    def dump(self):
        """ Return a dictionary representation of the relevant information about the picture
        """
        return {"filename": self.fname.split("/")[-1], "measurement": self.measurement}


def main(src, dest):
    # List of paths to all of the images
    imgs = [os.path.join(src, img) for img in os.listdir(src)]

    # Make the destination directory if it doesn't exist
    if os.path.exists(dest):
        if os.path.isfile(dest):
            print("Destination is not a directory")
            exit()

        if os.listdir(dest):
            print("Destination directory not empty")
            exit()
    else:
        os.mkdir(dest)

    # Create a ToothPic object for each image
    toothpics = [ToothPic(img) for img in imgs]

    if not DEBUG:
        # Save cropped and transparency-added images to the new directory
        for pic in toothpics:
            pic.save(dest)

        img_dicts = [pic.dump() for pic in toothpics]

        json_str = json.dumps(img_dicts, indent=4)

        with open(os.path.join(dest, "data.json"), "w") as f:
            f.write(json_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CitizenScience Image Processor')

    parser.add_argument('source')
    parser.add_argument('destination', nargs="?")

    parser.add_argument('--debug', '-d', action='store_true', help='debug flag')
    parser.add_argument('--verbose', '-v', action='store_true', help='verbose debugging flag')

    args = parser.parse_args()

    DEBUG = args.debug
    VERBOSE = args.verbose

    src = os.path.abspath(args.source)
    dest = os.path.abspath(args.destination)

    main(src, dest)
