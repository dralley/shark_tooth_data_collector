# Shark Tooth Data Collector

Uses OpenCV to take circle measurements of Shark Teeth, replace the background with transparency, and crop/shrink the images to a smaller size

### Background

This is kind of a hackish one-off project, but I thought it might be useful to someone. If not for shark teeth, perhaps as an example of how to use the OpenCV library.  I make no guarantees about the maintainability of this code, which was written over the course of 24 hours.

In October 2015, my group in a Computer Science elective class was assigned to work with a Paleontology professor at NC State University.  As a part of this assignment, I developed this tool to help automate measuring shark teeth from a picture containing a scale bar for reference.  It finds the minimum enclosing circle around each shark tooth, in millimeters, using a small blue 1mm scale bar in the image as a reference.  The background of each image is then replaced by transparency so that the teeth can be used as sprites for a "virtual fossil dig" website. Finally, filenames and measurements of all the teeth are dumped into a JSON file for external processing.

The shark teeth are measured using "circle measurements", a type of measurement wherein the tooth is fitted to the smallest diameter circle that can fully contain it.  Because we were trying to approximate the actual process of taking circle measurements, we abstained from attempting to take more accurate measurements than the scientific standard of 1mm.

### Usage

    ./collector.py input_directory_path output_directory_path

Example: 

    python3 collector.py ExamplePics OutputPics


##### Optional arguments:

    -d, --debug

Visually displays the measured size (green circle and text), the actual size (minimum enclosing circle, in red), and the fitting circle one size smaller than the fitted size for comparison purposes (in blue).  Debug mode allows you to iterate through the images by pressing any key on the keyboard, and also will not save the processed images.

    -v, --verbose

Visually displays extra intermediate steps for debugging purposes, including raw contours for the tooth and scaling bar, and the cropping box.


### Example Output

##### data.json:

    [
        {
            "measurement": 5,
            "filename": "4-15-1-lab.png"
        },
        ...
        ...
    ]

##### Output Image

(Note that the background is transparent, not white, and that the scale bar has been stripped out)

![](https://raw.githubusercontent.com/dralley/shark_tooth_data_collector/master/ExampleOutput/4-15-901-ling.png)


##### Debugging Output

![](http://i.imgur.com/xmULqSt.png)


Dependencies:

* OpenCV 3.x (see note below)
* NumPy

### Note:

You *must* use OpenCV version 3.0 or later.  Version 2.4.x (the most commonly-available version at this time) will not work due to several backwards-incompatible changes in the OpenCV 3.0 API
