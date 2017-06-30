#**Finding Lane Lines on the Road**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I blurred the images. Next, I used canny to find edges, and then I cropped the images in polygon. Next, I used hough lines to find lines and draw lines on the images.
For draw_lines () function, I detected the slope of the graph and divide them to the left and to the right. Then I calculated the average mean squared of slopes to reduce some extreme slopes and average mean of interception of lines. Next, I calculated the long lines for each side.


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the lane lines are not quite in this angle or in the polygon. Another shortcoming is when the segments of the lines are too short and large gaps between them.


###3. Suggest possible improvements to your pipeline

A possible improvement would be to always put the camera on the same angle.

Another potential improvement could be to tune the arguments more to get more precise lane lines.
