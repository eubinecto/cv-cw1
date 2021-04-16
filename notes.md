

## Task 1 - implement a convolution function

> motivation

Convolution. I'm doing this once and for all, by implementing this from scratch!

> What am I supposed to do?
1. write `convolve_img()` function.
   - you should pad the edges of the input image with zeros. (how did I do this?)
   - we should perform explicit looping over it. (2)
   - start with a 3 by 3 convolution filter. (optional: parameterize the size of the filter as n)
2. compute the effect of taking (this should add up to the conclusion) -> should be included later in the report:
   - average
   - weighted-average smoothing kernels for the convolution
    
> What does it mean to perform "convolution" on an image with "structuring element"?

THat means to reduce the image into a smaller one, while maintaining important features of the original one.

For example,
any examples from the lectures?

The "structuring element" probably refers to a convolution filter.


> Why do I want to perform a convolution on `kitty.bmp`? 

To extract some features? Yes. If you look at the description for task 2:
- compute the "horizontal and vertical gradient images"
- find the "edge strength image", given by the gradient magnitude


> Why do we pad the image? 

Because otherwise, the image won't be a square...?


> How do I implement `convolve_img()`?

Look it up on the lectures maybe?


> How do I implement `pad_img()`

This, look it up on the lectures.

> Wouldn't the image get smaller after the convolution? How do we detect edges of the original
image then?


> What is Sobel or Prewitt kernel? What do they have to do with mean and weighed mean kernel?

Are they mentioned in the lectures somehow?


> What should you do if the sum of the kernel is 0 You can't divide by zero; You can't 
> array / array.sum() ?





## Task 2 - **calculate** the edge strength image from the gradient magnitude image.

> what am I supposed to do?
- compute the horizontal and vertical gradient images
- from these, compute the gradient magnitude image
- for that, calculate the edge strength image (2)
- maybe visualise before and after all of these

## Task 3 - **detect** the edges of the cat

> what am I supposed to do?

- detect the major edges by **thresholding** the edge strength image (2)
  - may find it useful to plot histogram for the edge-strength image
    - why might this be useful though?
  - find the threshold that gives the edges of the cat, but those of fur and wood-grain.  
  - visualise the process, include in the report. 


> How do we find a great threshold value?

By looking at the histogram, it seems. It should have "dips" in it.

Someone has tried using cv2 trackbar to find a good value interactively.


> How do we implement thresholding?

Someone has used the "Simple method" from code snippets.

Use the Snippet 09, the purpose of which is to "Threshold a greyscale image".




## Task 4 - **compare** the edges obtained with a mean filter vs. those obtained with a weighted mean filter


> what am I supposed to do?
- compare edge strength images when using a weighted average kernel (2)



## Conclusions

> what am I supposed to do?

- a report, that includes conclusions from each task. 
