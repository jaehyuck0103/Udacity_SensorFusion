# Radar Target Generation and Detection

## Project Rubric (Write brief explanations for the following)

#### Implementation steps for the 2D CFAR process.

By using the integral image, the sum of values in rectangular areas can be efficiently calculated. Both `sum_trains` and `sum_guards` can be calculated from one integral image with different offsets.

https://en.wikipedia.org/wiki/Summed-area_table

I also avoid for_loop by matrix operations. 

#### Selection of Training, Guard cells and offset.

Parameters were set so that high CFAR values are obtained only in `initial_target_pos` and `target_vel`.

#### Steps taken to suppress the non-thresholded cells at the edges.

I give infinite threshold to the edges in order to suppress the CFAR values at the edges.
