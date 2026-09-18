"""Classical (non-deep-learning) computer vision toolkit for inspection tasks.

This package provides small, dependency-light OpenCV wrappers for four
common industrial/security inspection tasks:

* round-object detection (Hough circle transform)
* alignment / edge detection (probabilistic Hough line transform)
* blob/defect segmentation (thresholding + contours)
* camera tamper / occlusion detection (background modelling + edge density)

See the top-level README for design rationale and usage examples.
"""

__version__ = "0.1.0"
