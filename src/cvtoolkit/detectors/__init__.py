"""Detector implementations used by the CLI and by library users directly.

Each module in this package is self-contained and can be imported and used
without going through the CLI, e.g.::

    from cvtoolkit.detectors.hough_circles import detect_circles
    detections = detect_circles(image)
"""
