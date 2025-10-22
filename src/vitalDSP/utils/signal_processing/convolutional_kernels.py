"""
Utility Functions Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.signal_processing.convolutional_kernels import ConvolutionalKernels
    >>> signal = np.random.randn(1000)
    >>> processor = ConvolutionalKernels(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np


class ConvolutionKernels:
    """
    A class that provides a collection of common convolution kernels for signal processing.

    Convolution kernels are used to perform various operations on signals, such as smoothing, sharpening, and edge detection. These operations are crucial in preprocessing steps, particularly in fields like signal processing, image processing, and computer vision.

    Methods
    -------
    smoothing : static method
        Generates a simple smoothing (average) kernel.
    sharpening : static method
        Generates a kernel for sharpening signals.
    edge_detection : static method
        Generates a kernel for detecting edges in signals.
    custom_kernel : static method
        Accepts and returns a custom kernel provided by the user.
    """

    @staticmethod
    def smoothing(size=3):
        """
        Generate a smoothing kernel, also known as an averaging kernel.

        A smoothing kernel is used to reduce noise or to create a blurred version of the signal by averaging neighboring values. This is often the first step in signal processing to remove high-frequency noise.

        Parameters
        ----------
        size : int, optional
            The size of the kernel (must be odd). The default is 3.

        Returns
        -------
        numpy.ndarray
            The smoothing kernel, where each element is equal to `1/size`.

        Raises
        ------
        ValueError
            If the size of the kernel is not an odd number.

        Examples
        --------
        >>> kernel = ConvolutionKernels.smoothing(size=3)
        >>> print(kernel)
        [0.33333333 0.33333333 0.33333333]
        """
        if size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        return np.ones(size) / size

    @staticmethod
    def sharpening():
        """
        Generate a sharpening kernel.

        A sharpening kernel enhances the differences between adjacent values in a signal, making the features more pronounced. This is particularly useful in applications like edge enhancement in images.

        Returns
        -------
        numpy.ndarray
            The sharpening kernel, typically centered around a positive value with negative values on either side.

        Examples
        --------
        >>> kernel = ConvolutionKernels.sharpening()
        >>> print(kernel)
        [ -1  2  -1 ]
        """
        return np.array([-1, 2, -1])

    @staticmethod
    def edge_detection():
        """
        Generate an edge detection kernel.

        An edge detection kernel is designed to highlight the boundaries between different regions in a signal. This kernel is particularly useful in detecting transitions or changes in intensity, which is essential in edge detection tasks in image processing.

        Returns
        -------
        numpy.ndarray
            The edge detection kernel, typically highlighting changes in intensity between adjacent elements.

        Examples
        --------
        >>> kernel = ConvolutionKernels.edge_detection()
        >>> print(kernel)
        [ -1  0  1 ]
        """
        return np.array([-1, 0, 1])

    @staticmethod
    def custom_kernel(kernel):
        """
        Use a custom convolution kernel provided by the user.

        This method allows the user to define a custom kernel that can be applied to a signal for specialized filtering or other operations.

        Parameters
        ----------
        kernel : numpy.ndarray
            The custom kernel provided by the user.

        Returns
        -------
        numpy.ndarray
            The custom kernel itself.

        Examples
        --------
        >>> kernel = np.array([0.2, 0.5, 0.2])
        >>> custom_kernel = ConvolutionKernels.custom_kernel(kernel)
        >>> print(custom_kernel)
        [0.2 0.5 0.2]
        """
        return kernel
