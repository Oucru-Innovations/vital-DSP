import numpy as np

class ConvolutionKernels:
    """
    A class that provides a collection of common convolution kernels for signal processing.

    Methods:
    - smoothing: A simple smoothing kernel.
    - sharpening: A kernel for sharpening signals.
    - edge_detection: A kernel for detecting edges in signals.
    - custom_kernel: Accepts a custom kernel provided by the user.
    """

    @staticmethod
    def smoothing(size=3):
        """
        Generate a smoothing kernel.

        Parameters:
        size (int): Size of the kernel (must be odd).

        Returns:
        numpy.ndarray: The smoothing kernel.

        Example:
        >>> kernel = ConvolutionKernels.smoothing(size=3)
        >>> print(kernel)
        """
        if size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        return np.ones(size) / size

    @staticmethod
    def sharpening():
        """
        Generate a sharpening kernel.

        Returns:
        numpy.ndarray: The sharpening kernel.

        Example:
        >>> kernel = ConvolutionKernels.sharpening()
        >>> print(kernel)
        """
        return np.array([-1, 2, -1])

    @staticmethod
    def edge_detection():
        """
        Generate an edge detection kernel.

        Returns:
        numpy.ndarray: The edge detection kernel.

        Example:
        >>> kernel = ConvolutionKernels.edge_detection()
        >>> print(kernel)
        """
        return np.array([-1, 0, 1])

    @staticmethod
    def custom_kernel(kernel):
        """
        Use a custom convolution kernel provided by the user.

        Parameters:
        kernel (numpy.ndarray): The custom kernel.

        Returns:
        numpy.ndarray: The custom kernel itself.

        Example:
        >>> kernel = np.array([0.2, 0.5, 0.2])
        >>> custom_kernel = ConvolutionKernels.custom_kernel(kernel)
        >>> print(custom_kernel)
        """
        return kernel
