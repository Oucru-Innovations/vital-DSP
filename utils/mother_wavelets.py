class Wavelet:
    """
    A class for generating different types of mother wavelets.

    Methods:
    - haar: Generates a Haar wavelet.
    - db: Generates a Daubechies wavelet.
    - sym: Generates a Symlet wavelet.
    - coif: Generates a Coiflet wavelet.
    - custom_wavelet: Generates a custom wavelet provided by the user.
    """

    @staticmethod
    def haar():
        """
        Generate a Haar wavelet.

        Returns:
        numpy.ndarray: The Haar wavelet.

        Example:
        >>> haar_wavelet = Wavelet.haar()
        >>> print(haar_wavelet)
        """
        return np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])

    @staticmethod
    def db(order=4):
        """
        Generate a Daubechies wavelet of a given order.

        Parameters:
        order (int): The order of the Daubechies wavelet.

        Returns:
        numpy.ndarray: The Daubechies wavelet.

        Example:
        >>> db_wavelet = Wavelet.db(order=4)
        >>> print(db_wavelet)
        """
        if order == 1:
            return np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        elif order == 2:
            return np.array([-0.12940952255092145, 0.22414386804185735, 0.8365163037378077, 0.48296291314469025])
        elif order == 4:
            return np.array([0.48296291314469025, 0.8365163037378077, 0.22414386804185735, -0.12940952255092145])
        else:
            raise ValueError("Order not supported for Daubechies wavelet")

    @staticmethod
    def sym(order=4):
        """
        Generate a Symlet wavelet of a given order.

        Parameters:
        order (int): The order of the Symlet wavelet.

        Returns:
        numpy.ndarray: The Symlet wavelet.

        Example:
        >>> sym_wavelet = Wavelet.sym(order=4)
        >>> print(sym_wavelet)
        """
        if order == 4:
            return np.array([-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.803738751805386,
                             0.297857795605605, -0.09921954357695668, -0.012603967262037833, 0.032223100604078126])
        else:
            raise ValueError("Order not supported for Symlet wavelet")

    @staticmethod
    def coif(order=1):
        """
        Generate a Coiflet wavelet of a given order.

        Parameters:
        order (int): The order of the Coiflet wavelet.

        Returns:
        numpy.ndarray: The Coiflet wavelet.

        Example:
        >>> coif_wavelet = Wavelet.coif(order=1)
        >>> print(coif_wavelet)
        """
        if order == 1:
            return np.array([-0.01565572852898419, -0.0727326195128539, 0.38486484686420286, 0.8525720202122554,
                             0.3378976624578092, -0.0727326195128539])
        else:
            raise ValueError("Order not supported for Coiflet wavelet")

    @staticmethod
    def custom_wavelet(wavelet):
        """
        Use a custom wavelet provided by the user.

        Parameters:
        wavelet (numpy.ndarray): The custom wavelet coefficients.

        Returns:
        numpy.ndarray: The custom wavelet.

        Example:
        >>> custom_wavelet = Wavelet.custom_wavelet(np.array([0.2, 0.5, 0.2]))
        >>> print(custom_wavelet)
        """
        return wavelet
