import pywt
import numpy as np
from typing import Optional

class DiscreteWavelet: 
    """Description. 
    Discrete Wavelet Transform with specified wavelet and number of levels.
    
    Attributes: 
        - wavelet: wavelet type to use
        - level: decomposition level"""

    def __init__(self, wavelet: str="haar", level: int=4):
        self.wavelet = wavelet

        if level < 0: 
            raise ValueError("level must be >= 0.")
            
        self.level = level 

    def __repr__(self) -> str:
        return f"DiscreteWavelet(wavelet={self.wavelet}, level={self.level})"

    def _transform(self, x: np.ndarray): 
        """Description. 
        Perform the discrete wavelet transform on a 1D array."""

        self.coefs = pywt.wavedec(data=x, wavelet=self.wavelet, level=self.level)

    def _inverse_transform(self) -> np.ndarray: 
        """Description. 
        Perform the inverse discrete wavelet transform on a list of wavelet coefficients using the specified wavelet."""

        return pywt.waverec(coeffs=self.coefs, wavelet=self.wavelet)

    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Description.
        Perform soft thresholding on a 1D array with the specified threshold value."""
        
        return np.where(np.abs(x) > threshold, x - np.sign(x) * threshold, 0)

    def denoise(self, x: np.ndarray, threshold: Optional[float]=None) -> np.ndarray:
        """Description.
        Denoise a 1D array using the discrete wavelet transform with the specified wavelet, soft thresholding, and the inverse discrete wavelet transform.
        """
        if threshold is None:
            threshold = 2 * np.std(x)

        self._transform(x)
        self.coefs = list(map(lambda x: self._soft_threshold(x, threshold), self.coefs))

        x_denoised = self._inverse_transform()
        return x_denoised