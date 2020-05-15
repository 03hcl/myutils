from enum import Enum, auto

import numpy as np


class WindowFunction(Enum):

    # region Enum

    Rectangular = 0

    BSpline = auto()
    Triangular = auto()
    Parzen = auto()

    Hann = auto()
    Hanning = Hann
    Hamming = auto()
    Blackman = auto()
    Kaiser = auto()
    Bartlett = auto()

    BartlettHann = auto()
    ModifiedBartlettHann = BartlettHann
    Nuttall = auto()
    BlackmanHarris = auto()
    BlackmanNuttall = auto()
    FlatTop = auto()
    PlanckBessel = auto()
    HannPoisson = auto()
    RifeVincent = auto()

    Welch = auto()
    Sine = auto()
    PowerOfSine = auto()
    PowerOfCosine = PowerOfSine
    CosineSum = auto()
    Sinc = auto()
    Akaike = auto()
    Lanczos = auto()

    Vorbis = auto()
    KBD = auto()
    KaiserBesselDerived = KBD

    Gauss = auto()
    ConfinedGaussian = auto()
    ApproximateConfinedGaussian = auto()
    GeneralizedNormal = auto()
    Tukey = auto()
    PlanckTaper = auto()
    DPSS = auto()
    Slepian = DPSS
    DolphChebyshev = auto()
    Ultraspherical = auto()
    Exponential = auto()
    Poisson = Exponential

    # endregion

    def create(self, window_size: int, **kwargs) -> np.ndarray:
        if self == WindowFunction.Rectangular:
            return np.ones(window_size)
        if self == WindowFunction.Hamming:
            return np.hamming(window_size)
        if self == WindowFunction.Hann:
            return np.hanning(window_size)
        if self == WindowFunction.Blackman:
            return np.blackman(window_size)
        if self == WindowFunction.Bartlett:
            return np.bartlett(window_size)
        if self == WindowFunction.Kaiser:
            return np.kaiser(window_size, **kwargs)
        raise NotImplementedError
