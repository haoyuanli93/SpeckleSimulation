import numpy as np

# This file contains functions from XRaySimulation package.

pi = np.pi
two_pi = 2. * np.pi

hbar = 0.0006582119514  # This is the reduced planck constant in keV/fs

c = 299792458. * 1e-9  # The speed of light in um / fs


# --------------------------------------------------------------
#               Unit conversion
# --------------------------------------------------------------
def kev_to_petahertz_frequency(energy):
    return energy / hbar * 2 * pi


def kev_to_petahertz_angular_frequency(energy):
    return energy / hbar


def kev_to_wavevec_length(energy):
    return energy / hbar / c


def petahertz_frequency_to_kev(frequency):
    return hbar * 2 * pi * frequency


def petahertz_angular_frequency_to_kev(angular_frequency):
    return hbar * angular_frequency


def petahertz_angular_frequency_to_wave_number(angular_frequency):
    return angular_frequency / c


def wavevec_to_kev(wavevec):
    """
    Convert wavevector
    wavevector = 2 pi / wavelength
    :param wavevec:
    :return:
    """
    return wavevec * hbar * c


def wavenumber_to_kev(wavenumber):
    """
    Convert wave number to keV.
    wavenumber = 1 / wavelength
    :param wavenumber:
    :return:
    """
    return wavenumber * hbar * c * two_pi


def sigma_to_fwhm(sigma):
    return 2. * np.sqrt(2 * np.log(2)) * sigma


def fwhm_to_sigma(fwhm):
    return fwhm / (2. * np.sqrt(2 * np.log(2)))


def intensity_fwhm_to_field_sigma(fwhm):
    return fwhm / (2. * np.sqrt(2 * np.log(2))) * np.sqrt(2)


def field_sigma_to_intensity_fwhm(sigma):
    return sigma * (2. * np.sqrt(2 * np.log(2))) / np.sqrt(2)


# --------------------------------------------------------------
#          Uncertainty Principle
# --------------------------------------------------------------
def bandwidth_sigma_kev_to_duration_sigma_fs(bandwidth_kev):
    return hbar / 2. / bandwidth_kev
