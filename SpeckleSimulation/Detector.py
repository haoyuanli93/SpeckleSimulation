import numpy as np

Thomson_scattering_length = 2.818e-9  # um


######################################################
#      Define detector class and functions
######################################################
class Detector2D:
    def __init__(self,
                 pixel_size=None,
                 pixel_numbers=None,
                 normal_direction=np.array([0, 0, 1.]),
                 detector_center=np.array([0., 0., 1e6])):
        #####################################################
        #    User specify
        #####################################################
        # Define detector shape
        if pixel_numbers is None:
            pixel_numbers = [1024, 1024]
        if pixel_size is None:
            pixel_size = [50., 50.]

        self.pixel_size = pixel_size  # in um
        self.pixel_numbers = pixel_numbers  # in pixel

        # Define detector geometry: orientation and position
        self.normal_direction = normal_direction

        # Detector center position with respect to the interaction point.
        self.detector_center = detector_center

        #####################################################
        #    Quantities for automatic calculation
        #####################################################
        # Define direction that are perpendicular to the normal direction
        # The definition is such that dimension 0 is always parallel to the x-z plane
        self._edge_direction_1 = np.array([0, 1., 0.])
        self._edge_direction_2 = np.array([1., 0, 0.])

        # Create pixel position
        self.pixel_positions = np.zeros((self.pixel_numbers[0],
                                         self.pixel_numbers[1],
                                         3))

        self.pixel_distance = np.zeros((self.pixel_numbers[0],
                                        self.pixel_numbers[1]))

        self.pixel_direction = np.zeros((self.pixel_numbers[0],
                                         self.pixel_numbers[1],
                                         3))

        # Create holder for solid angle
        self.solid_angles = np.zeros((self.pixel_numbers[0],
                                      self.pixel_numbers[1]))

        self.projected_pixel_area = np.zeros((self.pixel_numbers[0],
                                              self.pixel_numbers[1]))

        #####################################################
        #    Calculate the quantities listed above
        #####################################################
        self.__update_detector_directions()
        self.__get_pixel_positions()
        self.__get_pixel_solid_angles()

    def __update_detector_directions(self):
        # Normalize the normal direction
        self.normal_direction /= np.linalg.norm(self.normal_direction)

        # Calculate edge direction 1
        self._edge_direction_1 = np.cross(np.array([0., 1., 0.]), self.normal_direction)
        self._edge_direction_1 /= np.linalg.norm(self._edge_direction_1)

        # Calculate edge direction 2
        self._edge_direction_2 = np.cross(self.normal_direction, self._edge_direction_1)
        self._edge_direction_2 /= np.linalg.norm(self._edge_direction_2)

    def __get_pixel_positions(self):
        # Create the relative pixel position with respect to the center
        x_coor_tmp = np.arange(self.pixel_numbers[0]).astype(np.float64)
        x_coor_tmp -= (float(self.pixel_numbers[0]) / 2. - 0.5)
        x_coor_tmp *= self.pixel_size[0]

        y_coor_tmp = np.arange(self.pixel_numbers[1]).astype(np.float64)
        y_coor_tmp -= (float(self.pixel_numbers[1]) / 2. - 0.5)
        y_coor_tmp *= self.pixel_size[1]

        # Create the pixel position with respect to the detector center
        self.pixel_positions[:, :, :] += np.outer(x_coor_tmp, self._edge_direction_1)[:, np.newaxis, :]
        self.pixel_positions[:, :, :] += np.outer(y_coor_tmp, self._edge_direction_2)[np.newaxis, :, :]

        # Calculate the pixel position with respect to the intersection point
        self.pixel_positions += self.detector_center[np.newaxis, np.newaxis, :]

    def __get_pixel_solid_angles(self):
        # Calcualte the distance between each pixel and the interaction position
        self.pixel_distance = np.sqrt(np.sum(np.square(self.pixel_positions), axis=-1))

        # Calculate the direction vector towards the center of each pixel.
        self.pixel_direction = self.pixel_positions / self.pixel_distance[:, :, np.newaxis]

        # Calculate the projection cos angle for each pixel
        cosines = np.dot(self.pixel_direction, self.normal_direction)

        # Calculate the pixel area projected to the tangent direction of the corresponding spere.
        self.projected_pixel_area = np.prod(self.pixel_size) * cosines

        # Calculate the solid angle
        self.solid_angles = self.projected_pixel_area / np.square(self.pixel_distance)


def get_detector_pixel_q_vectors(detectors, k_vec_in):
    """
    Given an incident wave vector, return the q vector for each pixel on the detector.
    Here wave vector is defined to be
        2 pi / wave length

    :param detectors:
    :param k_vec_in:
    :return: q vector for each pixel, q vector length for each pixel.
    """
    # Get kin direction
    k_len = np.linalg.norm(k_vec_in)
    k_direction = k_vec_in / k_len

    # Get the q vector for each pixel
    pixel_q_vec = k_direction[np.newaxis, np.newaxis, :] - detectors.pixel_direction
    pixel_q_vec *= k_len

    # Get the length of the q vectors
    pixel_q_len = np.sqrt(np.sum(np.square(pixel_q_vec), axis=-1))

    return pixel_q_vec, pixel_q_len


def get_polarization_correction(detector, polarization_in=np.array([0., 1., 0.])):
    """
    Get the polarization correction for each pixel.

    :param detector:
    :param polarization_in:
    :return:
    """
    # Normalize the polarization vector
    polarization = polarization_in / np.linalg.norm(polarization_in)

    # Get the cross product
    polarization_correction = np.sum(np.square(np.cross(detector.pixel_direction,
                                                        polarization)),
                                     axis=-1)
    return polarization_correction


def get_efield_amplitude(efield_phase, detector, k_in, efield_in):
    """
    Convert the calculated electric field phase into the electric field amplitude
    :param efield_phase:
    :param detector:
    :param k_in:
    :param efield_in:
    :return:
    """
    k_len = np.linalg.norm(k_in)
    # Notice that the sign of the phase is different from that of the interfering phase in gpu calculation
    pre_fectors = efield_in * np.exp(-1.j * detector.pixel_distance * k_len) / detector.pixel_distance
    e_field_amplitude = np.multiply(efield_phase, pre_fectors)
    return e_field_amplitude


def get_ideal_detector_image(pixel_projected_area, polarization_correction, e_field_amplitude):
    """
    Convert the electric field amplitude to the measured intensity by each pixel on the detector.
    Basically, one adds polarization correction, pixel area restrictions
    and Thomson_scattering_length ** 2

    :param pixel_projected_area:
    :param polarization_correction:
    :param e_field_amplitude:
    :return:
    """
    holder = e_field_amplitude.real ** 2 + e_field_amplitude.imag ** 2
    holder = np.multiply(pixel_projected_area, holder)
    holder = np.multiply(polarization_correction, holder)
    holder *= np.square(Thomson_scattering_length)

    return holder
