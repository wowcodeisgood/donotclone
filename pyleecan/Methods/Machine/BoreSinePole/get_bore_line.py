from ....Classes.Segment import Segment
from ....Methods import ParentMissingError
from numpy import linspace, pi, cos, tan, exp
from scipy.optimize import fmin

NN = 30  # number of segments per half pole


def get_bore_line(self, prop_dict=None):
    """Return the bore line description

    Parameters
    ----------
    self : BoreSinePole
        A BoreLSinePole object
    prop_dict : dict
        Property dictionary to apply on the lines

    Returns
    -------
    bore_list : list
        List of bore lines
    """

    if self.parent is not None:
        Rbo = self.parent.get_Rbo()
    else:
        raise ParentMissingError("Error: The slot is not inside a Lamination")

    logger = self.get_logger()

    # Compute the shape
    alpha1 = pi / self.N
    w_max = _get_pole_width_max(self)
    if self.W0 is not None and self.W0 < w_max:
        w_max = self.W0
    elif self.W0 is not None and self.W0 > w_max:
        logger.warning("Enforcing permissible pole width (W0).")

    phi_max = _get_phi(self, w_max)
    phi = linspace(-phi_max, phi_max, 2 * NN + 1)

    Zedge = self.get_pole_shape(-phi_max)
    xedge = (Zedge * exp(1j * alpha1)).real
    Rq = Rbo + self.delta_d - self.delta_q

    # Create the first pole bore line
    Z = [self.get_pole_shape(p) for p in phi]

    if self.delta_q is not None and Rq < xedge:
        line = Segment(Zedge * exp(1j * alpha1), 1j * Zedge.imag * exp(1j * alpha1))
        inter = line.intersect_line(Rq, Rq + 1j * Rq)
        if len(inter) != 1:
            raise ()  # TODO

        if inter[0].imag <= 0:
            inter = line.intersect_line(0, Rq)
            if len(inter) != 1:
                raise ()  # TODO

            logger.warning("Enforcing permissible q axis air gap (delta_q).")
            Z.insert(0, inter[0] * exp(-1j * alpha1))
            Z.append(Z[0].conjugate())

        else:
            Z.insert(0, inter[0] * exp(-1j * alpha1))
            Z.append(Z[0].conjugate())
            Z.insert(0, Rq * exp(-1j * alpha1))
            Z.append(Z[0].conjugate())

    else:
        Z.append(xedge * exp(1j * alpha1))
        Z.insert(0, Z[-1].conjugate())

    # Create the lines
    bore_list = list()
    for ii in range(self.N):
        for jj in range(len(Z) - 1):
            bore_list.append(
                Segment(
                    Z[jj] * exp(1j * (2 * alpha1 * (ii - 1 / 2) + self.alpha)),
                    Z[jj + 1] * exp(1j * (2 * alpha1 * (ii - 1 / 2) + self.alpha)),
                    prop_dict=prop_dict,
                )
            )

    return bore_list


def _get_pole_width_max(obj):
    """Return the max. pole width and the angle of the max. pole width

    Parameters
    ----------
    obj : BoreSinePole
        a BoreSinePole object


    Returns
    -------
    w_max : float
        max. pole width

    """

    phi_max = fmin(lambda x: -obj.get_pole_shape(x).imag, 0, disp=False)[0]
    phi_max = min(pi / 2, phi_max)
    w_max = 2 * obj.get_pole_shape(phi_max).imag

    return w_max


def _get_phi(obj, w):
    """Return the angle of the pole width

    Parameters
    ----------
    obj : BoreSinePole
        a BoreSinePole object

    Returns
    -------
    phi : float
        angle of the pole width
    """

    phi = fmin(lambda x: abs(obj.get_pole_shape(x).imag - w / 2), 0, disp=False)[0]
    return phi
