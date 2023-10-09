from numpy import pi
from ....Classes.Segment import Segment
from pyleecan.Classes.Arc1 import Arc1
from pyleecan.Classes.Segment import Segment


def comp_magnetization_dict(self, is_north=True):
    """Compute the dictionary of the magnetization direction of the magnets (key=magnet_X, value=angle[rad])
    Mangetization angle with Hole centered on Ox axis

    Parameters
    ----------
    self : HoleM63
        a HoleM63 object
    is_north: True
        True: comp north magnetization, else add pi [rad]

    Returns
    -------
    mag_dict: dict
        magnetization dictionary (key=magnet_X, value=angle[rad])
    """

    # Comp magnet
    point_dict = self._comp_point_coordinate()
    Rbo = self.get_Rbo()

    mag_dict = dict()
    if self.top_flat == 0:
        S0 = Arc1(
            begin=point_dict["Z2"],
            end=point_dict["Z3"],
            radius=Rbo - self.H1,
            is_trigo_direction=True,
        )
        mag_dict["magnet_0"] = S0.comp_normal()

    elif self.top_flat == 1:
        S0 = Segment((point_dict["Z7"], point_dict["Z6"]))
        mag_dict["magnet_0"] = S0.comp_normal()

    if not is_north:
        mag_dict["magnet_0"] += pi

    if self.magnetization_dict_offset is not None:
        for key, value in self.magnetization_dict_offset:
            mag_dict[key] += value

    return mag_dict