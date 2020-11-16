# -*- coding: utf-8 -*-
import numpy as np

from pyleecan.Functions.make_ndarray_equal import make_ndarray_equal


def get_axis_list(self, args=None):
    """Get the axis of variables stored in Solution.

    Parameters
    ----------
    self : SolutionVector
        an SolutionVector object
    field_name : str
        name of the field to return

    Returns
    -------
    field: array
        an array of field values

    """

    # Build axis list
    ax_name = list()
    ax_size = list()

    comp = self.field.components["x"]
    for axis in comp.axes:
        ax_name.append(axis.name)
        ax_size.append(comp.get_along(axis.name)[comp.symbol].size)

    ax_name.append("component")
    ax_size.append(len(self.field.components))

    # Check that the order of the axes is correct
    field = self.get_field
    size_field = field.size

    if size_field != ax_size:
        Isort, ax_size = make_ndarray_equal(size_field, ax_size)
        ax_name = ax_name[Isort]

    return ax_name, ax_size
