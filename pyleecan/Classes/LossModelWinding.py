# -*- coding: utf-8 -*-
# File generated according to Generator/ClassesRef/Simulation/LossModelWinding.csv
# WARNING! All changes made in this file will be lost!
"""Method code available at https://github.com/Eomys/pyleecan/tree/master/pyleecan/Methods/Simulation/LossModelWinding
"""

from os import linesep
from sys import getsizeof
from logging import getLogger
from ._check import check_var, raise_
from ..Functions.get_logger import get_logger
from ..Functions.save import save
from ..Functions.copy import copy
from ..Functions.load import load_init_dict
from ..Functions.Load.import_class import import_class
from .LossModel import LossModel

# Import all class method
# Try/catch to remove unnecessary dependencies in unused method
try:
    from ..Methods.Simulation.LossModelWinding.comp_loss import comp_loss
except ImportError as error:
    comp_loss = error

try:
    from ..Methods.Simulation.LossModelWinding.comp_coeff import comp_coeff
except ImportError as error:
    comp_coeff = error


from numpy import isnan
from ._check import InitUnKnowClassError


class LossModelWinding(LossModel):
    """Winding loss model"""

    VERSION = 1

    # Check ImportError to remove unnecessary dependencies in unused method
    # cf Methods.Simulation.LossModelWinding.comp_loss
    if isinstance(comp_loss, ImportError):
        comp_loss = property(
            fget=lambda x: raise_(
                ImportError(
                    "Can't use LossModelWinding method comp_loss: " + str(comp_loss)
                )
            )
        )
    else:
        comp_loss = comp_loss
    # cf Methods.Simulation.LossModelWinding.comp_coeff
    if isinstance(comp_coeff, ImportError):
        comp_coeff = property(
            fget=lambda x: raise_(
                ImportError(
                    "Can't use LossModelWinding method comp_coeff: " + str(comp_coeff)
                )
            )
        )
    else:
        comp_coeff = comp_coeff
    # save and copy methods are available in all object
    save = save
    copy = copy
    # get_logger method is available in all object
    get_logger = get_logger

    def __init__(
        self,
        temperature=20,
        type_skin_effect=1,
        name="",
        group="",
        is_show_fig=False,
        coeff_dict=None,
        init_dict=None,
        init_str=None,
    ):
        """Constructor of the class. Can be use in three ways :
        - __init__ (arg1 = 1, arg3 = 5) every parameters have name and default values
            for pyleecan type, -1 will call the default constructor
        - __init__ (init_dict = d) d must be a dictionary with property names as keys
        - __init__ (init_str = s) s must be a string
        s is the file path to load

        ndarray or list can be given for Vector and Matrix
        object or dict can be given for pyleecan Object"""

        if init_str is not None:  # Load from a file
            init_dict = load_init_dict(init_str)[1]
        if init_dict is not None:  # Initialisation by dict
            assert type(init_dict) is dict
            # Overwrite default value with init_dict content
            if "temperature" in list(init_dict.keys()):
                temperature = init_dict["temperature"]
            if "type_skin_effect" in list(init_dict.keys()):
                type_skin_effect = init_dict["type_skin_effect"]
            if "name" in list(init_dict.keys()):
                name = init_dict["name"]
            if "group" in list(init_dict.keys()):
                group = init_dict["group"]
            if "is_show_fig" in list(init_dict.keys()):
                is_show_fig = init_dict["is_show_fig"]
            if "coeff_dict" in list(init_dict.keys()):
                coeff_dict = init_dict["coeff_dict"]
        # Set the properties (value check and convertion are done in setter)
        self.temperature = temperature
        self.type_skin_effect = type_skin_effect
        # Call LossModel init
        super(LossModelWinding, self).__init__(
            name=name, group=group, is_show_fig=is_show_fig, coeff_dict=coeff_dict
        )
        # The class is frozen (in LossModel init), for now it's impossible to
        # add new properties

    def __str__(self):
        """Convert this object in a readeable string (for print)"""

        LossModelWinding_str = ""
        # Get the properties inherited from LossModel
        LossModelWinding_str += super(LossModelWinding, self).__str__()
        LossModelWinding_str += "temperature = " + str(self.temperature) + linesep
        LossModelWinding_str += (
            "type_skin_effect = " + str(self.type_skin_effect) + linesep
        )
        return LossModelWinding_str

    def __eq__(self, other):
        """Compare two objects (skip parent)"""

        if type(other) != type(self):
            return False

        # Check the properties inherited from LossModel
        if not super(LossModelWinding, self).__eq__(other):
            return False
        if other.temperature != self.temperature:
            return False
        if other.type_skin_effect != self.type_skin_effect:
            return False
        return True

    def compare(self, other, name="self", ignore_list=None, is_add_value=False):
        """Compare two objects and return list of differences"""

        if ignore_list is None:
            ignore_list = list()
        if type(other) != type(self):
            return ["type(" + name + ")"]
        diff_list = list()

        # Check the properties inherited from LossModel
        diff_list.extend(
            super(LossModelWinding, self).compare(
                other, name=name, ignore_list=ignore_list, is_add_value=is_add_value
            )
        )
        if (
            other._temperature is not None
            and self._temperature is not None
            and isnan(other._temperature)
            and isnan(self._temperature)
        ):
            pass
        elif other._temperature != self._temperature:
            if is_add_value:
                val_str = (
                    " (self="
                    + str(self._temperature)
                    + ", other="
                    + str(other._temperature)
                    + ")"
                )
                diff_list.append(name + ".temperature" + val_str)
            else:
                diff_list.append(name + ".temperature")
        if other._type_skin_effect != self._type_skin_effect:
            if is_add_value:
                val_str = (
                    " (self="
                    + str(self._type_skin_effect)
                    + ", other="
                    + str(other._type_skin_effect)
                    + ")"
                )
                diff_list.append(name + ".type_skin_effect" + val_str)
            else:
                diff_list.append(name + ".type_skin_effect")
        # Filter ignore differences
        diff_list = list(filter(lambda x: x not in ignore_list, diff_list))
        return diff_list

    def __sizeof__(self):
        """Return the size in memory of the object (including all subobject)"""

        S = 0  # Full size of the object

        # Get size of the properties inherited from LossModel
        S += super(LossModelWinding, self).__sizeof__()
        S += getsizeof(self.temperature)
        S += getsizeof(self.type_skin_effect)
        return S

    def as_dict(self, type_handle_ndarray=0, keep_function=False, **kwargs):
        """
        Convert this object in a json serializable dict (can be use in __init__).
        type_handle_ndarray: int
            How to handle ndarray (0: tolist, 1: copy, 2: nothing)
        keep_function : bool
            True to keep the function object, else return str
        Optional keyword input parameter is for internal use only
        and may prevent json serializability.
        """

        # Get the properties inherited from LossModel
        LossModelWinding_dict = super(LossModelWinding, self).as_dict(
            type_handle_ndarray=type_handle_ndarray,
            keep_function=keep_function,
            **kwargs
        )
        LossModelWinding_dict["temperature"] = self.temperature
        LossModelWinding_dict["type_skin_effect"] = self.type_skin_effect
        # The class name is added to the dict for deserialisation purpose
        # Overwrite the mother class name
        LossModelWinding_dict["__class__"] = "LossModelWinding"
        return LossModelWinding_dict

    def _set_None(self):
        """Set all the properties to None (except pyleecan object)"""

        self.temperature = None
        self.type_skin_effect = None
        # Set to None the properties inherited from LossModel
        super(LossModelWinding, self)._set_None()

    def _get_temperature(self):
        """getter of temperature"""
        return self._temperature

    def _set_temperature(self, value):
        """setter of temperature"""
        check_var("temperature", value, "float")
        self._temperature = value

    temperature = property(
        fget=_get_temperature,
        fset=_set_temperature,
        doc=u"""Winding temperature [°C]

        :Type: float
        """,
    )

    def _get_type_skin_effect(self):
        """getter of type_skin_effect"""
        return self._type_skin_effect

    def _set_type_skin_effect(self, value):
        """setter of type_skin_effect"""
        check_var("type_skin_effect", value, "int")
        self._type_skin_effect = value

    type_skin_effect = property(
        fget=_get_type_skin_effect,
        fset=_set_type_skin_effect,
        doc=u"""0 to ignore skin effect, 1 to consider it

        :Type: int
        """,
    )
