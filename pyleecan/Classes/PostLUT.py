# -*- coding: utf-8 -*-
# File generated according to Generator/ClassesRef/Post/PostLUT.csv
# WARNING! All changes made in this file will be lost!
"""Method code available at https://github.com/Eomys/pyleecan/tree/master/pyleecan/Methods/Post/PostLUT
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
from .PostMethod import PostMethod

# Import all class method
# Try/catch to remove unnecessary dependencies in unused method
try:
    from ..Methods.Post.PostLUT.run import run
except ImportError as error:
    run = error


from ._check import InitUnKnowClassError


class PostLUT(PostMethod):
    """Class to generate a LUT after the corresponding simulation"""

    VERSION = 1

    # cf Methods.Post.PostLUT.run
    if isinstance(run, ImportError):
        run = property(
            fget=lambda x: raise_(
                ImportError("Can't use PostLUT method run: " + str(run))
            )
        )
    else:
        run = run
    # save and copy methods are available in all object
    save = save
    copy = copy
    # get_logger method is available in all object
    get_logger = get_logger

    def __init__(self, is_save_LUT=True, init_dict=None, init_str=None):
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
            if "is_save_LUT" in list(init_dict.keys()):
                is_save_LUT = init_dict["is_save_LUT"]
        # Set the properties (value check and convertion are done in setter)
        self.is_save_LUT = is_save_LUT
        # Call PostMethod init
        super(PostLUT, self).__init__()
        # The class is frozen (in PostMethod init), for now it's impossible to
        # add new properties

    def __str__(self):
        """Convert this object in a readeable string (for print)"""

        PostLUT_str = ""
        # Get the properties inherited from PostMethod
        PostLUT_str += super(PostLUT, self).__str__()
        PostLUT_str += "is_save_LUT = " + str(self.is_save_LUT) + linesep
        return PostLUT_str

    def __eq__(self, other):
        """Compare two objects (skip parent)"""

        if type(other) != type(self):
            return False

        # Check the properties inherited from PostMethod
        if not super(PostLUT, self).__eq__(other):
            return False
        if other.is_save_LUT != self.is_save_LUT:
            return False
        return True

    def compare(self, other, name="self", ignore_list=None):
        """Compare two objects and return list of differences"""

        if ignore_list is None:
            ignore_list = list()
        if type(other) != type(self):
            return ["type(" + name + ")"]
        diff_list = list()

        # Check the properties inherited from PostMethod
        diff_list.extend(super(PostLUT, self).compare(other, name=name))
        if other._is_save_LUT != self._is_save_LUT:
            diff_list.append(name + ".is_save_LUT")
        # Filter ignore differences
        diff_list = list(filter(lambda x: x not in ignore_list, diff_list))
        return diff_list

    def __sizeof__(self):
        """Return the size in memory of the object (including all subobject)"""

        S = 0  # Full size of the object

        # Get size of the properties inherited from PostMethod
        S += super(PostLUT, self).__sizeof__()
        S += getsizeof(self.is_save_LUT)
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

        # Get the properties inherited from PostMethod
        PostLUT_dict = super(PostLUT, self).as_dict(
            type_handle_ndarray=type_handle_ndarray,
            keep_function=keep_function,
            **kwargs
        )
        PostLUT_dict["is_save_LUT"] = self.is_save_LUT
        # The class name is added to the dict for deserialisation purpose
        # Overwrite the mother class name
        PostLUT_dict["__class__"] = "PostLUT"
        return PostLUT_dict

    def _set_None(self):
        """Set all the properties to None (except pyleecan object)"""

        self.is_save_LUT = None
        # Set to None the properties inherited from PostMethod
        super(PostLUT, self)._set_None()

    def _get_is_save_LUT(self):
        """getter of is_save_LUT"""
        return self._is_save_LUT

    def _set_is_save_LUT(self, value):
        """setter of is_save_LUT"""
        check_var("is_save_LUT", value, "bool")
        self._is_save_LUT = value

    is_save_LUT = property(
        fget=_get_is_save_LUT,
        fset=_set_is_save_LUT,
        doc=u"""True to save LUT in PostLUT

        :Type: bool
        """,
    )
