# -*- coding: utf-8 -*-
# File generated according to Generator/ClassesRef/Converter/Convert.csv
# WARNING! All changes made in this file will be lost!
"""Method code available at https://github.com/Eomys/pyleecan/tree/master/pyleecan/Methods/Converter/Convert
"""

from os import linesep
from sys import getsizeof
from logging import getLogger
from ._check import check_var, raise_
from ..Functions.get_logger import get_logger
from ..Functions.save import save
from ..Functions.load import load_init_dict
from ..Functions.Load.import_class import import_class
from copy import deepcopy
from ._frozen import FrozenClass

from numpy import isnan
from ._check import InitUnKnowClassError


class Convert(FrozenClass):
    """initialisation of conversion abstract"""

    VERSION = 1

    # generic save method is available in all object
    save = save
    # get_logger method is available in all object
    get_logger = get_logger

    def __init__(
        self,
        file_path="0",
        other_dict=-1,
        machine=None,
        rules_list=-1,
        is_P_to_other=False,
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
            if "file_path" in list(init_dict.keys()):
                file_path = init_dict["file_path"]
            if "other_dict" in list(init_dict.keys()):
                other_dict = init_dict["other_dict"]
            if "machine" in list(init_dict.keys()):
                machine = init_dict["machine"]
            if "rules_list" in list(init_dict.keys()):
                rules_list = init_dict["rules_list"]
            if "is_P_to_other" in list(init_dict.keys()):
                is_P_to_other = init_dict["is_P_to_other"]
        # Set the properties (value check and convertion are done in setter)
        self.parent = None
        self.file_path = file_path
        self.other_dict = other_dict
        self.machine = machine
        self.rules_list = rules_list
        self.is_P_to_other = is_P_to_other

        # The class is frozen, for now it's impossible to add new properties
        self._freeze()

    def __str__(self):
        """Convert this object in a readeable string (for print)"""

        Convert_str = ""
        if self.parent is None:
            Convert_str += "parent = None " + linesep
        else:
            Convert_str += "parent = " + str(type(self.parent)) + " object" + linesep
        Convert_str += 'file_path = "' + str(self.file_path) + '"' + linesep
        Convert_str += "other_dict = " + str(self.other_dict) + linesep
        if self.machine is not None:
            tmp = self.machine.__str__().replace(linesep, linesep + "\t").rstrip("\t")
            Convert_str += "machine = " + tmp
        else:
            Convert_str += "machine = None" + linesep + linesep
        Convert_str += (
            "rules_list = "
            + linesep
            + str(self.rules_list).replace(linesep, linesep + "\t")
            + linesep
        )
        Convert_str += "is_P_to_other = " + str(self.is_P_to_other) + linesep
        return Convert_str

    def __eq__(self, other):
        """Compare two objects (skip parent)"""

        if type(other) != type(self):
            return False
        if other.file_path != self.file_path:
            return False
        if other.other_dict != self.other_dict:
            return False
        if other.machine != self.machine:
            return False
        if other.rules_list != self.rules_list:
            return False
        if other.is_P_to_other != self.is_P_to_other:
            return False
        return True

    def compare(self, other, name="self", ignore_list=None, is_add_value=False):
        """Compare two objects and return list of differences"""

        if ignore_list is None:
            ignore_list = list()
        if type(other) != type(self):
            return ["type(" + name + ")"]
        diff_list = list()
        if other._file_path != self._file_path:
            if is_add_value:
                val_str = (
                    " (self="
                    + str(self._file_path)
                    + ", other="
                    + str(other._file_path)
                    + ")"
                )
                diff_list.append(name + ".file_path" + val_str)
            else:
                diff_list.append(name + ".file_path")
        if other._other_dict != self._other_dict:
            if is_add_value:
                val_str = (
                    " (self="
                    + str(self._other_dict)
                    + ", other="
                    + str(other._other_dict)
                    + ")"
                )
                diff_list.append(name + ".other_dict" + val_str)
            else:
                diff_list.append(name + ".other_dict")
        if (other.machine is None and self.machine is not None) or (
            other.machine is not None and self.machine is None
        ):
            diff_list.append(name + ".machine None mismatch")
        elif self.machine is not None:
            diff_list.extend(
                self.machine.compare(
                    other.machine,
                    name=name + ".machine",
                    ignore_list=ignore_list,
                    is_add_value=is_add_value,
                )
            )
        if other._rules_list != self._rules_list:
            if is_add_value:
                val_str = (
                    " (self="
                    + str(self._rules_list)
                    + ", other="
                    + str(other._rules_list)
                    + ")"
                )
                diff_list.append(name + ".rules_list" + val_str)
            else:
                diff_list.append(name + ".rules_list")
        if other._is_P_to_other != self._is_P_to_other:
            if is_add_value:
                val_str = (
                    " (self="
                    + str(self._is_P_to_other)
                    + ", other="
                    + str(other._is_P_to_other)
                    + ")"
                )
                diff_list.append(name + ".is_P_to_other" + val_str)
            else:
                diff_list.append(name + ".is_P_to_other")
        # Filter ignore differences
        diff_list = list(filter(lambda x: x not in ignore_list, diff_list))
        return diff_list

    def __sizeof__(self):
        """Return the size in memory of the object (including all subobject)"""

        S = 0  # Full size of the object
        S += getsizeof(self.file_path)
        if self.other_dict is not None:
            for key, value in self.other_dict.items():
                S += getsizeof(value) + getsizeof(key)
        S += getsizeof(self.machine)
        if self.rules_list is not None:
            for value in self.rules_list:
                S += getsizeof(value)
        S += getsizeof(self.is_P_to_other)
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

        Convert_dict = dict()
        Convert_dict["file_path"] = self.file_path
        Convert_dict["other_dict"] = (
            self.other_dict.copy() if self.other_dict is not None else None
        )
        if self.machine is None:
            Convert_dict["machine"] = None
        else:
            Convert_dict["machine"] = self.machine.as_dict(
                type_handle_ndarray=type_handle_ndarray,
                keep_function=keep_function,
                **kwargs
            )
        Convert_dict["rules_list"] = (
            self.rules_list.copy() if self.rules_list is not None else None
        )
        Convert_dict["is_P_to_other"] = self.is_P_to_other
        # The class name is added to the dict for deserialisation purpose
        Convert_dict["__class__"] = "Convert"
        return Convert_dict

    def copy(self):
        """Creates a deepcopy of the object"""

        # Handle deepcopy of all the properties
        file_path_val = self.file_path
        if self.other_dict is None:
            other_dict_val = None
        else:
            other_dict_val = self.other_dict.copy()
        if self.machine is None:
            machine_val = None
        else:
            machine_val = self.machine.copy()
        if self.rules_list is None:
            rules_list_val = None
        else:
            rules_list_val = self.rules_list.copy()
        is_P_to_other_val = self.is_P_to_other
        # Creates new object of the same type with the copied properties
        obj_copy = type(self)(
            file_path=file_path_val,
            other_dict=other_dict_val,
            machine=machine_val,
            rules_list=rules_list_val,
            is_P_to_other=is_P_to_other_val,
        )
        return obj_copy

    def _set_None(self):
        """Set all the properties to None (except pyleecan object)"""

        self.file_path = None
        self.other_dict = None
        if self.machine is not None:
            self.machine._set_None()
        self.rules_list = None
        self.is_P_to_other = None

    def _get_file_path(self):
        """getter of file_path"""
        return self._file_path

    def _set_file_path(self, value):
        """setter of file_path"""
        check_var("file_path", value, "str")
        self._file_path = value

    file_path = property(
        fget=_get_file_path,
        fset=_set_file_path,
        doc=u"""file path to conevrt

        :Type: str
        """,
    )

    def _get_other_dict(self):
        """getter of other_dict"""
        return self._other_dict

    def _set_other_dict(self, value):
        """setter of other_dict"""
        if type(value) is int and value == -1:
            value = dict()
        check_var("other_dict", value, "dict")
        self._other_dict = value

    other_dict = property(
        fget=_get_other_dict,
        fset=_set_other_dict,
        doc=u"""convertion file .mot in dict

        :Type: dict
        """,
    )

    def _get_machine(self):
        """getter of machine"""
        return self._machine

    def _set_machine(self, value):
        """setter of machine"""
        if isinstance(value, str):  # Load from file
            try:
                value = load_init_dict(value)[1]
            except Exception as e:
                self.get_logger().error(
                    "Error while loading " + value + ", setting None instead"
                )
                value = None
        if isinstance(value, dict) and "__class__" in value:
            class_obj = import_class(
                "pyleecan.Classes", value.get("__class__"), "machine"
            )
            value = class_obj(init_dict=value)
        elif type(value) is int and value == -1:  # Default constructor
            Machine = import_class("pyleecan.Classes", "Machine", "machine")
            value = Machine()
        check_var("machine", value, "Machine")
        self._machine = value

        if self._machine is not None:
            self._machine.parent = self

    machine = property(
        fget=_get_machine,
        fset=_set_machine,
        doc=u"""machine pyleecan

        :Type: Machine
        """,
    )

    def _get_rules_list(self):
        """getter of rules_list"""
        return self._rules_list

    def _set_rules_list(self, value):
        """setter of rules_list"""
        if type(value) is int and value == -1:
            value = list()
        check_var("rules_list", value, "list")
        self._rules_list = value

    rules_list = property(
        fget=_get_rules_list,
        fset=_set_rules_list,
        doc=u"""list differents rules

        :Type: list
        """,
    )

    def _get_is_P_to_other(self):
        """getter of is_P_to_other"""
        return self._is_P_to_other

    def _set_is_P_to_other(self, value):
        """setter of is_P_to_other"""
        check_var("is_P_to_other", value, "bool")
        self._is_P_to_other = value

    is_P_to_other = property(
        fget=_get_is_P_to_other,
        fset=_set_is_P_to_other,
        doc=u"""booleen to select the direction of conversion

        :Type: bool
        """,
    )