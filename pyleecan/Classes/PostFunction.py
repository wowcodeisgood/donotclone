# -*- coding: utf-8 -*-
# File generated according to Generator/ClassesRef/Post/PostFunction.csv
# WARNING! All changes made in this file will be lost!
"""Method code available at https://github.com/Eomys/pyleecan/tree/master/pyleecan/Methods/Post/PostFunction
"""

from os import linesep
from logging import getLogger
from ._check import check_var, raise_
from ..Functions.get_logger import get_logger
from ..Functions.save import save
from .Post import Post

from inspect import getsource
from cloudpickle import dumps, loads
from ._check import CheckTypeError
from ._check import InitUnKnowClassError


class PostFunction(Post):
    """Post-processing from a user-defined function"""

    VERSION = 1

    # save method is available in all object
    save = save

    # generic copy method
    def copy(self):
        """Return a copy of the class"""
        return type(self)(init_dict=self.as_dict())

    # get_logger method is available in all object
    get_logger = get_logger

    def __init__(self, run=None, init_dict=None, init_str=None):
        """Constructor of the class. Can be use in three ways :
        - __init__ (arg1 = 1, arg3 = 5) every parameters have name and default values
            for Matrix, None will initialise the property with an empty Matrix
            for pyleecan type, None will call the default constructor
        - __init__ (init_dict = d) d must be a dictionnary with every properties as keys
        - __init__ (init_str = s) s must be a string
        s is the file path to load

        ndarray or list can be given for Vector and Matrix
        object or dict can be given for pyleecan Object"""

        if init_str is not None:  # Initialisation by str
            from ..Functions.load import load

            assert type(init_str) is str
            # load the object from a file
            obj = load(init_str)
            assert type(obj) is type(self)
            run = obj.run
        if init_dict is not None:  # Initialisation by dict
            assert type(init_dict) is dict
            # Overwrite default value with init_dict content
            if "run" in list(init_dict.keys()):
                run = init_dict["run"]
        # Initialisation by argument
        self.run = run
        # Call Post init
        super(PostFunction, self).__init__()
        # The class is frozen (in Post init), for now it's impossible to
        # add new properties

    def __str__(self):
        """Convert this objet in a readeable string (for print)"""

        PostFunction_str = ""
        # Get the properties inherited from Post
        PostFunction_str += super(PostFunction, self).__str__()
        if self._run[1] is None:
            PostFunction_str += "run = " + str(self._run[1])
        else:
            PostFunction_str += (
                "run = " + linesep + str(self._run[1]) + linesep + linesep
            )
        return PostFunction_str

    def __eq__(self, other):
        """Compare two objects (skip parent)"""

        if type(other) != type(self):
            return False

        # Check the properties inherited from Post
        if not super(PostFunction, self).__eq__(other):
            return False
        if other.run != self.run:
            return False
        return True

    def as_dict(self):
        """Convert this objet in a json seriable dict (can be use in __init__)"""

        # Get the properties inherited from Post
        PostFunction_dict = super(PostFunction, self).as_dict()
        if self.run is None:
            PostFunction_dict["run"] = None
        else:
            PostFunction_dict["run"] = [
                dumps(self._run[0]).decode("ISO-8859-2"),
                self._run[1],
            ]
        # The class name is added to the dict fordeserialisation purpose
        # Overwrite the mother class name
        PostFunction_dict["__class__"] = "PostFunction"
        return PostFunction_dict

    def _set_None(self):
        """Set all the properties to None (except pyleecan object)"""

        self.run = None
        # Set to None the properties inherited from Post
        super(PostFunction, self)._set_None()

    def _get_run(self):
        """getter of run"""
        return self._run[0]

    def _set_run(self, value):
        """setter of run"""
        try:
            check_var("run", value, "list")
        except CheckTypeError:
            check_var("run", value, "function")
        if isinstance(value, list):  # Load function from saved dict
            self._run = [loads(value[0].encode("ISO-8859-2")), value[1]]
        elif value is None:
            self._run = [None, None]
        elif callable(value):
            self._run = [value, getsource(value)]
        else:
            raise TypeError(
                "Expected function or list from a saved file, got: " + str(type(value))
            )

    run = property(
        fget=_get_run,
        fset=_set_run,
        doc=u"""Post-processing that takes an output in argument

        :Type: function
        """,
    )
