from sympy import Symbol
from sympy.solvers import solve


def convert_to_other(self, other_dict, unit_list, machine):
    # self.param_other
    # self.param_pyleecan
    # self.scaling_to_P

    scaling = self.scaling_to_P

    # replace varialble mot
    for param in self.param:
        if param["src"] == "other":
            dict_temp = other_dict
            if not param["variable"] == "y":
                for temp in param["path"]:
                    dict_temp = dict_temp[temp]

                scaling = scaling.replace(param["variable"], str(dict_temp))

    # replace variable pyleecan
    for param in self.param:
        if param["src"] == "pyleecan":
            value_split = param["path"].split(".")

            path = value_split[0]
            for temp in range(1, len(value_split) - 1):
                path = eval('path+"."+value_split[temp]')

            val_P = getattr(
                eval(path),
                value_split[-1],
            )

            scaling = scaling.replace(param["variable"], str(val_P))

    # equation cleaning, delete space and replace + and - to delete =
    scaling = scaling.replace(" ", "")
    scaling = scaling.split("=")

    equation = scaling[0] + "-(" + scaling[1] + ")"
    # result = eval(equation)

    value = solve(equation)
    value = float(value[0])

    # adding value in dict_other
    dict_temp = other_dict

    for variable in self.param:
        if variable["variable"] == "y":
            list_path = variable["path"]

            # self.set_other(other_dict, value)
            dict_temp = other_dict
            for key in list_path[:-1]:
                if key not in dict_temp:
                    dict_temp[key] = dict()
                dict_temp = dict_temp[key]
            # Set the value
            last_key = list_path[-1]
            dict_temp[last_key] = value

    return other_dict