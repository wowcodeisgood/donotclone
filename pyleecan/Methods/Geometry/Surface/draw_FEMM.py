# -*- coding: utf-8 -*-

from ....Classes.Arc1 import Arc1
from ....Classes.Arc2 import Arc2
from ....Classes.Arc3 import Arc3
from ....Functions.FEMM.get_mesh_param import get_mesh_param
from ....Functions.labels import decode_label


def draw_FEMM(
    self, femm, nodeprop=None, maxseg=None, propname=None, FEMM_dict=None, hide=False
):
    """draw the Surface in FEMM

    Parameters
    ----------
    femm : FEMMHandler
        client to send command to a FEMM instance
    nodeprop :
        Nodal property
         (Default value = None)
    maxseg :
        Meshed with elements that span at most maxsegdeg degrees per element
         (Default value = None)
    propname :
        Boundary property ’propname’
         (Default value = None)
    FEMM_dict : dict
        Dictionnary containing the main parameters of FEMM
    hide :
        0 = not hidden in post-processor, 1 == hidden in post processor
        (Default value = False)

    Returns
    -------
    None

    """
    # Check if the Surface is correct
    self.check()

    # Draw all the lines
    lines = self.get_lines()

    for line in lines:
        label_dict = decode_label(self.label)  # Use surface mesh param
        if line.prop_dict not in [None, dict()]:  # Use surface mesh param
            pass
        mesh_dict = get_mesh_param(label_dict, FEMM_dict)
        if type(line) in [Arc1, Arc2, Arc3]:
            line.draw_FEMM(
                femm=femm,
                nodeprop=nodeprop,
                maxseg=maxseg,
                propname=propname,
                hide=hide,
                group=mesh_dict["group"],
            )
        else:
            line.draw_FEMM(
                femm=femm,
                nodeprop=nodeprop,
                propname=propname,
                element_size=mesh_dict["element_size"],
                automesh=mesh_dict["automesh"],
                hide=hide,
                group=mesh_dict["group"],
            )
