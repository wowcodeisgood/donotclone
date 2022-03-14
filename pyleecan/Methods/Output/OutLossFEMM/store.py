from numpy import zeros

from SciDataTool import DataFreq

from ....Classes.SolutionData import SolutionData
from ....Classes.MeshSolution import MeshSolution


def store(self, out_dict, axes_dict, is_get_meshsolution=False):
    """Store the outputs of LossFEMM model that are temporarily in out_dict

    Parameters
    ----------
    self : OutLossFEMM
        the OutLossFEMM object to update
    out_dict : dict
        Dict containing all losses quantities that have been calculated in comp_losses
    axes_dict : dict
        Dict containing axes for loss calculation
    is_get_meshsolution: bool
        True to store meshsolution of loss density
    """

    # Store dict of axes
    if axes_dict is not None:
        self.axes_dict = axes_dict

    # Store scalar losses
    if "Pstator" in out_dict:
        self.Pstator = out_dict["Pstator"]
    if "Protor" in out_dict:
        self.Protor = out_dict["Protor"]
    if "Pjoule" in out_dict:
        self.Pjoule = out_dict["Pjoule"]
    if "Pprox" in out_dict:
        self.Pprox = out_dict["Pprox"]
    if "Pmagnet" in out_dict:
        self.Pmagnet = out_dict["Pmagnet"]

    # Store loss density as meshsolution
    if is_get_meshsolution:

        meshsol = self.parent.mag.meshsolution

        Nfreq = axes_dict["freqs"].get_values().size
        Nelem = meshsol.mesh[0].cell["triangle"].nb_cell

        loss_density = zeros((Nfreq, Nelem))

        if "Pstator_density" in out_dict:
            loss_density[:, meshsol.group["stator core"]] += out_dict["Pstator_density"]
        if "Protor_density" in out_dict:
            loss_density[:, meshsol.group["rotor core"]] += out_dict["Protor_density"]
        if "Pjoule_density" in out_dict:
            loss_density[:, meshsol.group["stator winding"]] += out_dict[
                "Pjoule_density"
            ]
        if "Pprox_density" in out_dict:
            loss_density[:, meshsol.group["stator winding"]] += out_dict[
                "Pprox_density"
            ]
        if "Pmagnet_density" in out_dict:
            loss_density[:, meshsol.group["rotor magnets"]] += out_dict[
                "Pmagnet_density"
            ]

        Loss_density_df = DataFreq(
            name="Loss density",
            unit="W/m3",
            symbol="L",
            values=loss_density,
            is_real=True,
            axes=[axes_dict["freqs"], axes_dict["indice"]],
        )

        Loss_density_sd = SolutionData(
            label=Loss_density_df.name, field=Loss_density_df, unit=Loss_density_df.unit
        )

        self.meshsolution = MeshSolution(
            label=Loss_density_sd.label,
            group=meshsol.group,
            is_same_mesh=True,
            mesh=meshsol.mesh,
            solution=[Loss_density_sd],
        )
