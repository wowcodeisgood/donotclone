from os.path import join

import numpy as np
from numpy.testing import assert_almost_equal

from pyleecan.Classes.InputCurrent import InputCurrent
from pyleecan.Classes.Simu1 import Simu1
from pyleecan.Classes.OPdq import OPdq
from pyleecan.Classes.MagFEMM import MagFEMM
from pyleecan.Classes.LossFEMM import LossFEMM
from pyleecan.Classes.OutLossFEMM import OutLossFEMM

from pyleecan.Functions.load import load

from pyleecan.definitions import DATA_DIR

from SciDataTool.Functions.Plot.plot_2D import plot_2D


is_show_fig = False


def test_FEMM_Loss_SPMSM():
    """Test to calculate losses in SPMSM using LossFEMM model"""

    machine = load(join(DATA_DIR, "Machine", "SPMSM_18s16p_loss.json"))

    Ch = 143  # hysteresis loss coefficient [W/(m^3*T^2*Hz)]
    Ce = 0.530  # eddy current loss coefficients [W/(m^3*T^2*Hz^2)]
    Cprox = 4.1018  # sigma_w * cond.Hwire * cond.Wwire

    simu = Simu1(name="test_FEMM_Loss_SPMSM", machine=machine)

    simu.input = InputCurrent(
        Nt_tot=60 * 16,
        Na_tot=1000 * 2,
        OP=OPdq(N0=4000, Id_ref=0, Iq_ref=np.sqrt(2)),
        is_periodicity_t=True,
        is_periodicity_a=True,
    )

    simu.mag = MagFEMM(
        is_periodicity_a=True,
        is_periodicity_t=True,
        nb_worker=1,
        is_get_meshsolution=True,
        FEMM_dict_enforced={
            "simu": {"minangle": 30},
            "mesh": {
                "meshsize_airgap": 0.00014,
                "elementsize_airgap": 0.00014,
                "smart_mesh": 1,
            },
        },
        # is_close_femm=False,
    )

    simu.loss = LossFEMM(Ce=Ce, Cp=Cprox, Ch=Ch, is_get_meshsolution=True, Tsta=120)

    out = simu.run()

    speed_array = np.linspace(10, 8000, 100)
    p = machine.get_pole_pair_number()
    outloss_list = list()
    for speed in speed_array:
        out_dict = {"coeff_dict": out.loss.coeff_dict}
        outloss = OutLossFEMM()
        outloss.store(out_dict, felec=speed / 60 * p)
        outloss_list.append(outloss)

    joule_list = [o.Pjoule for o in outloss_list]
    sc_list = [o.Pstator for o in outloss_list]
    rc_list = [o.Protor for o in outloss_list]
    prox_list = [o.Pprox for o in outloss_list]
    mag_list = [o.Pmagnet for o in outloss_list]
    ovl_list = [o.get_loss_overall() for o in outloss_list]

    plot_2D(
        [speed_array],
        [ovl_list, joule_list, sc_list, rc_list, prox_list, mag_list],
        xlabel="Speed [rpm]",
        ylabel="Losses [W]",
        legend_list=[
            "Overall",
            "Winding Joule",
            "Stator core",
            "Rotor core",
            "Winding proximity",
            "Magnets",
        ],
    )

    power_dict = {
        "total_power": out.mag.Pem_av,
        "overall_losses": out.loss.get_loss_overall(),
        "stator_loss": out.loss.Pstator,
        "copper_loss": out.loss.Pjoule,
        "rotor_loss": out.loss.Protor,
        "magnet_loss": out.loss.Pmagnet,
        "proximity_loss": out.loss.Pprox,
    }
    print(power_dict)

    if is_show_fig:
        out.loss.meshsolution.plot_contour(
            "freqs=sum",
            label="Loss",
            group_names=[
                "stator core",
                "stator winding",
                "rotor core",
                "rotor magnets",
            ],
            # clim=[1e4, 1e7],
        )

    # out.loss.meshsolution.plot_contour(
    #     "freqs=sum",
    #     label="Loss",
    #     group_names=["stator core", "stator winding"],
    #     # clim=[2e4, 2e7],
    # )

    # out.loss.meshsolution.plot_contour(
    #     "freqs=sum",
    #     label="Loss",
    #     group_names=["rotor core", "rotor magnets"],
    #     # clim=[2e4, 2e7],
    # )


def test_FEMM_Loss_Prius():
    """Test to calculate losses in Toyota_Prius using LossFEMM model"""

    machine = load(join(DATA_DIR, "Machine", "Toyota_Prius.json"))

    Ch = 143  # hysteresis loss coefficient [W/(m^3*T^2*Hz)]
    Ce = 0.530  # eddy current loss coefficients [W/(m^3*T^2*Hz^2)]
    Cprox = 1  # sigma_w * cond.Hwire * cond.Wwire

    simu = Simu1(name="test_FEMM_Loss_Prius", machine=machine)

    simu.input = InputCurrent(
        Nt_tot=40 * 8,
        Na_tot=200 * 8,
        OP=OPdq(N0=1000, Id_ref=-100, Iq_ref=200),
        is_periodicity_t=True,
        is_periodicity_a=True,
    )

    simu.mag = MagFEMM(
        is_periodicity_a=True,
        is_periodicity_t=True,
        nb_worker=1,
        is_get_meshsolution=True,
    )

    simu.loss = LossFEMM(Ce=Ce, Cp=Cprox, Ch=Ch, is_get_meshsolution=True, Tsta=100)

    out = simu.run()

    power_dict = {
        "total_power": out.mag.Pem_av,
        "overall_losses": out.loss.get_loss_overall(),
        "stator_loss": out.loss.Pstator,
        "copper_loss": out.loss.Pjoule,
        "rotor_loss": out.loss.Protor,
        "magnet_loss": out.loss.Pmagnet,
        "proximity_loss": out.loss.Pprox,
    }
    print(power_dict)

    if is_show_fig:
        out.loss.meshsolution.plot_contour(
            "freqs=sum",
            label="Loss",
            group_names=[
                "stator core",
                "stator winding",
                "rotor core",
                "rotor magnets",
            ],
            # clim=[2e4, 2e7],
        )

    # out.loss.meshsolution.plot_contour(
    #     "freqs=sum",
    #     label="Loss",
    #     group_names=["stator core", "stator winding"],
    #     # clim=[2e4, 2e7],
    # )

    # out.loss.meshsolution.plot_contour(
    #     "freqs=sum",
    #     label="Loss",
    #     group_names=["rotor core", "rotor magnets"],
    #     # clim=[2e4, 2e7],
    # )


# To run it without pytest
if __name__ == "__main__":

    # out = test_FEMM_Loss_SPMSM()

    out = test_FEMM_Loss_Prius()
