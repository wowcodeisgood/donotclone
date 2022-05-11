from numpy import matmul, abs as np_abs, sum as np_sum, sqrt as np_sqrt


def comp_loss(self):
    """Calculate loss density in iron core given by group "stator core" or "rotor core"
    assuming power density is given by a Steinmetz model

        Pcore = Ph + Pe = k_hy * f^alpha_f * B^self.alpha_B + k_ed * f^2 * B^2

    Parameters
    ----------
    self: LossFEMM
        a LossFEMM object
    group: str
        Name of part in which to calculate core losses
    coeff_dict: dict
        Dict containing coefficient A, B, C, a, b, c to calculate overall losses
        such as P = A * felec^a + B * felec^b + C * felec^c

    Returns
    -------
    Pcore_density : ndarray
        Core loss density function of frequency and elements [W/m3]
    freqs: ndarray
        frequency vector [Hz]
    """

    if self.parent.parent.parent is None:
        raise Exception("Cannot calculate core losses if simu is not in an Output")
    else:
        output = self.parent.parent.parent

    machine = output.simu.machine

    per_a = output.geo.per_a
    if output.geo.is_antiper_a:
        per_a *= 2

    lamination = machine.stator if "stator" in self.group else machine.rotor
    Lst = lamination.L1
    # Taking into account the stacking factor
    Kf = lamination.Kf1
    rho = lamination.mat_type.struct.rho

    if None in [
        self.k_hy,
        self.k_ed,
        self.alpha_f,
        self.alpha_B,
    ]:
        material = lamination.mat_type
        self.comp_coeff(material)

    # Get hysteresis and eddy current loss coefficients
    k_hy = self.k_hy / Kf * rho
    k_ed = self.k_ed / Kf * rho
    alpha_f = self.alpha_f
    alpha_B = self.alpha_B

    # Get fundamental frequency
    felec = output.elec.OP.get_felec()

    if output.mag is None:
        raise Exception("Cannot calculate core losses if OutMag is None")

    if output.mag.meshsolution is None:
        raise Exception("Cannot calculate core losses if OutMag.meshsolution is None")
    else:
        meshsol = output.mag.meshsolution

    group_list = list(meshsol.group.keys())

    if self.group not in group_list:
        raise Exception("Cannot calculate core losses for group=" + self.group)

    label_list = [sol.label for sol in meshsol.solution]

    if "B" not in label_list:
        raise Exception("Cannot calculate core losses if B is not in meshsolution")
    else:
        ind = label_list.index("B")

    # Get element indices associated to group
    Igrp = meshsol.group[self.group]

    # Get element surface associated to group
    Se = meshsol.mesh[0].get_cell_area()[Igrp]

    Bvect = meshsol.solution[ind].field
    axes_list = Bvect.get_axes()
    Time_orig = axes_list[0]
    Time = Time_orig.copy()

    # Check Time axis periodicity in function of group
    is_change_Time = False
    if "rotor" in self.group:
        if "antiperiod" in Time_orig.symmetries:
            Time.symmetries = {"period": Time_orig.symmetries["antiperiod"]}
            is_change_Time = True
    if is_change_Time:
        for comp in Bvect.components.values():
            comp.axes[0] = Time

    # # Plot 2D to check periodicity
    # # ii = Igrp[0]
    # ii = 1560
    # Bvect.components["comp_x"].plot_2D_Data(
    #     "time",
    #     "indice[" + str(ii) + "]",
    #     data_list=[Bvect.components["comp_y"]],
    #     legend_list=["Bx", "By"],
    # )

    # Bvect.components["comp_x"].plot_2D_Data(
    #     "freqs",
    #     "indice[" + str(ii) + "]",
    #     data_list=[Bvect.components["comp_y"]],
    #     legend_list=["Bx", "By"],
    # )

    # Compute magnetic flux density FFT
    Bfft = Bvect.get_xyz_along("freqs", "indice=" + str(Igrp), "z[0]")
    freqs = Bfft["freqs"]

    # Compute FFT square of magnetic flux density
    Bfft_magnitude = np_sqrt(np_abs(Bfft["comp_x"]) ** 2 + np_abs(Bfft["comp_y"]) ** 2)

    # Eddy-current loss density (or proximity loss density) for each frequency and element
    Pcore_density = k_ed * freqs[:, None] ** 2 * Bfft_magnitude ** 2

    if is_change_Time:
        # Change periodicity back to original periodicity
        for comp in Bvect.components.values():
            comp.axes[0] = Time_orig

    # Integrate loss density over group volume
    coeff = Lst * per_a * matmul(Bfft_magnitude ** 2, Se)
    # Get frequency orders
    n = freqs / felec
    # Get polynomial coefficients
    A = np_sum(k_ed * coeff * n ** 2)
    if k_hy == 0:
        B = 0
        alpha_f = 0
    else:
        coeff = Lst * per_a * matmul(Bfft_magnitude ** alpha_B, Se)
        B = np_sum(k_hy * coeff * n ** alpha_f)
    self.coeff_dict = {"A": A, "B": B, "C": 0, "a": 2, "b": alpha_f, "c": 0}

    with open("loss_models.txt", "w") as f:
        f.write(
            f"""
k_hy: {self.k_hy}
k_ed: {self.k_ed}
alpha_f: {self.alpha_f}
alpha_B: {self.alpha_B}
coeff: {coeff}
A: {A}
B: {B}
Lst: {Lst}
Igrp: {Igrp}
freqs: {freqs}
"""
        )

    return Pcore_density, freqs