from numpy.lib.shape_base import expand_dims
from ....Methods.Simulation.Input import InputError
from ....Classes.EEC_SCIM import EEC_SCIM
from ....Classes.EEC_PMSM import EEC_PMSM
from ....Classes.MachineSCIM import MachineSCIM
from ....Classes.MachineSIPMSM import MachineSIPMSM
from ....Classes.MachineIPMSM import MachineIPMSM


def run(self):
    """Run the Electrical module"""
    if self.parent is None:
        raise InputError("The Electrical object must be in a Simulation object to run")
    if self.parent.parent is None:
        raise InputError("The Simulation object must be in an Output object to run")

    self.get_logger().info("Starting Electric module")

    output = self.parent.parent

    machine = output.simu.machine

    if self.eec is None:
        # Init EEC depending on machine type
        if isinstance(machine, MachineSCIM):
            self.eec = EEC_SCIM()
        elif isinstance(machine, (MachineSIPMSM, MachineIPMSM)):
            self.eec = EEC_PMSM()
    else:
        # Check that EEC is consistent with machine type
        if isinstance(machine, MachineSCIM) and not isinstance(self.eec, EEC_SCIM):
            raise Exception(
                "Cannot run Electrical model if machine is SCIM and eec is not EEC_SCIM"
            )
        elif isinstance(machine, (MachineSIPMSM, MachineIPMSM)) and not isinstance(
            self.eec, EEC_PMSM
        ):
            raise Exception(
                "Cannot run Electrical model if machine is PMSM and eec is not EEC_PMSM"
            )

    if self.ELUT_enforced is not None:
        # enforce parameters of EEC coming from enforced ELUT at right temperatures
        if self.eec.parameters is None:
            self.eec.parameters = dict()
        self.eec.parameters.update(self.ELUT_enforced.get_param_dict())

    # Generate drive
    # self.eec.gen_drive(output)
    # self.eec.parameters["U0_ref"] = output.elec.U0_ref
    # self.eec.parameters["Ud_ref"] = output.elec.OP.get_Ud_Uq()["Ud"]
    # self.eec.parameters["Uq_ref"] = output.elec.OP.get_Ud_Uq()["Uq"]

    # Ud_ref, Ud_ref, fe => OP_fund
    # for harm
    #     Ud, Uq, Fharm => Op_harm
    # Compute parameters of the electrical equivalent circuit if some parameters are missing in ELUT
    out_dict = self.eec.comp_parameters(
        machine,
        OP=output.elec.OP,
        Tsta=self.Tsta,
        Trot=self.Trot,
    )

    # Solve the electrical equivalent circuit
    out_dict = self.eec.solve_EEC(output)

    # Solve for each harmonic in case of Us_harm
    if output.elec.Us_harm is not None:
        freqs = output.elec.Us_harm.get_axes("freqs")[0].get_values().tolist()
        for f in freqs:
            out_dict_harm = self.eec.solve_EEC(output)
    else:
        out_dict_harm = None

    # Compute losses due to Joule effects
    out_dict = self.eec.comp_joule_losses(out_dict, machine)

    # Compute electromagnetic power
    out_dict = self.comp_power(out_dict, machine)

    # Compute torque
    self.comp_torque(out_dict, output.elec.OP.get_N0())

    # Store electrical quantities contained in out_dict in OutElec, as Data object if necessary
    output.elec.store(out_dict, out_dict_harm)
