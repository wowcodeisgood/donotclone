def other_to_P(self, machine):
    H1 = machine.rotor.slot.get_H1()

    machine.rotor.slot.H1 = H1
    machine.rotor.slot.H1_is_rad = False
    # machine.rotor.slot = _comp_W(machine.rotor.slot)
    return machine


def P_to_other(self, other_dict):
    print("other_to_P")
    return other_dict