def select_SIPMSM_rules(self):
    """select step to have rules for motor SIPMSM

    Parameters
    ----------
    self : ConvertMC
        A ConvertMC object
    """

    # step for stator
    self.select_LamSlotWind_rules(is_stator=True)

    # step for rotor
    is_stator = False
    self.select_magnet_rules(is_stator)
    self.select_lamination_rules(is_stator)
    self.select_skew_rules(is_stator)
