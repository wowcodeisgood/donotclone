import pytest
from numpy import pi

from pyleecan.Classes.RuleComplex import RuleComplex
from pyleecan.Classes.MachineSIPMSM import MachineSIPMSM
from pyleecan.Classes.LamSlotMag import LamSlotMag
from pyleecan.Classes.SlotM12 import SlotM12


class TestComplexRuleSlotM12(object):
    def test_inset_breadloaf_slotM12(self):
        """test rule complex"""

        other_dict = {
            "[Dimensions]": {
                "Magnet_Thickness": 4,
                "Magnet_Arc_[ED]": 120,
            }
        }

        machine = MachineSIPMSM()
        machine.rotor = LamSlotMag()
        machine.rotor.slot = SlotM12()

        rule = RuleComplex(fct_name="inset_breadloaf_slotM12", folder="MotorCAD")
        # first rule complex use to define a slot
        machine = rule.convert_to_P(
            other_dict, machine, {"ED": (2 / 8) * (pi / 180), "m": 0.001}
        )

        assert machine.rotor.slot.W1 == pytest.approx(0.5155675378442213)
        assert machine.rotor.slot.W0 == pytest.approx(0.5155675378442213)


if __name__ == "__main__":
    a = TestComplexRuleSlotM12()
    a.test_inset_breadloaf_slotM12()
    print("Test Done")