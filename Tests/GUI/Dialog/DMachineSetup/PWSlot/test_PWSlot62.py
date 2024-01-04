# -*- coding: utf-8 -*-

import sys
import mock
import pytest
from PySide2 import QtWidgets
from PySide2.QtTest import QTest

from pyleecan.Classes.LamSlotWind import LamSlotWind
from pyleecan.Classes.SlotW62 import SlotW62
from Tests.GUI import gui_option  # Set unit as [m]
from pyleecan.GUI.Dialog.DMachineSetup.SWPole.PWSlot62.PWSlot62 import PWSlot62
from pyleecan.Methods.Slot.SlotW62 import *


class TestPWSlot62(object):
    """Test that the widget PWSlot62 behave like it should"""

    @classmethod
    def setup_class(cls):
        """Start the app for the test"""
        print("\nStart Test TestPWSlot62")
        if not QtWidgets.QApplication.instance():
            cls.app = QtWidgets.QApplication(sys.argv)
        else:
            cls.app = QtWidgets.QApplication.instance()

    @classmethod
    def teardown_class(cls):
        """Exit the app after the test"""
        cls.app.quit()

    def setup_method(self):
        """Run at the begining of every test to setup the gui"""

        self.test_obj = LamSlotWind(Rint=0.1, Rext=0.2)
        self.test_obj.slot = SlotW62(
            H0=0.05,
            H1=0.01,
            H2=0.01,
            W0=0.01,
            W1=0.02,
            W2=0.005,
            H3=0.001,
            W3=0.001,
        )
        self.widget = PWSlot62(self.test_obj)

    def test_init(self):
        """Check that the Widget spinbox initialise to the lamination value"""

        assert self.widget.lf_H0.value() == 0.05
        assert self.widget.lf_H1.value() == 0.01
        assert self.widget.lf_H2.value() == 0.01
        assert self.widget.lf_W0.value() == 0.01
        assert self.widget.lf_W1.value() == 0.02
        assert self.widget.lf_W2.value() == 0.005
        assert self.widget.lf_H3.value() == 0.001
        assert self.widget.lf_W3.value() == 0.001

    def test_out(self):
        """Checking output"""
        self.test_obj = LamSlotWind(
            Rint=0, Rext=0.135, is_internal=True, is_stator=False, L1=0.9
        )
        self.test_obj.slot = SlotW62(
            Zs=12,
            H0=70e-3,
            W0=50e-3,
            H1=30e-3,
            W1=120e-3,
            W2=15e-3,
            H2=40e-3,
            W3=10e-3,
            H3=12e-3,
        )

        self.widget = PWSlot62(self.test_obj)
        assert self.widget.out_slot_height.text() == "Slot height: 0.1024 m"
        assert self.widget.out_tooth_width.isHidden()

    def test_set_W0(self):
        """Check that the Widget allow to update W0"""
        self.widget.lf_W0.clear()  # Clear the field before writing
        QTest.keyClicks(self.widget.lf_W0, "0.31")
        self.widget.lf_W0.editingFinished.emit()  # To trigger the slot

        assert self.widget.slot.W0 == 0.31
        assert self.test_obj.slot.W0 == 0.31

        self.widget.lf_W0.clear()
        QTest.keyClicks(self.widget.lf_W0, "0")
        self.widget.lf_W0.editingFinished.emit()  # To trigger the slot
        try:
            self.widget.lamination.slot.check()
            assert False
        except S62_W0Error:
            assert True

    def test_set_W1(self):
        """Check that the Widget allow to update W1"""
        self.widget.lf_W1.clear()  # Clear the field before writing
        QTest.keyClicks(self.widget.lf_W1, str(0.32))
        self.widget.lf_W1.editingFinished.emit()  # To trigger the slot

        assert self.widget.slot.W1 == 0.32
        assert self.test_obj.slot.W1 == 0.32

        self.widget.lf_W1.clear()
        QTest.keyClicks(self.widget.lf_W1, "0")
        self.widget.lf_W1.editingFinished.emit()  # To trigger the slot
        try:
            self.widget.lamination.slot.check()
            assert False
        except S62_W1Error:
            assert True

    def test_set_W2(self):
        """Check that the Widget allow to update W2"""
        self.widget.lf_W2.clear()  # Clear the field before writing
        QTest.keyClicks(self.widget.lf_W2, "0.33")
        self.widget.lf_W2.editingFinished.emit()  # To trigger the slot

        assert self.widget.slot.W2 == 0.33
        assert self.test_obj.slot.W2 == 0.33

        self.widget.lf_W2.clear()
        QTest.keyClicks(self.widget.lf_W2, "0")
        self.widget.lf_W2.editingFinished.emit()  # To trigger the slot
        try:
            self.widget.lamination.slot.check()
            assert False
        except S62_W2Error:
            assert True

    def test_set_H0(self):
        """Check that the Widget allow to update H0"""
        self.widget.lf_H0.clear()  # Clear the field before writing
        QTest.keyClicks(self.widget.lf_H0, "0.34")
        self.widget.lf_H0.editingFinished.emit()  # To trigger the slot

        assert self.widget.slot.H0 == 0.34
        assert self.test_obj.slot.H0 == 0.34

        self.widget.lf_H0.clear()
        QTest.keyClicks(self.widget.lf_H0, "0.001")
        self.widget.lf_H0.editingFinished.emit()  # To trigger the slot
        try:
            self.widget.lamination.slot.check()
            assert False
        except S62_WindHError:
            assert True

    def test_set_H1(self):
        """Check that the Widget allow to update H1"""
        self.widget.lf_H1.clear()  # Clear the field before writing
        QTest.keyClicks(self.widget.lf_H1, "0.35")
        self.widget.lf_H1.editingFinished.emit()  # To trigger the slot

        assert self.widget.slot.H1 == 0.35
        assert self.test_obj.slot.H1 == 0.35

        self.widget.lf_H1.clear()
        QTest.keyClicks(self.widget.lf_H1, "0")
        self.widget.lf_H1.editingFinished.emit()  # To trigger the slot
        try:
            self.widget.lamination.slot.check()
            assert False
        except S62_H1Error:
            assert True

    def test_set_H2(self):
        """Check that the Widget allow to update H2"""
        self.widget.lf_H2.clear()  # Clear the field before writing
        QTest.keyClicks(self.widget.lf_H2, "0.36")
        self.widget.lf_H2.editingFinished.emit()  # To trigger the slot

        assert self.widget.slot.H2 == 0.36
        assert self.test_obj.slot.H2 == 0.36

        self.widget.lf_H2.clear()
        QTest.keyClicks(self.widget.lf_H2, "0")
        self.widget.lf_H2.editingFinished.emit()  # To trigger the slot
        try:
            self.widget.lamination.slot.check()
            assert False
        except S62_H2Error:
            assert True

    def test_set_W3(self):
        """Check that the Widget allow to update W3"""
        self.widget.lf_W3.clear()  # Clear the field before writing
        QTest.keyClicks(self.widget.lf_W3, "0.37")
        self.widget.lf_W3.editingFinished.emit()  # To trigger the slot

        assert self.widget.slot.W3 == 0.37
        assert self.test_obj.slot.W3 == 0.37

    def test_set_H3(self):
        """Check that the Widget allow to update H3"""
        self.widget.lf_H3.clear()  # Clear the field before writing
        QTest.keyClicks(self.widget.lf_H3, "0.38")
        self.widget.lf_H3.editingFinished.emit()  # To trigger the slot

        assert self.widget.slot.H3 == 0.38
        assert self.test_obj.slot.H3 == 0.38

    def test_check(self):
        """Check that the check is working correctly"""

        self.test_obj = LamSlotWind(Rint=0.7, Rext=0.5)
        self.test_obj.slot = SlotW62(
            H0=None,
            H1=0.11,
            H2=0.12,
            W0=0.13,
            W1=0.14,
            W2=0.15,
            H3=0.16,
            W3=0.17,
        )
        self.widget = PWSlot62(self.test_obj)
        assert self.widget.check(self.test_obj) == "You must set H0 !"
        self.test_obj.slot = SlotW62(
            H0=0.10,
            H1=None,
            H2=0.12,
            W0=0.13,
            W1=0.14,
            W2=0.15,
            H3=0.16,
            W3=0.18,
        )
        assert self.widget.check(self.test_obj) == "You must set H1 !"
        self.test_obj.slot = SlotW62(
            H0=0.10,
            H1=0.11,
            H2=None,
            W0=0.13,
            W1=0.14,
            W2=0.15,
            H3=0.16,
            W3=0.18,
        )
        assert self.widget.check(self.test_obj) == "You must set H2 !"
        self.test_obj.slot = SlotW62(
            H0=0.10,
            H1=0.11,
            H2=0.12,
            W0=None,
            W1=0.14,
            W2=0.15,
            H3=0.16,
            W3=0.18,
        )
        assert self.widget.check(self.test_obj) == "You must set W0 !"
        self.test_obj.slot = SlotW62(
            H0=0.10,
            H1=0.11,
            H2=0.12,
            W0=0.13,
            W1=None,
            W2=0.15,
            H3=0.16,
            W3=0.18,
        )
        assert self.widget.check(self.test_obj) == "You must set W1 !"
        self.test_obj.slot = SlotW62(
            H0=0.10,
            H1=0.11,
            H2=0.12,
            W0=0.13,
            W1=0.14,
            W2=None,
            H3=0.16,
            W3=0.18,
        )
        assert self.widget.check(self.test_obj) == "You must set W2 !"
        self.test_obj.slot = SlotW62(
            H0=0.10,
            H1=0.11,
            H2=0.12,
            W0=0.13,
            W1=0.14,
            W2=0.15,
            W3=0.18,
            H3=None,
        )
        assert self.widget.check(self.test_obj) == "You must set H3 !"

        self.test_obj.slot = SlotW62(
            H0=0.10,
            H1=0.11,
            H2=0.12,
            W0=0.13,
            W1=0.14,
            W2=0.15,
            H3=0.16,
            W3=None,
        )
        assert self.widget.check(self.test_obj) == "You must set W3 !"

        self.test_obj.slot = SlotW62(
            H0=0.50,
            H1=0.11,
            H2=0.12,
            W0=0.13,
            W1=0.14,
            W2=0.12,
            H3=0.16,
            W3=0.00000158,
        )
        assert (
            self.widget.check(self.test_obj)
            == "The slot height is greater than the lamination !"
        )

        self.test_obj.slot = SlotW62(
            H0=70e-3,
            W0=50e-3,
            H1=30e-3,
            W1=120e-3,
            W2=15e-3,
            H2=60e-3,
            W3=20e-3,
            H3=12e-3,
        )
        assert self.widget.check(self.test_obj) == "You must have H3+H2 < H0"


if __name__ == "__main__":
    a = TestPWSlot62()
    a.setup_class()
    a.setup_method()
    a.test_init()
    a.teardown_class()
    print("Done")
