import json
import tempfile
import unittest
from pathlib import Path

from gkp_systole import (
    CyclotomicFivePolarization,
    ImaginaryQuadraticOrder,
    QuadraticHermitianForm,
    SystoleLedgerEntry,
    certify_cyclotomic_five_systole_interval,
    cyclotomic_five_moduli_family,
    quadratic_hermitian_moduli_family,
    reduce_quadratic_hermitian_form,
    write_systole_ledger,
)


class ElementaryHermitianReductionTests(unittest.TestCase):
    def test_reduction_is_idempotent_and_removes_a_shear(self):
        order = ImaginaryQuadraticOrder(-4)
        reduced = QuadraticHermitianForm(order, 2, 2, 1, 1)
        sheared = QuadraticHermitianForm(order, 2, 6, 3, 1)
        self.assertEqual(reduce_quadratic_hermitian_form(reduced), reduced)
        self.assertEqual(reduce_quadratic_hermitian_form(sheared), reduced)


class PhaseSevenModuliTests(unittest.TestCase):
    def test_new_quadratic_record_is_a_valid_search_center(self):
        form = QuadraticHermitianForm(ImaginaryQuadraticOrder(-24), 6, 6, 3, -2)
        family = quadratic_hermitian_moduli_family(form)
        self.assertEqual(family.coordinate_dimension, 6)
        self.assertEqual(family.polarization.type, (1, 3))
        self.assertAlmostEqual(
            family.evaluate((0.0,) * 6).squared_systole,
            0.8164965809277261,
        )

    def test_quartic_record_is_a_valid_search_center(self):
        polarization = CyclotomicFivePolarization(2, -1)
        family = cyclotomic_five_moduli_family(polarization)
        self.assertEqual(family.coordinate_dimension, 6)
        self.assertEqual(family.polarization.type, (1, 5))
        self.assertAlmostEqual(
            family.evaluate((0.0,) * 6).squared_systole,
            0.5505527681884692,
        )

    def test_interval_certificate_separates_quartic_minimum(self):
        certificate = certify_cyclotomic_five_systole_interval(
            CyclotomicFivePolarization(2, -1),
            decimal_places=35,
        )
        self.assertTrue(certificate.certified)
        self.assertEqual(certificate.class_multiplicity, 10)
        self.assertEqual(certificate.lift_multiplicity, 10)
        self.assertLess(float(certificate.interval_width), 1e-30)
        self.assertGreater(float(certificate.separation_gap), 0.1)
        self.assertEqual(certificate.annihilating_polynomial, (3125, 0, -1000, 0, 16))

    def test_result_ledger_round_trip(self):
        entry = SystoleLedgerEntry(
            candidate_id="test",
            phase=7,
            dimension_g=2,
            polarization_type="(1,2)",
            family="test family",
            cm_data="test CM",
            ell_squared_decimal="1.0",
            ell_squared_exact="1",
            class_multiplicity=3,
            lift_multiplicity=24,
            metric_convention="polarization_scaled",
            arithmetic_status="exact",
            search_status="test",
            search_scope="unit test",
        )
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "ledger.json"
            csv_path = Path(directory) / "ledger.csv"
            write_systole_ledger((entry,), json_path=json_path, csv_path=csv_path)
            self.assertEqual(json.loads(json_path.read_text())[0]["candidate_id"], "test")
            self.assertIn("candidate_id", csv_path.read_text())


if __name__ == "__main__":
    unittest.main()
