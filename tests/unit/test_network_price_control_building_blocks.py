"""Backlog item G2 / plan item N6: the network price-control building blocks.

The extraction is only useful if it is the right table, on the right basis, for a
period that covers the modelled year. Each of those three is checked here.
"""

from pathlib import Path

import pytest

from scripts.consumer_bill.extract_network_price_control_building_blocks import (
    DNOS, ET_LICENSEES, ED2_PARAMETERS, RIIO3_ET_PARAMETERS, extract_ed2_appendix5, extract_ed2_appendix8, extract_riio3_et,
    semi_nominal_to_cpih_real, summarise,
)

ANNEX = Path("data/market/network_cost_sources/price_controls")
ED2_PDF = ANNEX / "ofgem_riio_ed2_final_determinations_finance_annex_2022_11.pdf"
RIIO3_PDF = ANNEX / "ofgem_riio3_final_determinations_finance_annex_2025_12.pdf"
pytestmark = pytest.mark.skipif(not (ED2_PDF.exists() and RIIO3_PDF.exists()),
                                reason="price-control Finance Annexes not downloaded")


class TestBasisConversion:
    def test_only_the_fixed_rate_debt_portion_carries_inflation(self):
        # Equity is already CPIH-real and index-linked debt carries no inflation,
        # so the adjustment is inflation x gearing x fixed-rate share.
        got = semi_nominal_to_cpih_real(0.0564, gearing=0.55, inflation=0.02, fixed_rate_debt_share=0.75)
        assert got == pytest.approx(0.0564 - 0.02 * 0.55 * 0.75)
        # No fixed-rate debt means nothing to remove.
        assert semi_nominal_to_cpih_real(0.0564, gearing=0.55, inflation=0.02,
                                         fixed_rate_debt_share=0.0) == pytest.approx(0.0564)
        # A semi-nominal rate is always at or above its real equivalent.
        assert semi_nominal_to_cpih_real(0.0564, gearing=0.55, inflation=0.02, fixed_rate_debt_share=1.0) < 0.0564

    @pytest.mark.parametrize("bad", [{"gearing": 1.5}, {"fixed_rate_debt_share": -0.1}])
    def test_shares_must_be_shares(self, bad):
        kwargs = {"gearing": 0.55, "inflation": 0.02, "fixed_rate_debt_share": 0.75, **bad}
        with pytest.raises(ValueError, match="share between 0 and 1"):
            semi_nominal_to_cpih_real(0.0564, **kwargs)


class TestExtraction:
    @pytest.fixture(scope="class")
    def extracted(self):
        ed2 = extract_ed2_appendix8(ED2_PDF)
        et = extract_riio3_et(RIIO3_PDF)
        return ed2, et, summarise(ed2, et)

    def test_each_dno_is_counted_once(self, extracted):
        """The annex prints every DNO table twice (Appendix 5 and the revised Appendix 8).

        Summing both doubled the distribution RAV (69.0 bn at 31 March 2028, found 6 Oct 2026);
        the 14 revised tables close at 34.46 bn (2020/21 prices).
        """
        ed2, _, summary = extracted
        assert not ed2.duplicated(["licensee", "item"]).any()
        assert summary["distribution"]["closing_rav_gbp_m_2027_28"] == pytest.approx(34464.9, abs=0.5)

    def test_summarising_both_copies_is_refused(self, extracted):
        _, et, _ = extracted
        with pytest.raises(ValueError, match="more than once"):
            summarise(extract_ed2_appendix5(ED2_PDF), et)

    def test_every_licensee_is_found(self, extracted):
        ed2, et, _ = extracted
        assert set(ed2["licensee"]) == set(DNOS)
        assert set(et["licensee"]) == set(ET_LICENSEES)

    def test_the_parse_reproduces_the_published_wacc(self, extracted):
        """Return divided by RAV must come back as the WACC Ofgem published.

        This is the check that a mis-parsed row cannot survive: the two numbers
        come from different rows of the table and only agree if both are right.
        """
        _, _, summary = extracted
        implied = summary["distribution"]["implied_vanilla_wacc_from_extract"]
        assert implied == pytest.approx(ED2_PARAMETERS["vanilla_wacc_cpih_real_infrequent_issuers"], abs=0.0015)
        assert summary["distribution"]["implied_wacc_check_passes"]

    def test_transmission_revenue_is_not_the_incentive_table_above_it(self, extracted):
        """Table 30 lists the same licensees against single-digit incentive amounts."""
        _, et, summary = extracted
        revenue = et.loc[et["item"].str.startswith("Ex ante")].set_index("licensee")["period_average"]
        assert revenue["NGET"] > 1000.0 and revenue["SHET"] > 1000.0
        assert summary["transmission"]["base_revenue_gbp_m_period_average"] > 4000.0

    def test_total_rav_is_the_equity_portion_grossed_up_by_gearing(self, extracted):
        _, _, summary = extracted
        t = summary["transmission"]
        assert t["implied_total_rav_gbp_m_period_average"] == pytest.approx(
            t["equity_portion_of_rav_gbp_m_period_average"] / (1 - RIIO3_ET_PARAMETERS["notional_gearing"]))

    def test_the_gap_in_coverage_is_stated_rather_than_papered_over(self, extracted):
        """2030 is inside RIIO-3 but past the end of ED2, and the summary must say so."""
        _, _, summary = extracted
        coverage = summary["coverage"]
        assert coverage["modelled_year"] == 2030
        assert coverage["transmission_covered"] is True
        assert coverage["distribution_covered"] is False
        assert "ED3" in coverage["distribution_note"]
        assert "CPIH-real" in summary["basis_warning"] and "semi-nominal" in summary["basis_warning"]
