from des_multi_agent.chemistry.claim_grounding import PartnerVerdict
from des_multi_agent.reporting import format_report


def _verdict(status, detail, penalty, disp):
    return PartnerVerdict(claim="partner reality: X", status=status,
                          detail=detail, penalty=penalty, disposition=disp)


def test_report_renders_partner_statuses():
    verdicts = [
        _verdict("known", "known/attested compound", 0.0, "keep"),
        _verdict("novel_plausible", "novel; complementarity=strong", 0.0, "keep"),
        _verdict("novel_implausible", "no H-bond complementarity with component A", 0.25, "demote"),
    ]
    out = format_report([], claim_verdicts=verdicts)
    assert "✓ known" in out
    assert "◆ novel (plausible)" in out
    assert "✗ implausible — no H-bond complementarity with component A" in out
