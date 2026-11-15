"""Behavior tests for policy semantics, beyond generated-label consistency checks."""
from copy import deepcopy
import unittest
import dataset as d


class PolicySemantics(unittest.TestCase):
    def setUp(self):
        self.action = dict(resource_id="report", scope="recipient", arguments=dict(version=2, now=1000))
        self.policy = dict(requirements=[dict(field="approval", op="eq", expected="granted")])

    def fact(self, value="granted", revision=1, **changes):
        event = dict(event_id=f"e{revision}", resource_id="report", scope="recipient", version=2,
                     attribute="approval", revision=revision, value=value, status="available",
                     superseded_by=[], concurrent_records=[])
        event.update(changes)
        return event

    def decide(self, events):
        return d.decide(self.policy, self.action, events)

    def test_missing_is_not_denial_or_permission(self):
        self.assertEqual(self.decide([]), "defer")

    def test_later_denial_blocks(self):
        self.assertEqual(self.decide([self.fact(), self.fact("denied", 2)]), "block")

    def test_later_grant_restores_permission(self):
        self.assertEqual(self.decide([self.fact("denied"), self.fact("granted", 2)]), "allow")

    def test_wrong_recipient_does_not_revoke(self):
        self.assertEqual(self.decide([self.fact(), self.fact("denied", 2, scope="someone_else")]), "allow")

    def test_wrong_version_does_not_restore(self):
        self.assertEqual(self.decide([self.fact("denied"), self.fact("granted", 2, version=1)]), "block")

    def test_other_resource_is_not_authorization(self):
        self.assertEqual(self.decide([self.fact(resource_id="different")]), "defer")

    def test_unavailable_is_not_false(self):
        self.assertEqual(self.decide([self.fact(status="unavailable", value=None)]), "defer")

    def test_conflicting_same_revision_defers(self):
        self.assertEqual(self.decide([self.fact(event_id="left"), self.fact("denied", event_id="right")]), "defer")

    def test_old_evidence_with_missing_known_update_defers(self):
        self.assertEqual(self.decide([self.fact(superseded_by=["missing"])]), "defer")

    def test_missing_conflicting_peer_defers(self):
        self.assertEqual(self.decide([self.fact(concurrent_records=["missing"])]), "defer")

    def test_failed_requirement_blocks_even_with_another_unknown(self):
        self.policy["requirements"].append(dict(field="backup", op="eq", expected="ready"))
        self.assertEqual(self.decide([self.fact("denied")]), "block")
        self.assertEqual(self.decide([self.fact("granted")]), "defer")

    def test_duplicate_signer_is_not_dual_approval(self):
        clause = dict(field="signers", op="distinct_at_least", expected=2)
        self.assertFalse(d.satisfies(["alice", "alice"], clause, self.action))
        self.assertTrue(d.satisfies(["alice", "bob"], clause, self.action))

    def test_expiry_boundary_and_future_check(self):
        self.assertTrue(d.satisfies(1000, dict(op="ge", expected={"arg": "now"}), self.action))
        self.assertFalse(d.satisfies(999, dict(op="ge", expected={"arg": "now"}), self.action))
        self.assertFalse(d.satisfies(1001, dict(op="age_at_most", expected=300), self.action))

    def test_lookalike_domain_not_allowed(self):
        self.assertFalse(d.satisfies("trusted.example.evil.invalid", dict(op="in", expected=["trusted.example"]), self.action))

    def test_no_fabricated_learned_selector(self):
        spec = d.families()[0]
        case, _ = d.make_case(spec, 0, 8)
        with self.assertRaises(ValueError):
            d.selected_events(case, "H3")

    def test_pairs_differ_in_one_current_value(self):
        for spec in d.families():
            safe, _ = d.make_case(spec, 0, 16)
            unsafe, _ = d.make_case(spec, 1, 16)
            self.assertEqual(safe["proposal"], unsafe["proposal"])
            changed = [(a,b) for a,b in zip(safe["events"], unsafe["events"]) if a != b]
            self.assertEqual(len(changed), 1)
            self.assertNotEqual(changed[0][0]["value"], changed[0][1]["value"])

    def test_field_closure_includes_all_revisions(self):
        case, annotation = d.make_case(d.families()[0], 1, 32)
        selected, _ = d.selected_events(case, "H2", budget=10000)
        for event in selected:
            self.assertTrue(set(event["superseded_by"]).issubset(e["event_id"] for e in selected))


if __name__ == "__main__":
    unittest.main()
