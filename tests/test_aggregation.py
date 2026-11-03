from __future__ import annotations

import unittest

import torch

from mics.federated import ComponentUpdate, aggregate_components


class AggregationTests(unittest.TestCase):
    def test_componentwise_weighted_averaging_and_carry_forward(self) -> None:
        global_state = {
            "encoder:t1": {"weight": torch.tensor([0.0])},
            "encoder:t2": {"weight": torch.tensor([10.0])},
            "fusion": {"weight": torch.tensor([0.0])},
            "head": {"weight": torch.tensor([7.0])},
        }
        client_a = ComponentUpdate(
            client_id="a",
            states={
                "encoder:t1": {"weight": torch.tensor([2.0])},
                "fusion": {"weight": torch.tensor([2.0])},
            },
            weight={"encoder:t1": 3.0, "fusion": 3.0},
        )
        client_b = ComponentUpdate(
            client_id="b",
            states={
                "encoder:t2": {"weight": torch.tensor([4.0])},
                "fusion": {"weight": torch.tensor([8.0])},
            },
            weight={"encoder:t2": 1.0, "fusion": 1.0},
        )

        result = aggregate_components(global_state, [client_a, client_b])

        self.assertTrue(torch.equal(result["encoder:t1"]["weight"], torch.tensor([2.0])))
        self.assertTrue(torch.equal(result["encoder:t2"]["weight"], torch.tensor([4.0])))
        self.assertTrue(torch.allclose(result["fusion"]["weight"], torch.tensor([3.5])))
        self.assertTrue(torch.equal(result["head"]["weight"], torch.tensor([7.0])))

    def test_unsuccessful_update_is_ignored(self) -> None:
        global_state = {"fusion": {"weight": torch.tensor([1.0])}}
        failed = ComponentUpdate(
            client_id="failed",
            states={"fusion": {"weight": torch.tensor([9.0])}},
            successful=False,
        )
        result = aggregate_components(global_state, [failed])
        self.assertTrue(torch.equal(result["fusion"]["weight"], torch.tensor([1.0])))


if __name__ == "__main__":
    unittest.main()

