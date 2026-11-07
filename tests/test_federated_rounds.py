"""Tests for sequential synchronous federated rounds."""

from __future__ import annotations

import unittest
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from mics import MultiEncoderUNet3D
from mics.federated import ClientRoundSpec, run_federated_round
from mics.training import LocalTrainingConfig


def _loader(modality: str, availability: tuple[bool, bool], seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    sample = {
        "inputs": {
            modality: torch.randn(
                1,
                8,
                8,
                8,
                generator=generator,
            )
        },
        "target": torch.randint(
            0,
            4,
            (8, 8, 8),
            generator=generator,
        ),
        "availability": torch.tensor(availability),
        "case_id": f"case-{seed}",
    }
    return DataLoader([sample], batch_size=1, shuffle=False)


def _floating_distance(
    before: dict[str, dict[str, torch.Tensor]],
    after: dict[str, dict[str, torch.Tensor]],
) -> float:
    total = 0.0
    for component, state in before.items():
        for name, value in state.items():
            if value.is_floating_point():
                delta = after[component][name] - value
                total += float(delta.square().sum())
    return total**0.5


class FederatedRoundTests(unittest.TestCase):
    def test_two_clients_train_from_global_and_aggregate_for_two_rounds(self) -> None:
        model = MultiEncoderUNet3D(
            modalities=OrderedDict((("flair", 1), ("t1ce", 1))),
            num_classes=4,
            base_channels=2,
            num_levels=2,
            fusion="mean_var",
        )
        clients = (
            ClientRoundSpec(
                client_id="C1",
                modalities=("flair",),
                batches=_loader("flair", (True, False), 1),
                aggregation_weight=3.0,
            ),
            ClientRoundSpec(
                client_id="C6",
                modalities=("t1ce",),
                batches=_loader("t1ce", (False, True), 2),
                aggregation_weight=5.0,
            ),
        )
        config = LocalTrainingConfig(
            amp=False,
            max_batches_per_epoch=1,
        )

        initial = model.component_state_dict(clone=True)
        first = run_federated_round(
            model,
            clients,
            round_index=0,
            device="cpu",
            training_config=config,
        )
        after_first = model.component_state_dict(clone=True)
        second = run_federated_round(
            model,
            clients,
            round_index=1,
            device="cpu",
            training_config=config,
        )
        after_second = model.component_state_dict(clone=True)

        expected_counts = {
            "encoder:flair": 1,
            "encoder:t1ce": 1,
            "fusion": 2,
            "decoder": 2,
            "head": 2,
        }
        self.assertEqual(first.component_update_counts, expected_counts)
        self.assertEqual(second.component_update_counts, expected_counts)
        self.assertEqual(first.selected_client_ids, ("C1", "C6"))
        self.assertEqual(second.selected_client_ids, ("C1", "C6"))
        self.assertEqual(first.successful_client_ids, ("C1", "C6"))
        self.assertGreater(_floating_distance(initial, after_first), 0.0)
        self.assertGreater(_floating_distance(after_first, after_second), 0.0)
        for state in after_second.values():
            for tensor in state.values():
                self.assertEqual(tensor.device.type, "cpu")

    def test_rejects_duplicate_selected_clients(self) -> None:
        model = MultiEncoderUNet3D(
            modalities=OrderedDict((("flair", 1), ("t1ce", 1))),
            num_classes=4,
            base_channels=2,
            num_levels=2,
        )
        client = ClientRoundSpec(
            client_id="C1",
            modalities=("flair",),
            batches=_loader("flair", (True, False), 1),
            aggregation_weight=1.0,
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            run_federated_round(
                model,
                (client, client),
                round_index=0,
                device="cpu",
                training_config=LocalTrainingConfig(amp=False),
            )

    def test_all_failed_attempts_carry_the_global_model_forward(self) -> None:
        model = MultiEncoderUNet3D(
            modalities=OrderedDict((("flair", 1), ("t1ce", 1))),
            num_classes=4,
            base_channels=2,
            num_levels=2,
        )
        before = model.component_state_dict(clone=True)

        result = run_federated_round(
            model,
            (),
            attempted_client_ids=("C1", "C6"),
            round_index=0,
            device="cpu",
            training_config=LocalTrainingConfig(amp=False),
        )
        after = model.component_state_dict(clone=True)

        self.assertEqual(result.selected_client_ids, ("C1", "C6"))
        self.assertEqual(result.successful_client_ids, ())
        self.assertEqual(result.examples_seen, 0)
        self.assertTrue(
            all(value == 0 for value in result.component_update_counts.values())
        )
        self.assertEqual(_floating_distance(before, after), 0.0)


if __name__ == "__main__":
    unittest.main()
