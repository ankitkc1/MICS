"""Tests for reusable local-client optimization."""

from __future__ import annotations

import copy
import unittest
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from mics import MultiEncoderUNet3D
from mics.training import LocalTrainingConfig, train_local_client


class LocalTrainingTests(unittest.TestCase):
    def _loader(self) -> DataLoader:
        samples = []
        for seed in (1, 2):
            generator = torch.Generator().manual_seed(seed)
            samples.append(
                {
                    "inputs": {
                        "flair": torch.randn(
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
                    "availability": torch.tensor([True, False]),
                    "case_id": f"case-{seed}",
                }
            )
        return DataLoader(samples, batch_size=1, shuffle=False)

    def _model(self) -> MultiEncoderUNet3D:
        return MultiEncoderUNet3D(
            modalities=OrderedDict((("flair", 1), ("t1ce", 1))),
            num_classes=4,
            base_channels=2,
            num_levels=2,
            fusion="mean_var",
        )

    def test_trains_and_exports_only_updated_components(self) -> None:
        model = self._model()
        unavailable_before = {
            name: value.detach().clone()
            for name, value in model.encoders["t1ce"].state_dict().items()
        }
        result = train_local_client(
            model,
            self._loader(),
            client_id="C7",
            available_modalities=("flair",),
            device="cpu",
            config=LocalTrainingConfig(amp=False),
        )

        self.assertEqual(result.examples_seen, 2)
        self.assertEqual(result.aggregation_weight, 2.0)
        self.assertEqual(result.batches_processed, 2)
        self.assertEqual(result.optimizer_steps, 2)
        self.assertEqual(result.amp_overflow_retries, 0)
        self.assertTrue(torch.isfinite(torch.tensor(result.mean_loss)))
        self.assertGreater(result.mean_gradient_norm, 0.0)
        self.assertEqual(set(result.region_dice), {"WT", "TC", "ET"})
        self.assertEqual(
            set(result.update.states),
            {"encoder:flair", "fusion", "decoder", "head"},
        )
        for state in result.update.states.values():
            for tensor in state.values():
                self.assertEqual(tensor.device.type, "cpu")

        for name, value in model.encoders["t1ce"].state_dict().items():
            self.assertTrue(torch.equal(value, unavailable_before[name]))

    def test_multiple_epochs_do_not_multiply_aggregation_weight(self) -> None:
        result = train_local_client(
            self._model(),
            self._loader(),
            client_id="C1",
            available_modalities=("flair",),
            device="cpu",
            config=LocalTrainingConfig(local_epochs=2, amp=False),
        )

        self.assertEqual(result.examples_seen, 4)
        self.assertEqual(result.aggregation_weight, 2.0)
        self.assertEqual(result.optimizer_steps, 4)

    def test_rejects_empty_training_iterable(self) -> None:
        with self.assertRaisesRegex(ValueError, "no batches"):
            train_local_client(
                self._model(),
                [],
                client_id="C1",
                available_modalities=("flair",),
                device="cpu",
                config=LocalTrainingConfig(amp=False),
            )

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "local_epochs"):
            LocalTrainingConfig(local_epochs=0)

    def test_client_seed_isolates_modality_dropout_from_global_rng(self) -> None:
        torch.manual_seed(8)
        first_model = MultiEncoderUNet3D(
            modalities=OrderedDict((("flair", 1), ("t1ce", 1))),
            num_classes=4,
            base_channels=2,
            num_levels=2,
            fusion="mean_var",
            modality_dropout=0.5,
        )
        second_model = copy.deepcopy(first_model)
        generator = torch.Generator().manual_seed(17)
        batch = {
            "inputs": {
                name: torch.randn(
                    1,
                    1,
                    8,
                    8,
                    8,
                    generator=generator,
                )
                for name in ("flair", "t1ce")
            },
            "target": torch.randint(
                0,
                4,
                (1, 8, 8, 8),
                generator=generator,
            ),
            "availability": torch.tensor([[True, True]]),
        }
        settings = LocalTrainingConfig(amp=False, max_batches_per_epoch=1)
        first = train_local_client(
            first_model,
            [batch],
            client_id="C1",
            available_modalities=("flair", "t1ce"),
            device="cpu",
            config=settings,
            aggregation_weight=1.0,
            random_seed=1234,
        )
        torch.manual_seed(999)
        torch.rand(100)
        second = train_local_client(
            second_model,
            [batch],
            client_id="C1",
            available_modalities=("flair", "t1ce"),
            device="cpu",
            config=settings,
            aggregation_weight=1.0,
            random_seed=1234,
        )

        self.assertEqual(first.random_seed, 1234)
        self.assertEqual(first.mean_loss, second.mean_loss)
        for component, state in first.update.states.items():
            for name, value in state.items():
                self.assertTrue(
                    torch.equal(value, second.update.states[component][name])
                )


if __name__ == "__main__":
    unittest.main()
