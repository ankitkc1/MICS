from __future__ import annotations

import unittest

import torch

from mics.models import MultiEncoderUNet3D


MODALITIES = {"t1": 1, "t1ce": 1, "t2": 1, "flair": 1}


def make_model(**kwargs: object) -> MultiEncoderUNet3D:
    return MultiEncoderUNet3D(
        MODALITIES,
        num_classes=4,
        base_channels=4,
        num_levels=3,
        **kwargs,
    )


class MultiEncoderUNet3DTests(unittest.TestCase):
    def test_forward_all_modalities_preserves_volume_shape(self) -> None:
        model = make_model()
        model.eval()
        inputs = {name: torch.randn(2, 1, 16, 16, 16) for name in MODALITIES}
        with torch.no_grad():
            logits = model(inputs)
        self.assertEqual(logits.shape, (2, 4, 16, 16, 16))

    def test_arbitrary_modalities_and_per_sample_mask(self) -> None:
        for fusion in ("mean", "mean_var", "attention"):
            with self.subTest(fusion=fusion):
                model = make_model(fusion=fusion)
                model.eval()
                inputs = {
                    "t1": torch.randn(2, 1, 16, 16, 16),
                    "flair": torch.randn(2, 1, 16, 16, 16),
                }
                mask = torch.tensor(
                    [
                        [True, False, False, False],
                        [False, False, False, True],
                    ]
                )
                with torch.no_grad():
                    output = model(inputs, mask, return_aux=True)
                self.assertEqual(output["logits"].shape, (2, 4, 16, 16, 16))
                self.assertTrue(torch.equal(output["effective_mask"], mask))

    def test_rejects_sample_without_any_modality(self) -> None:
        model = make_model()
        inputs = {"t1": torch.randn(2, 1, 16, 16, 16)}
        mask = torch.tensor(
            [
                [True, False, False, False],
                [False, False, False, False],
            ]
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            model(inputs, mask)

    def test_client_configuration_freezes_unavailable_encoders(self) -> None:
        model = make_model()
        model.configure_client_modalities({"t1", "flair"})
        self.assertEqual(
            set(model.trainable_component_names()),
            {
                "encoder:t1",
                "encoder:flair",
                "fusion",
                "decoder",
                "head",
            },
        )

        inputs = {
            "t1": torch.randn(1, 1, 16, 16, 16),
            "flair": torch.randn(1, 1, 16, 16, 16),
        }
        loss = model(inputs).square().mean()
        loss.backward()

        self.assertTrue(any(parameter.grad is not None for parameter in model.encoders["t1"].parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.encoders["flair"].parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.encoders["t1ce"].parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.encoders["t2"].parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.decoder.parameters()))

    def test_component_state_round_trip(self) -> None:
        model = make_model()
        expected = {
            "encoder:t1",
            "encoder:t1ce",
            "encoder:t2",
            "encoder:flair",
            "fusion",
            "decoder",
            "head",
        }
        state = model.component_state_dict()
        self.assertEqual(set(state), expected)
        self.assertEqual(set(model.parameter_component_map().values()), expected)
        model.load_component_state_dict(state)

    def test_modality_dropout_never_removes_every_input(self) -> None:
        model = make_model(modality_dropout=0.99)
        model.train()
        inputs = {name: torch.randn(3, 1, 16, 16, 16) for name in MODALITIES}
        output = model(inputs, return_aux=True)
        self.assertTrue(torch.all(output["effective_mask"].any(dim=1)))


if __name__ == "__main__":
    unittest.main()

