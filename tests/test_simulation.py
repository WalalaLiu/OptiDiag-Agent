import csv

import numpy as np

from optidiag.constants import IMAGE_TYPES
from optidiag.simulation.dataset_generator import DatasetGenerator
from optidiag.simulation.diffraction import DiffractionSimulator


def test_simulator_renders_all_image_types():
    rng = np.random.default_rng(42)
    simulator = DiffractionSimulator(image_size=96, grid_size=192)

    for image_type in IMAGE_TYPES:
        image, params = simulator.render(image_type, rng=rng)
        assert image.shape == (96, 96)
        assert image.min() >= 0.0
        assert image.max() <= 1.0
        assert params


def test_dataset_generator_writes_labels_csv(tmp_path):
    generator = DatasetGenerator(image_size=64, grid_size=128, seed=42)
    labels_path = generator.generate(tmp_path / "dataset", num_images=6)

    assert labels_path.exists()
    with labels_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 6
    assert rows[0]["image_path"].endswith(".png")
    assert rows[0]["json_path"].endswith(".json")
    assert rows[0]["primary_issue"]

