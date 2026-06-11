# Legacy Reference Code

The original `prompt/` directory in the workspace is preserved as legacy reference code and is not moved or deleted.

Main files:

- `prompt/DiffractionSimulator.py`: Fraunhofer diffraction simulation, legacy augmentation, and dataset generation ideas.
- `prompt/train.py`: ResNet18/34 classification training for six aperture classes.
- `prompt/flask_server.py`: Flask `/predict` classification server.
- `prompt/re_ui.py`: PyQt inference UI that calls a local HTTP service.
- `prompt/fulanghefei.m`: MATLAB interactive diffraction simulator.
- `prompt/best_model.pth`: legacy trained classification weights.

The MVP in `OptiDiag-Agent/` borrows the general workflow but changes the task from simple aperture classification to rule-first optical experiment image diagnosis:

- image type recognition
- acquisition issue/noise scoring
- explainable image metrics
- Chinese operating suggestions
- FastAPI `/analyze` endpoint for agent plugin integration

Known legacy mismatch:

- The old README mentions `image_url`.
- The old Flask server actually rejects `image_url` and only accepts multipart file upload.

The new API fixes this: `/analyze` supports both `file` and `image_url`, and `file` takes precedence when both are provided.

