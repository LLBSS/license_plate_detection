# Copilot / AI Agent Instructions for license_plate_detection

Purpose
- Small OpenCV-based license plate detection project. Entry point: `main.py`.

Big picture (how data flows)
- CLI loads an image (`image_utils.load_image`) in `main.py` and constructs `LicensePlateDetector` (`plate_detector.py`).
- If color detection enabled, `LicensePlateDetector.detect_plates_by_multiple_colors` builds HSV masks (uses `color_config.PlateColorConfig`).
- Masks are cleaned (`process_color_mask`) and contours found (`find_contours_in_mask`).
- Candidate regions are filtered (`filter_and_validate_contours`), merged (`merge_similar_regions`) and finally drawn with `draw_boxes`.
- `shape_detector.ShapeDetector` provides alternate shape-based detection utilities (e.g. `combine_shape_features`, `detect_rectangles`).

Where to look when changing behavior
- Color rules: edit `color_config.py` and ensure HSV ranges are in `PlateColorConfig.get_all_colors()` format.
- Morphology / mask handling: `plate_detector.py` — functions `detect_plates_by_multiple_colors` and `process_color_mask` (kernels, close/open order).
- Geometric thresholds (sizes, aspect ratios, edge density): `LicensePlateDetector.__init__` in `plate_detector.py`.
- Shape heuristics and polygon checks: `shape_detector.py` (use `is_parallelogram`, `detect_parallelograms`).

Dev workflows & commands
- Install deps: `pip install -r requirements.txt` (project includes a `.venv/` directory but virtualenv use is optional).
- Run example (PowerShell):
```powershell
pip install -r requirements.txt;
python .\main.py .\examples\sample.jpg --display-masks
```
- CLI flags (see `main.py`): `--no-color`, `--color [blue|yellow|green|white|black]`, `--display-masks`.
- Visual output uses `cv2.imshow` — run on a desktop/session with a display.

Project-specific conventions
- Docstrings and runtime messages are in Chinese; keep new user-facing prints consistent.
- Naming: snake_case for functions and variables; classes are PascalCase (`LicensePlateDetector`, `ShapeDetector`).
- No test suite present — be conservative with changes; prefer small, isolated edits and manual testing via `main.py`.

Examples for quick edits
- To add a new plate color: update `PlateColorConfig.get_all_colors()` in `color_config.py` with the new HSV ranges and label.
- To relax detection: increase `min_plate_width` / decrease `min_edge_density` in `LicensePlateDetector.__init__`.
- To debug masks programmatically:
```python
from plate_detector import LicensePlateDetector
from image_utils import load_image
img = load_image('examples/sample.jpg')
det = LicensePlateDetector()
masks = det.detect_plates_by_multiple_colors(img)
for name, mask in masks.items():
    print(name, mask.shape)
```

Integration points & external deps
- OpenCV (`cv2`) for image ops, NumPy for arrays, `matplotlib` only present in requirements but not required by core flow.
- Files to inspect when integrating: `image_utils.py` (preprocessing), `color_config.py` (HSV ranges), `plate_detector.py` (core logic), `shape_detector.py` (shape helpers).

What agents should NOT change without confirmation
- CLI argument names in `main.py`.
- Public function signatures in `plate_detector.py` (`detect`, `draw_boxes`) used by `main.py` and examples.

If anything is unclear or you want the instructions in Chinese, tell me which sections to expand or translate.
