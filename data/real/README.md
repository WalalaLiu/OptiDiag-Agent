# Real Diffraction Image Data

This directory is reserved for real laboratory diffraction images. Do not commit real images unless the course explicitly allows sharing them.

Recommended local structure:

```text
data/real/
  raw/
    single_slit/
    double_slit/
    triple_slit/
    rectangle_aperture/
    circular_aperture/
    double_circular_aperture/
  annotations/
    real_labels.csv
  reviewed/
```

Recommended file naming:

```text
YYYYMMDD_experimentType_imageType_issue_operator_index.png
```

Examples:

```text
20260611_diffraction_double_slit_misalignment_doudou_001.png
20260611_diffraction_circular_aperture_blur_defocus_doudou_002.png
20260611_diffraction_single_slit_clean_doudou_003.png
```

Suggested `real_labels.csv` columns:

```text
image_path
image_type
issues
primary_issue
severity
exposure_ms
gain
wavelength_nm
focal_length_mm
aperture_notes
operator
capture_date
review_notes
usable_for_training
```

Annotation guidance:

- `issues` can contain multiple semicolon-separated labels, for example `misalignment;low_contrast`.
- Use `clean` as `primary_issue` only when the image is suitable as a good reference.
- Keep ambiguous samples, but mark `usable_for_training=false` until reviewed.
- Record acquisition settings when available; they are valuable for explaining model failures.
- Do not place student/private identifying information in filenames.

