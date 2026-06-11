# OpenAPI Notes

The agent plugin should call:

```text
POST /analyze
```

Input mode:

- `multipart/form-data`
- `file`: binary image upload, preferred
- `image_url`: optional HTTP/HTTPS URL if no file is uploaded
- `experiment_type`: optional, default `diffraction`

Priority rule:

1. If `file` is present, analyze `file`.
2. Else if `image_url` is present, download and analyze it.
3. Else return HTTP 400.

Export schema:

```bash
python scripts/export_openapi.py --out docs/openapi.json
```

FastAPI also serves the schema at:

```text
http://127.0.0.1:8000/openapi.json
```

