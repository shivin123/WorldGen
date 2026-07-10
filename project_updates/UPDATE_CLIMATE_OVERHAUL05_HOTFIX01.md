# UPDATE_CLIMATE_OVERHAUL05_HOTFIX01

Fixes a Web UI startup crash introduced in UPDATE_CLIMATE_OVERHAUL05_BIGPACK.

## Fixed

- Escaped the `.viewer-overlay-tools { ... }` CSS block inside the Python f-string page template.
- This prevents `NameError: name 'position' is not defined` when loading the Web UI home page.
- Also removes the harmless invalid escape warning from the affected string replacement where possible.

## Validation

- `python -m py_compile worldgen/webui.py` passed.
