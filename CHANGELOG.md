# Changelog

## Unreleased

- Introduced phased hardening roadmap for KnowMat 2.0.
- Removed hard-coded secrets from version control and added `.env.example`.
- Fixed `setup.cfg` metadata and core runtime dependencies.
- Switched `app_config.Settings` to Pydantic v2 `model_config`.
- Made orchestrator configuration per-run instead of mutating a global singleton.
- Fixed validator type annotations and several UTF-8 comment issues.
- Added unit tests for `schema_converter` and `domain_rules`.
- Externalised evaluation prompt to `prompts/evaluation.yaml`.
- Adjusted temperature parsing to use 273.15 K offset for Celsius.
- Unified regression diff default AI output directory to `data/output`.
- Added optional `standardization` extra for heavy vector/torch dependencies.
- Added basic CI workflow running lint, type checks and tests.
