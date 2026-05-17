"""
Environment and import smoke test for the standalone revision project.

Revision skeleton alignment:
- Section 3.2 / confirms the current task definition can locate the EGMS data
- Section 3.3 / confirms the main first-round model dependencies are installed
- Section 3.11 / keeps SHAP as an explicitly optional dependency
"""

from __future__ import annotations

import argparse
import json
import sys

from revision_config import RevisionConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether the standalone revision project is ready to run.")
    parser.add_argument("--strict-core", action="store_true", help="Exit with code 1 if any first-round core dependency is missing.")
    parser.add_argument("--strict-all", action="store_true", help="Exit with code 1 if any core or optional dependency is missing.")
    args = parser.parse_args()

    core_modules = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch", "lightgbm"]
    optional_modules = ["shap"]

    imports = {}
    missing_core = []
    missing_optional = []

    for module_name in core_modules + optional_modules:
        try:
            module = __import__(module_name)
            imports[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            imports[module_name] = f"ERROR: {exc}"
            if module_name in core_modules:
                missing_core.append(module_name)
            else:
                missing_optional.append(module_name)

    config = RevisionConfig()
    payload = {
        "imports": imports,
        "core_ready": len(missing_core) == 0,
        "optional_ready": len(missing_optional) == 0,
        "missing_core": missing_core,
        "missing_optional": missing_optional,
        "resolved_csv_path": str(config.resolve_csv_path()),
        "split_seed": config.split_seed,
        "grid_size": config.grid_size,
        "history_length": config.history_length,
        "target_col": config.target_col,
    }

    try:
        import torch

        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        payload["torch_cuda_version"] = torch.version.cuda
        payload["torch_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        payload["torch_runtime_probe"] = f"ERROR: {exc}"

    print(json.dumps(payload, indent=2))

    if args.strict_all and (missing_core or missing_optional):
        sys.exit(1)
    if args.strict_core and missing_core:
        sys.exit(1)


if __name__ == "__main__":
    main()
