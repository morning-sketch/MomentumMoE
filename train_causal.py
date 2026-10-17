import os
import runpy
import sys


def _find_data_path(argv):
    for i, arg in enumerate(argv):
        if arg == "--data" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--data="):
            return arg.split("=", 1)[1]
    return ""


def main():
    data_path = _find_data_path(sys.argv)
    is_reasoning = (
        bool(data_path)
        and os.path.exists(os.path.join(data_path, "train.json"))
        and os.path.exists(os.path.join(data_path, "valid.json"))
    )
    target = "train_reasoning.py" if is_reasoning else "train_lm.py"
    target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), target)
    sys.argv[0] = target_path
    runpy.run_path(target_path, run_name="__main__")


if __name__ == "__main__":
    main()
