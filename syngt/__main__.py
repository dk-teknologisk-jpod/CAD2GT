import multiprocessing as mp
import sys

if __name__ == "__main__":
    mp.freeze_support()
    if len(sys.argv) < 2 or sys.argv[1] == "gui":
        # fast path: show the splash before importing numpy / trimesh / mujoco etc.
        from syngt.gui.app import run
        sys.exit(run())
    from syngt.pipeline import main
    sys.exit(main())
