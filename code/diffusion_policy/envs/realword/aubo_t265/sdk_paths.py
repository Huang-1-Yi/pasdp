"""Opt-in paths for the site's native SDKs; no global installation changes."""
from pathlib import Path
import platform
import sys


def bootstrap_sdk():
    if (sys.version_info[:2] != (3, 9) or platform.system() != "Linux" or
            platform.machine().lower() not in ("x86_64", "amd64")):
        raise RuntimeError("Bundled AUBO/T265 SDKs require Linux x86_64 CPython 3.9")
    root = Path(__file__).resolve().parent / "sdk"
    for name in ("rs-t265", "aubo_sdk"):
        path = str(root / name)
        if path not in sys.path:
            sys.path.insert(0, path)


class CabinetLocks:
    """Prevent two instances of this launcher from controlling the same cabinet."""
    def __init__(self, arms):
        self.ips = sorted(arm["ip"] for arm in arms)
        self.files = []

    def __enter__(self):
        import fcntl
        import hashlib
        import tempfile
        try:
            for ip in self.ips:
                key = hashlib.sha256(ip.encode()).hexdigest()[:20]
                handle = open(Path(tempfile.gettempdir()) / f"dp_aubo_{key}.lock", "a+")
                self.files.append(handle)
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(f"Another AUBO teleop launcher owns {ip}") from exc
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *_):
        for handle in self.files:
            handle.close()
        self.files.clear()
