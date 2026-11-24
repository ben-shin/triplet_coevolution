"""Run a native command with a measured resource receipt even when the command fails."""

import os
import signal
import sys
import threading
import time
from pathlib import Path

from triplet_coevolution.operations import put_json


def peak_resident_bytes():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        # PROCESS_MEMORY_COUNTERS starts with two DWORDs followed by eight SIZE_Ts.
        counters = ctypes.create_string_buffer(8 + 8 * ctypes.sizeof(ctypes.c_size_t))
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
        if not get_info(wintypes.HANDLE(-1), counters, len(counters)):
            raise ctypes.WinError()
        return ctypes.c_size_t.from_buffer(counters, 8).value
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def main():
    # The supervisor holds the write end open. An OS kill closes it even when no
    # Python finally block can run; the abandoned attempt cannot publish a receipt.
    def watch_parent():
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel.OpenProcess(
                0x00100000, False, int(os.environ["TRIPLET_SUPERVISOR_PID"])
            )
            if handle:
                kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
                kernel.CloseHandle(handle)
        else:
            os.read(os.dup(0), 1)
            if os.getpgrp() == os.getpid():
                os.killpg(os.getpgrp(), signal.SIGTERM)
        os._exit(125)

    if os.environ.get("TRIPLET_SUPERVISED_STAGE") == "1":
        threading.Thread(target=watch_parent, daemon=True).start()
    path = Path(sys.argv.pop(1))
    start = time.monotonic()
    cpu = time.process_time()
    try:
        if sys.argv[1] == "_folder-stage":
            from triplet_coevolution.folder_stages import worker

            worker(*sys.argv[2:])
        else:
            from triplet_coevolution.cli import app

            app()
    finally:
        put_json(
            path,
            {
                "process_peak_rss_bytes": peak_resident_bytes(),
                "process_cpu_seconds": time.process_time() - cpu,
                "elapsed_seconds": time.monotonic() - start,
                "scope": "Native worker process including interpreter. External tool descendants excluded.",
            },
        )


if __name__ == "__main__":
    main()
