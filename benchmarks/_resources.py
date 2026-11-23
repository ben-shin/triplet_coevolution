"""Cross-platform process peak resident memory in bytes."""

import sys


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
