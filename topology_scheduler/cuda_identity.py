"""Read the CUDA-visible UUID without selecting a device or running a kernel."""

import ctypes
import sys
import uuid


def read_cuda_identity() -> dict:
    """Query the driver after Ray has configured this worker's visibility.

    Exactly one CUDA-visible device is required. Importing the module does not
    load a driver; unavailable libraries and driver errors produce diagnostics.
    """
    loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
    names = ("nvcuda.dll",) if sys.platform == "win32" else ("libcuda.so.1", "libcuda.so")
    library = None
    for name in names:
        try:
            library = loader(name)
            break
        except OSError:
            pass
    if library is None:
        return {"available": False, "reason": "No CUDA driver library is available"}

    def call(name, argtypes, *args):
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        code = function(*args)
        if code != 0:
            raise RuntimeError(f"{name} returned {code}")

    try:
        call("cuInit", [ctypes.c_uint], 0)
        count = ctypes.c_int()
        call("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)], ctypes.byref(count))
        if count.value != 1:
            return {"available": False, "device_count": count.value,
                    "reason": f"CUDA exposes {count.value} devices; exactly one is required"}
        device = ctypes.c_int()
        call("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), 0)
        raw = (ctypes.c_ubyte * 16)()
        name = "cuDeviceGetUuid_v2" if hasattr(library, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
        call(name, [ctypes.c_void_p, ctypes.c_int], ctypes.byref(raw), device.value)
        return {"available": True, "device_count": 1,
                "uuid": f"GPU-{uuid.UUID(bytes=bytes(raw))}"}
    except (AttributeError, OSError, RuntimeError) as error:
        return {"available": False, "reason": str(error)}
