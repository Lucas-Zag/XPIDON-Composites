from pathlib import Path
import numpy as np

DATA_DIR = Path("debug_generated_data")

for path in sorted(DATA_DIR.glob("*.npy")):
    arr = np.load(path)

    print("\n" + "=" * 80)
    print(path.name)
    print("=" * 80)
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    print("min:", np.nanmin(arr))
    print("max:", np.nanmax(arr))
    print("mean:", np.nanmean(arr))
    print("first 5 rows:")
    print(arr[:5])