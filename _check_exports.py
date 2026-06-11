import ocean
import sys

bad = []
for name in ocean.__all__:
    try:
        getattr(ocean, name)
    except Exception as e:
        bad.append(f"FAIL: {name}: {e}")
        sys.stderr.write(f"FAIL: {name}: {e}\n")

if bad:
    sys.stderr.write(f"\n{len(bad)} failures\n")
    sys.exit(1)
else:
    sys.stderr.write(f"All {len(ocean.__all__)} exports OK\n")
