#!/usr/bin/env python3
"""ouroboros.

    python3 ouroboros.py         render it
    python3 ouroboros.py 11      any odd order works

"""

import sys


def pattern(order: int = 7) -> str:
    m = order // 2
    return "\n".join(
        " ".join(
            "*"
            if m in (r, c)
            or (r == 0 and c > m)
            or (c == 0 and r < m)
            or (r == order - 1 and c < m)
            or (c == order - 1 and r > m)
            else " "
            for c in range(order)
        ).rstrip()
        for r in range(order)
    )


def main(argv: list[str]) -> int:
    order = int(argv[1]) if len(argv) > 1 else 7
    if order < 3 or order % 2 == 0:
        print(f"lattice order must be an odd integer >= 3, got {order}", file=sys.stderr)
        return 2
    print(pattern(order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
