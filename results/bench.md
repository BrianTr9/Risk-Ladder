| Grid | States | Build | VI (iterations) | PI (iterations) | Evaluate | Peak memory |
|---|---|---|---|---|---|---|
| 12x12 | 97 | 0.00 s | 0.00 s (69) | 0.00 s (7) | 0 ms | 67 MB |
| 34x34 | 983 | 0.00 s | 0.00 s (120) | 0.02 s (12) | 2 ms | 69 MB |
| 102x102 | 9,603 | 0.03 s | 0.08 s (318) | 0.49 s (23) | 17 ms | 134 MB |
| 320x320 | 97,119 | 0.31 s | 1.71 s (724) | 26.27 s (53) | 475 ms | 2,488 MB |
| 1002x1002 | 960,399 | 3.17 s | 63.80 s (2449) | skipped | 10657 ms | 2,525 MB |

Measured on macOS-26.5.1-arm64-arm-64bit (arm64), Python 3.12.16, NumPy 2.5.5, SciPy 1.18.1.
