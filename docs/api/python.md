# Python API

Everything a typical user needs is importable from the top-level package:

```python
from cointf import (
    Meioses, loglik_fast, loglik_quartet, ChebyshevKernel,
    simulate_meiosis, simulate_meioses, simulate_quartets,
    loglik_naive, loglik_meiosis_naive, loglik_quartet_naive,
)
```

Each module has its own page, documenting every public class and function in full. That includes
lower-level functions that are not re-exported at the top level, such as the series for $f^*$,
$S^*$ and $G$ and the kernel interface.

| Module | Contents |
|---|---|
| [`cointf.fastlik`](python/fastlik.rst) | `Meioses`, `loglik_fast`, the exact series (`log_fstar_fast`, `log_sstar_fast`, `log_tail_fast`) and `series_kernel` |
| [`cointf.quartet`](python/quartet.rst) | `loglik_quartet` |
| [`cointf.chebyshev`](python/chebyshev.rst) | `ChebyshevKernel`, `PiecewiseChebyshev` |
| [`cointf.simintf`](python/simintf.rst) | `simulate_meiosis`, `simulate_meioses`, `simulate_quartets` |
| [`cointf.naive`](python/naive.rst) | Reference implementations: `loglik_naive`, `loglik_meiosis_naive`, `loglik_quartet_naive` and the 200-term series |

```{toctree}
:hidden:

python/fastlik
python/quartet
python/chebyshev
python/simintf
python/naive
```
