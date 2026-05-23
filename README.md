<!-- Copyright 2026 Regit.io — Nicolas Koenig -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# regit-curves

Audit-grade interest-rate yield curve bootstrap and interpolation. Zero-dependency, pure Rust.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

## What it does

`regit-curves` bootstraps interest-rate yield curves from market instruments —
deposits, FRAs, STIR futures, fixed-floating vanilla swaps, OIS swaps, and
basis swaps — and exposes the resulting curve as four mutually consistent
views: discount factor, zero rate, instantaneous forward and par yield.

It supports both the classical **single-curve** convention and the
post-2008 **multi-curve** (OIS-discounted, IBOR-projection) framework, and
ships a documented family of interpolation methods — from log-linear on
discount factors (Hagan & West's recommended default) through Steffen and
Hyman monotone splines, Fritsch-Carlson monotone cubics, and natural /
clamped / not-a-knot cubic splines.

Every formula is hand-rolled from primary paper sources with no external
dependencies. A regulator, quant auditor, or new engineer can open any
source file and trace every number to a citable derivation in [MATH.md](MATH.md).

## Why this crate exists

An interest-rate curve is the input to every discount, every forward, and
every fixed-income risk number. Markets quote a sparse, discrete set of
instruments — but pricing and risk need a *continuous* curve.

The naive fix is to interpolate the quotes. **Interpolation silently changes
prices.** Splining zero rates introduces non-monotone forwards; piecewise
linear discount factors give negative forwards; the wrong interpolation
domain (zero rate vs log-discount vs instantaneous forward) re-prices the
same instrument differently. A curve with either defect produces mispriced
swaps, unstable hedges, and risk numbers that cannot be trusted, and the
defect is invisible unless you test for it.

`regit-curves` solves this at the bootstrap level — by re-pricing every
bootstrap instrument to zero residual at every curve node — and at the
interpolation level — by exposing the interpolation method as a first-class
choice, propagating it consistently through every derived view, and citing
its mathematical and convergence properties to the primary source.

This sits within [Regit OS](https://www.regit.io): `regit-curves` is the
yield-curve layer. It is self-contained — day-count conventions, calendar
arithmetic, and every numerical primitive ship inside the crate — and
produces a clean, audit-traceable curve for pricing and risk downstream.

## Quick start

```toml
[dependencies]
regit-curves = "1.0"
```

See [`examples/quickstart.rs`](examples/quickstart.rs) for a complete working
example covering single-curve bootstrap, multi-curve OIS-discounting, and the
full set of derived views.

## Curve views

| View | Definition | Use case |
|---|---|---|
| `DiscountCurve` | `D(t)`, `D(0) = 1` | Canonical representation; pricing of fixed cash flows |
| `ZeroCurve`     | `z(t)` with `D(t) = exp(-z(t) · t)` | Reporting; what desks quote |
| `ForwardCurve`  | `f(t) = -d/dt log D(t)` | Risk; sensitivities w.r.t. instantaneous forwards |
| `ParCurve`      | Par swap / par yield by tenor | Mark-to-market against the par market |

Conversions between any two views are total and round-trip exactly at the
curve nodes.

## Bootstrap instruments

| Instrument | Quote | Constrains |
|---|---|---|
| `Bond`            | Clean price (+ accrued) | Discount factor at coupon/maturity dates |
| `Deposit`         | Money-market rate | Short-end discount factor |
| `Fra`             | Forward rate      | Forward over `[t_1, t_2]` |
| `Future`          | Price (+ convexity adjustment) | Forward at futures expiry |
| `SwapFixedFloat`  | Par fixed rate    | Discount factors out to maturity |
| `OisSwap`         | OIS rate          | OIS discount curve |
| `BasisSwap`       | Tenor / cross-currency spread | Multi-curve projection |

## Interpolation methods

| Method | Family | Notes |
|---|---|---|
| `Linear`                    | piecewise linear         | On discount / zero / forward |
| `LogLinear`                 | piecewise log-linear     | Linear on log-D = linear on zero rate |
| `LinearInZero`              | piecewise linear         | Hagan & West's recommended default |
| `CubicSpline`               | C² spline                | Natural / clamped / not-a-knot |
| `HermiteBessel`             | C¹ Hermite               | Bessel-slope cubics |
| `MonotoneCubic`             | C¹ Hermite               | Fritsch-Carlson (1980) |
| `MonotoneSteffen`           | C¹ Hermite               | Steffen (1990) |
| `MonotoneHyman`             | C¹ Hermite               | Hyman (1983) filter on cubic |
| `ConvexMonotone`            | Hagan–West Method 7      | Arbitrage-free monotone-convex (2008) |
| `PiecewiseConstantForward`  | piecewise constant `f`   | Flat forwards between nodes |

Each method is C^k (or piecewise C^k) in the documented sense and is propagated
consistently through every derived view.

> **Note on `ConvexMonotone`.** Hagan–West Method 7 is designed for positive
> monotone non-increasing discount factors — the canonical yield-curve setting
> in which the paper proves the non-negative-forward guarantee. On that domain
> it agrees bit-exactly with independent implementations
> (verified against `tf-quant-finance` to 2.2 × 10⁻¹⁶ relative). Outside that
> domain — oscillating inputs where the implied discrete forwards change sign —
> the §3.6 proof no longer applies and the `fhat` clipping per §4 eq. 25 (used
> here verbatim) can differ from implementations that omit it. Use
> [`CubicSpline`](src/interpolation/cubic_spline.rs) or
> [`HermiteBessel`](src/interpolation/hermite_bessel.rs) for general-purpose
> non-monotone interpolation.

## Architecture

```
src/
  lib.rs                       # Module declarations + re-exports
  types.rs                     # Date, Tenor, Compounding, Daycount enum
  errors.rs                    # Typed errors — bootstrap and curve
  math/                        # Hand-rolled numerical primitives
    linear_solve.rs            # Gaussian elimination + Cholesky
    tridiag.rs                 # Thomas algorithm for spline systems
    brent.rs                   # Bracketed root-finder (Brent 1973)

  instruments/                 # Bootstrap instruments
    basis_swap.rs              # Tenor / cross-currency basis swap
    bond.rs                    # Coupon-bearing bond
    deposit.rs                 # Money-market deposit
    fra.rs                     # Forward-rate agreement
    future.rs                  # STIR future with convexity adjustment
    ois_swap.rs                # OIS swap
    schedule.rs                # SwapSchedule helper (regular schedules)
    swap_fixed_float.rs        # Vanilla fixed-floating swap

  interpolation/               # Interpolation methods
    convex_monotone.rs         # Hagan–West Method 7 (monotone-convex)
    cubic_spline.rs            # natural / clamped / not-a-knot
    hermite_bessel.rs          # Bessel-slope Hermite
    linear.rs                  # piecewise linear
    linear_in_zero.rs          # Hagan & West default
    log_linear.rs              # piecewise log-linear
    monotone_cubic.rs          # Fritsch & Carlson 1980
    monotone_hyman.rs          # Hyman 1983 filter
    monotone_steffen.rs        # Steffen 1990
    piecewise_constant_forward.rs

  curves/                      # Curve views and conversions
    discount.rs                # DiscountCurve — canonical
    zero.rs                    # ZeroCurve     — z(t)
    forward.rs                 # ForwardCurve  — f(t)
    par.rs                     # ParCurve      — par yields

  bootstrap.rs                 # Sequential iterative bootstrap engine
  multi_curve.rs               # OIS-discounted multi-curve bootstrap
```

One file, one domain. Each function is pure, deterministic, and composable.

## Testing

```bash
cargo test                      # 792 tests
cargo run --example quickstart  # End-to-end single-curve + multi-curve workflow
cargo bench                     # Criterion benchmarks
```

**629 unit tests** — golden values from primary papers, every error path,
every accessor, day-count round-trips against ISDA 2006 §4.16 worked
examples, daycount and calendar arithmetic on `[1900..2100]`, Hyman 1983
RPN15A monotonicity oracle, Steffen oracle fixture, Brent root-finder
golden roots, Thomas tridiagonal solver cross-check against dense
Gaussian, and instrument residual-on-flat-curve identities.

**29 integration tests** across six suites: golden anchors transcribed
from QuantLib's `PiecewiseYieldCurve` test suite (Modified BSD) and
Google tf-quant-finance's `bond_curve_test.py` (Apache-2.0); curve-view
round-trip identities (discount ↔ zero ↔ forward ↔ par); arbitrage
oracle (positive discount factors, monotone discount); the Hyman 1983
RPN15A monotonicity discriminator; `proptest` invariants
(`bootstrap_never_panics` under random inputs); and a full
multi-curve end-to-end OIS + 3M IBOR re-pricing certificate.

**134 doc-tests** — every public item carries a runnable example.

## Code quality

- `#![forbid(unsafe_code)]` crate-wide
- `clippy::pedantic` with zero warnings
- Every public function documented with its mathematical reference
- No `unwrap()` or `panic!()` in library code — all failure paths typed
- Deterministic: same input produces bit-identical output
- WASM-clean: `cargo build --target wasm32-unknown-unknown` with no changes
- 792 tests — unit, integration, `proptest` invariants, doc-tests — plus
  `criterion` benchmarks (see [Testing](#testing))

The crate is **`std`-only** — curve nodes are variable-size, so clean
`Vec`-based code beats `no_std` gymnastics for the heavy linear algebra.
Zero **runtime** dependencies are still enforced.

## Dependencies

**Runtime: zero.** Only `std`. No `nalgebra`, no `argmin`, no `libm`, no FFI.
Every linear solver, every root-finder, every interpolation algorithm is
hand-rolled from its primary source.

License and supply-chain policy is enforced via `cargo-deny` (`deny.toml`).
No copyleft dependencies.

## Algorithms

All implemented from primary paper sources. No ports from Python, no reading
existing Rust crates.

| Algorithm | Reference |
|---|---|
| Yield-curve bootstrap | Hagan, P. S. & West, G., *Interpolation methods for curve construction*, Applied Mathematical Finance 13(2):89–129 (2006) |
| Multi-curve / OIS discounting | Bianchetti, M., *Two curves, one price*, Risk magazine (2010); Mercurio, F., *Interest rates and the credit crunch*, SSRN (2009) |
| Cubic spline interpolation | de Boor, C., *A Practical Guide to Splines*, Springer (1978/2001), Ch. IV |
| Fritsch-Carlson monotone cubic | Fritsch, F. N. & Carlson, R. E., *Monotone piecewise cubic interpolation*, SIAM J. Numer. Anal. 17(2):238–246 (1980) |
| Steffen monotone | Steffen, M., *A simple method for monotonic interpolation in one dimension*, Astronomy & Astrophysics 239:443–450 (1990) |
| Hyman monotone filter | Hyman, J. M., *Accurate monotonicity preserving cubic interpolation*, SIAM J. Sci. Stat. Comput. 4(4):645–654 (1983) |
| Hermite-Bessel slopes | de Boor, C., *A Practical Guide to Splines*, Springer (1978/2001) |
| Tridiagonal solve | Thomas, L. H., Watson Sci. Comput. Lab. report (1949); Press et al., *Numerical Recipes*, 3rd edn., §2.4 |
| Brent's root-finder | Brent, R. P., *Algorithms for Minimization Without Derivatives*, Prentice-Hall (1973) |

Cross-checked against published numerical examples from Hagan & West (2008
worked tables), QuantLib's `PiecewiseYieldCurve` test suite, and Andersen &
Piterbarg, *Interest Rate Modeling* (Atlantic Financial Press, 2010), vol. 1
§6.

## Documentation

- [MATH.md](MATH.md) — Full mathematical derivations for every algorithm
- [CHANGELOG.md](CHANGELOG.md) — Release history
- [SECURITY.md](SECURITY.md) — Vulnerability disclosure policy

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

```
Copyright 2026 Regit.io — Nicolas Koenig
```

---

Part of [Regit OS](https://www.regit.io) — the operating system for investment products. From Luxembourg.
