<!-- Copyright 2026 Regit.io — Nicolas Koenig -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

(none)

## [1.0.1] — 2026-07-13

### Changed
- Bumped the `criterion` dev/bench dependency from 0.5 to 0.8.
- No runtime or API changes. The public API remains identical to 1.0.0
  and the crate still has zero runtime dependencies.

## [1.0.0] — 2026-05-23

First stable release. The public API is frozen under semantic versioning.

### Added — library implementation

The full yield-curve library lands in this release. Every formula is
hand-rolled from the primary paper sources cited in [`MATH.md`](MATH.md);
the crate has zero runtime dependencies and compiles cleanly to
`wasm32-unknown-unknown`.

#### Core types (`src/types.rs`)
- `Date` — proleptic-Gregorian serial day (`i32`, epoch `1970-01-01`) with
  `from_ymd`, `year`, `month`, `day`, `weekday`, ordering, integer
  arithmetic; round-trip tested on `[1900..2100]`
- `Tenor` — `(count, TenorUnit)` with `Days`, `Weeks`, `Months`, `Years`
  units and validated constructors; date-shift via `Date::add_tenor`
- `Daycount` — `Act360`, `Act365F`, `Thirty360BondBasis`, `ThirtyE360`,
  `ActActIsda`, `ActActIcma`, `Business252` with `year_fraction` against
  the ISDA 2006 §4.16 worked examples
- `Compounding` — `Simple`, `Continuous`, `Annual`, `SemiAnnual`,
  `Quarterly`, `Monthly` with `discount_to_rate` / `rate_to_discount`
  conversions
- `Frequency` — payment frequency enum (`Annual`, `SemiAnnual`,
  `Quarterly`, `Monthly`); `months_between` and `periods_per_year`
- `BusinessDayConvention` — documentation-only enum (calendar resolution
  is out-of-scope by design; supply pre-adjusted dates)

#### Typed errors (`src/errors.rs`)
- `TypeError` — invalid calendar dates, non-positive ranges, non-finite
  numeric input, invalid tenors / frequencies
- `CurveError` — anchor mismatch, non-monotone times, non-positive
  discount factors, interpolation domain violations, evaluation failures
- `BootstrapError` — invalid instruments, non-convergence, mis-ordered
  pillars, day-count failures (wraps `TypeError`)
- All three implement `Display`, `Debug`, `std::error::Error`, with `From`
  conversions; `Copy` throughout

#### Numerical primitives (`src/math/`)
- `brent` — bracketed root-finding (Brent 1973) with `BrentConfig`
  (`xtol = 1e-12`, `ftol = 1e-14`, `max_iter = 100`); golden roots
  including the Dottie number and `x³ − x − 2 = 0`
- `tridiag` — `O(n)` Thomas algorithm for tridiagonal systems, used by
  the cubic-spline solver and cross-checked against dense Gaussian
- `linear_solve` — dense Gaussian elimination with partial pivoting and
  Cholesky decomposition (Higham 2002 stability bounds); Hilbert-matrix
  and SPD round-trip tests

#### Bootstrap instruments (`src/instruments/`)
- `Deposit` — money-market deposit pricing `D(fixing)/D(payment) = 1 + r·τ`
  (Hagan & West 2006 §2, ISDA 2006 §4.6/§7.1)
- `Fra` — forward-rate agreement pinning `D(start)/D(end) = 1 + r·τ` over
  `[start, end]`
- `Future` — STIR future with caller-supplied convexity adjustment in
  rate units (model-free at the crate boundary)
- `SwapFixedFloat` — vanilla fixed-floating swap with separate fixed and
  float schedules, day-counts, and frequencies; single-curve par-rate
  identity `r = (D(t_0) − D(t_N)) / Σ τ_i D(t_i)`
- `OisSwap` — OIS swap, both legs discounted on the OIS curve in
  single-curve mode
- `BasisSwap` — tenor / cross-currency basis swap with `BasisLeg` per
  side and a spread on `leg_a`; structurally defined for both
  single-curve and multi-curve modes
- `Bond` — coupon-bearing bond with regular schedule, par-bond and
  priced-bond bootstrap support; dirty-price identity
  `dirty = coupon * SUM tau_i D(t_i) * N + N * D(t_N)`, with `coupon_pv`
  / `principal_pv` accessors and `InstrumentLike::residual` driving to
  zero against the bootstrap curve
- `SwapSchedule` — regular schedule generator from
  `(start, maturity, freq)`; irregular schedules are caller-built
- `Instrument` — `#[non_exhaustive]` enum with seven variants, dispatching
  the internal `InstrumentLike` trait

#### Interpolation methods (`src/interpolation/`)
- `Linear` — piecewise linear, flat-extrapolating (Hagan & West Method 0)
- `LogLinear` — piecewise log-linear (equivalent to piecewise-constant
  continuously-compounded zero rate)
- `LinearInZero` — linear on continuously-compounded zero rate `z`
  (Hagan & West's recommended default for discount curves)
- `PiecewiseConstantForward` — flat forwards between knots
  (equivalent to log-linear on `D`)
- `CubicSpline` — `C²` natural / clamped / not-a-knot spline via Thomas
  tridiagonal solve; not-a-knot is the default (matches QuantLib);
  Hyman 1983 RPN15A monotonicity oracle in unit tests
- `HermiteBessel` — `C¹` Hermite cubic with Bessel slopes (de Boor 2001)
- `MonotoneCubic` — Fritsch-Carlson 1980 monotone cubic with
  Fritsch-Butland 1984 weighted-harmonic interior slopes
- `MonotoneSteffen` — Steffen 1990 local monotone cubic (cross-checked
  against the GSL `steffen.c` reference)
- `MonotoneHyman` — Hyman 1983 monotonicity filter applied to a
  cubic-spline base, with the Dougherty-Edelman-Hyman 1989 improvement
  to the degenerate-slope case
- `ConvexMonotone` — Hagan–West Method 7 monotone-convex interpolant
  (Wilmott 2008); piecewise-quadratic instantaneous forward,
  integral-preserving over each segment, arbitrage-free for positive
  inputs; local (no outer iteration required)
- `Interpolation` enum, `InterpolationImpl` dispatch, `Interpolator`
  trait — uniform interface across all ten methods

#### Curve views (`src/curves/`)
- `DiscountCurve` — canonical store: knot times, discount factors,
  reference date, daycount, interpolation method; anchor and
  monotonicity invariants enforced at construction
- `ZeroCurve` — borrowing view returning `z(t)` under a chosen
  `Compounding`
- `ForwardCurve` — borrowing view returning instantaneous forwards
  `f(t) = −d/dt ln D(t)` and simply-compounded forwards `L(t_1, t_2)`;
  `C¹` derivative via the interpolator
- `ParCurve` — borrowing view returning the single-curve par swap rate
  against a chosen `Frequency` and accrual `Daycount`

#### Single-curve bootstrap (`src/bootstrap.rs`)
- `Bootstrap` — sequential iterative engine driving each instrument's
  residual to zero via Brent root-finding at successive curve pillars
- `BootstrapConfig` — `tolerance = 1e-10`, `max_iter = 50`,
  `bracket = (1e-6, 10.0)`, with an outer iteration for non-local
  interpolants (cubic spline, Hyman filter); defaults match
  Andersen & Piterbarg 2010 §6.4
- `build` returns a `DiscountCurve` that re-prices every input
  instrument to within tolerance

#### Multi-curve bootstrap (`src/multi_curve.rs`)
- `MultiCurveBootstrap` — OIS discount curve first, then per-tenor
  projection curves; floats discount on OIS and project on the tenor
  curve (post-2008 framework — Bianchetti 2010, Mercurio 2009)
- `MultiCurve` — owns the OIS curve plus a tenor-keyed map of
  projection curves
- Basis swaps are rejected as projection-curve instruments (they
  require a joint solve); a `basis_swap_residual_multi_curve` helper
  is exposed inside the module for re-pricing-check evaluation

#### Tests, benchmarks, example
- **629 inline unit tests** across all modules — golden values from
  primary papers, every error path, every accessor, ISDA 2006 §4.16
  day-count round-trips, Hyman 1983 RPN15A monotonicity oracle, Steffen
  oracle fixture, Brent and Thomas golden cross-checks, instrument
  residual-on-flat-curve identities
- **29 integration tests** (`tests/integration.rs`) in six suites —
  QuantLib `PiecewiseYieldCurve` (Modified BSD) and Google
  tf-quant-finance `bond_curve_test.py` (Apache-2.0) golden anchors,
  curve-view round-trip identities, arbitrage oracle (positive
  monotone-decreasing discount factors), the Hyman RPN15A discriminator,
  `proptest` invariants (`bootstrap_never_panics`), and a full
  multi-curve OIS + 3M IBOR re-pricing certificate
- **134 doc-tests** — every public item carries a runnable example
- **792 tests total** across unit, integration, and doc-test layers
- **Criterion benchmarks** (`benches/curves.rs`) for interpolator
  evaluation, single-curve bootstrap, multi-curve bootstrap, and the
  four curve views with documented performance targets
- **`examples/quickstart.rs`** — end-to-end walkthrough: single-curve
  bootstrap, multi-curve OIS + 3M IBOR, and the four curve views

#### Crate metadata
- `clippy::pedantic` clean across all targets
- `#![forbid(unsafe_code)]` at the crate root; no `unwrap` / `expect` /
  `panic!` / `todo!` / `unimplemented!` / `unreachable!` in library code
- `std`-only by design (curve nodes are variable-size; clean `Vec`-based
  code beats `no_std` gymnastics for the linear algebra); zero
  **runtime** dependencies; WASM-clean
  (`cargo build --target wasm32-unknown-unknown --release` passes)
- Dev-dependencies `approx`, `proptest`, `criterion` and the `[[bench]]`
  target declared in `Cargo.toml`; `deny.toml` licence allow-list covers
  the permissive dev-dependency tree

[Unreleased]: https://github.com/org-regit-io/regit-curves/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/org-regit-io/regit-curves/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/org-regit-io/regit-curves/releases/tag/v1.0.0
