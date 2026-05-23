<!-- Copyright 2026 Regit.io — Nicolas Koenig -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MATH.md — regit-curves

> Full formula derivations for every algorithm in this crate. Each section
> maps to the source module named in its heading and cites the primary paper
> reference. All formulas are shown in plain-text notation using code blocks.
>
> Every formula stated here is implemented by that module — the crate is the
> executable form of this document, and each `src/` file carries the same
> derivations and citations in its own doc comments.

---

## Table of contents

1. [Time, day-count and compounding](#time-day-count-and-compounding--srctypesrs)
2. [Numerical primitives](#numerical-primitives--srcmath)
3. [Bootstrap instruments](#bootstrap-instruments--srcinstruments)
4. [Interpolation methods](#interpolation-methods--srcinterpolation)
5. [Curve views and conversions](#curve-views-and-conversions--srccurves)
6. [Single-curve bootstrap](#single-curve-bootstrap--srcbootstraprs)
7. [Multi-curve bootstrap](#multi-curve-bootstrap--srcmulti_curvers)
8. [Algorithm references](#algorithm-references)

---

## Time, day-count and compounding — `src/types.rs`

**Source:** ISDA, *2006 ISDA Definitions*, §4.16 ("Day Count Fraction");
ICMA, *Rule 251* ("Actual/Actual (ICMA)"); Hinnant, H.,
*chrono-Compatible Low-Level Date Algorithms*,
<https://howardhinnant.github.io/date_algorithms.html>.

A yield curve lives on a one-dimensional time axis measured in
**year fractions** from the curve's reference date. The conversion from a
pair of calendar dates `(d_1, d_2)` to a year fraction `tau(d_1, d_2)` is
the **day-count convention**, and the conversion from a discount factor
`D(t)` to a quoted rate `r` is the **compounding convention**. Both are
purely conventional — neither carries any modelling assumption — but they
are mandatory inputs to every formula in the crate.

### Date model

A `Date` is a signed day-serial counted from the proleptic-Gregorian epoch
`1970-01-01`. The conversion `(year, month, day) <-> serial` is exact
integer arithmetic (Hinnant's `days_from_civil` / `civil_from_days`); no
floats are used in calendar conversions. The proleptic-Gregorian calendar
extends the Gregorian rule backwards through the pre-1582 era so that the
serial is monotonic, exact, and unambiguous for every financial use case.

### Tenor model

A `Tenor` is a literal `(count, unit)` pair with `unit in {Days, Weeks,
Months, Years}`. Conversion to a date is via `Tenor::add_to`:

```
Days, Weeks:    start.add_days(count * 1 or 7)
Months, Years:  end-of-month preserved (clip to days_in_month of target)
```

The end-of-month-preserved rule states: if `start.day` exceeds the number
of days in the target month, the result is clipped to the last day of that
month (so `Jan 31 + 1M = Feb 28` in a non-leap year, `Feb 29` in a leap
year).

### Day-count conventions

Let `Y_i, M_i, D_i` be the calendar year, month, day of `d_i`, and write
`span = d_2 - d_1` for the integer day count between two dates.

**Act/360 — ISDA 4.16(e).** The money-market default.

```
tau = span / 360
```

**Act/365 (Fixed) — ISDA 4.16(d).**

```
tau = span / 365
```

**30/360 (Bond Basis) — ISDA 4.16(f).** Apply the adjustments

```
(R1)  if D_1 = 31 then D_1 := 30
(R2)  if D_2 = 31 and D_1 in {30, 31} then D_2 := 30
```

then compute

```
tau = ( 360*(Y_2 - Y_1) + 30*(M_2 - M_1) + (D_2 - D_1) ) / 360
```

**30E/360 (Eurobond) — ISDA 4.16(g).** Both `D_1` and `D_2` are clipped to
`30` unconditionally; then the same 30/360 formula applies. The clipped
versions are `min(D_i, 30)`.

The discriminator between Bond Basis and Eurobond is end-of-February
behaviour: for the pair `(2008-02-28, 2008-08-31)`, Bond Basis returns
`183` days while Eurobond returns `182`.

**Act/Act (ISDA) — ISDA 4.16(b).** Split the range at every calendar-year
boundary; each sub-range contributes `days / denom` where `denom = 366` if
the year is leap, `365` otherwise. With `d_1 in Y_1` and `d_2 in Y_2`,

```
if Y_1 = Y_2:
    tau = span / (366 or 365)

else:
    first  = days(d_1, (Y_1+1)-01-01) / denom(Y_1)
    middle = (Y_2 - Y_1 - 1)                        full whole years
    last   = days(Y_2-01-01, d_2)     / denom(Y_2)
    tau    = first + middle + last
```

Leap-year detection uses the standard rule
`leap(y) <=> (y mod 4 = 0 and y mod 100 != 0) or y mod 400 = 0`.

**Act/Act (ICMA) — ICMA Rule 251.** For a single regular coupon period
under a coupon schedule with `n` periods per year,

```
tau = 1 / n
```

Callers that split an irregular period into multiple regular ones must
compose this convention themselves; the crate does not impute period
membership.

**Business/252.** Brazilian convention. Requires a jurisdiction-specific
business-day calendar and is rejected by the year-fraction routine; callers
supply already-computed year fractions for this convention.

### Year-fraction worked examples

ISDA 4.16(e), `d_1 = 2003-11-01`, `d_2 = 2004-05-01`:

```
span = 182,    tau(Act/360) = 182 / 360 = 0.5055555...
```

ISDA 4.16(b), same pair (cross-year split at `2004-01-01`):

```
first  = 61 / 365                 (61 days in non-leap 2003)
last   = 121 / 366                (121 days in leap 2004)
middle = 0
tau    = 61/365 + 121/366 = 0.49772438...
```

### Compounding conventions

For time `t > 0` and zero rate `r`,

| Variant            | Discount factor `D`                | Inverse `r(D, t)`              |
|--------------------|------------------------------------|--------------------------------|
| `Simple`           | `D = 1 / (1 + r*t)`                | `r = (1/D - 1) / t`            |
| `Continuous`       | `D = exp(-r*t)`                    | `r = -ln(D) / t`               |
| `Periodic { n }`   | `D = (1 + r/n)^(-n*t)`             | `r = n * (D^(-1/(n*t)) - 1)`   |

At `t = 0` the discount factor is exactly `1` for any rate, and the
"rate implied by `D = 1`" is `0`. Any other `t = 0` query is rejected.

All formulas above are implemented by `src/types.rs`.

---

## Numerical primitives — `src/math/`

**Source:** Brent, R. P., *Algorithms for Minimization Without Derivatives*,
Prentice-Hall (1973), Chapter 4; Thomas, L. H., *Elliptic Problems in
Linear Difference Equations over a Network*, Watson Sci. Comput. Lab.
Report (Columbia University, 1949); Golub, G. H. & Van Loan, C. F.,
*Matrix Computations*, 4th edition, Johns Hopkins University Press (2013),
Chapters 3 and 4; Press, W. H., Teukolsky, S. A., Vetterling, W. T. &
Flannery, B. P., *Numerical Recipes: The Art of Scientific Computing*,
3rd edition, Cambridge University Press (2007), §2.4 and §9.3.

The crate is zero-dependency, so every solver is hand-rolled from its
primary source. All routines are pure functions, deterministic
(bit-identical output for identical input), and `std`-only. Errors surface
through the local `MathError` enum (`Singular`, `NotSpd`,
`DimensionMismatch`, `NoConvergence`, `BracketNotStraddling`).

### Brent root-finder — `src/math/brent.rs`

Brent (1973) combines the guaranteed convergence of bisection with the
super-linear speed of inverse-quadratic interpolation (IQI) and the secant
rule, falling back to bisection whenever the interpolant would step
outside the current bracket or fail a progress test. On any continuous
function with `f(a)` and `f(b)` of opposite sign the method converges to a
root in `O(log((b-a)/xtol))` evaluations in the worst case and at a
super-linear rate on smooth targets.

The algorithm maintains four scalars `(a, b, c, d)` and their function
values, with the running invariants

```
|f(b)| <= |f(a)|     (b is the current best estimate)
f(a) * f(b) < 0      (the bracket straddles a root)
c                    is the previous value of b
d                    is the value of b two iterations back
```

At each step, an IQI trial point

```
s = a*fb*fc / ((fa-fb)*(fa-fc))
  + b*fa*fc / ((fb-fa)*(fb-fc))
  + c*fa*fb / ((fc-fa)*(fc-fb))
```

is accepted iff it lies in the band `s in [(3a+b)/4, b]` (Brent's
progress test) and the step `|s - b|` is at most half the previous step
(`|b - c|/2` if the last action was a bisection, `|c - d|/2` otherwise).
If IQI is rejected the method takes a secant step

```
s = b - fb * (b - a) / (fb - fa)
```

subject to the same band-and-step tests. If both fail, the method bisects:
`s = (a + b) / 2`. After the trial, the bracket is updated so that the new
`(a, b)` again straddles zero and `|f(b)| <= |f(a)|`.

Convergence is declared when `|b - a| <= xtol`, `|f(b)| <= ftol`, or `f`
evaluates exactly to zero. Default tolerances are `xtol = 1e-12`,
`ftol = 1e-14`, `max_iter = 100`.

### Thomas tridiagonal solver — `src/math/tridiag.rs`

The Thomas algorithm (Thomas 1949) solves the tridiagonal system

```
  b_0   c_0                              x_0       d_0
  a_1   b_1   c_1                        x_1       d_1
        ...   ...   ...                  ...   =   ...
                  a_{n-2}  b_{n-2}  c_{n-2}    x_{n-2}   d_{n-2}
                                a_{n-1}  b_{n-1}    x_{n-1}   d_{n-1}
```

in `O(n)` operations via a forward sweep that eliminates the sub-diagonal
and a back-substitution that recovers the solution. Writing
`c'_0 = c_0 / b_0`, `d'_0 = d_0 / b_0`, and for `i = 1, ..., n-1`,

```
denom_i = b_i - a_i * c'_{i-1}
c'_i    = c_i / denom_i                                 (i < n-1)
d'_i    = (d_i - a_i * d'_{i-1}) / denom_i
```

then back-substitute

```
x_{n-1} = d'_{n-1}
x_i     = d'_i - c'_i * x_{i+1}    for i = n-2, ..., 0
```

The algorithm assumes no zero pivot along the elimination path. Diagonally
dominant systems — which is the case for the natural / clamped / not-a-knot
cubic-spline system and for the Hyman filter's spline base — satisfy this
condition by construction (Golub & Van Loan §4.3 for the strict-diagonal-
dominance argument). A pivot collapsing below `1e-14` raises
`MathError::Singular`.

### Gaussian elimination with partial pivoting — `src/math/linear_solve.rs`

For a general square dense matrix `A`, Gaussian elimination with partial
(row) pivoting solves `A x = b` by transforming `A` to upper-triangular
form, swapping rows at each elimination step to use the row whose
absolute pivot value is largest:

```
for k = 0, ..., n-1:
    pivot_row = argmax_{i >= k} |a_{i,k}|
    swap rows k and pivot_row in (A, b)
    if |a_{k,k}| < PIVOT_TOL:  return Singular
    for i = k+1, ..., n-1:
        factor   = a_{i,k} / a_{k,k}
        a_{i, :} -= factor * a_{k, :}
        b_i      -= factor * b_k
```

After elimination, back-substitute the upper-triangular system. The pivot
tolerance is `1e-14`; below this the matrix is treated as singular up to
round-off. The method is Golub & Van Loan §3.4 (Algorithm 3.4.1).

### Cholesky decomposition — `src/math/linear_solve.rs`

For a symmetric positive-definite matrix `A`, the Cholesky factorisation
`A = L L^T` writes `A` as the product of a lower-triangular matrix and its
transpose. The factor entries are computed row by row:

```
for j = 0, ..., n-1:
    L_{j,j} = sqrt( A_{j,j} - SUM_{k<j} L_{j,k}^2 )
    for i = j+1, ..., n-1:
        L_{i,j} = ( A_{i,j} - SUM_{k<j} L_{i,k} * L_{j,k} ) / L_{j,j}
```

A non-positive radicand signals that `A` is not SPD and raises
`MathError::NotSpd`. Once `L` is in hand, `A x = b` is solved by two
triangular sweeps: `L y = b` (forward), then `L^T x = y` (back). The
method is Golub & Van Loan §4.2 (Algorithm 4.2.2). Cholesky is preferred
to general LU on SPD matrices both for the `2x` operation-count saving
and for its unconditional numerical stability (no pivoting required).

All four primitives above are implemented in `src/math/`.

---

## Bootstrap instruments — `src/instruments/`

**Source:** Hagan, P. S. & West, G., *Interpolation methods for curve
construction*, *Applied Mathematical Finance* 13(2):89-129 (June 2006),
§2; Mercurio, F., *Interest Rates and The Credit Crunch: New Formulas and
Market Models*, Bloomberg Portfolio Research Paper No. 2010-01-FRONTIERS
(February 2009), §3; ISDA, *2006 ISDA Definitions*, §4.6
("Calculation Period") and §7.1 ("Single-Period Floating Rate Notes");
Hull, J. C., *Options, Futures, and Other Derivatives*, 10th edition,
Pearson (2018), §6.3; Andersen, L. B. G. & Piterbarg, V. V., *Interest
Rate Modeling, Volume I: Foundations and Vanilla Models*, Atlantic
Financial Press (2010), §6.3.

The bootstrap engine consumes a list of market instruments and, for each
one, drives a candidate discount factor at the instrument's pillar date
so that the instrument's residual against the in-progress curve falls to
zero. Each instrument exposes:

- a **pillar date** — the latest date it constrains; and
- a **residual** — the difference between the instrument's market quote
  and the price implied by an in-progress discount curve.

The residual is zero at the bootstrap solution. The pricing identities
below give the residual formulas for every supported instrument.

### Deposit — `src/instruments/deposit.rs`

A money-market deposit settles at the value date `t_v` and is repaid at
the maturity date `t_m`, accruing at a quoted simply-compounded rate `r`
over the day-count accrual `tau = dc.year_fraction(t_v, t_m)`. By
no-arbitrage in the single-curve world,

```
D(t_v) / D(t_m) = 1 + r * tau
```

so the instrument residual is

```
residual = D(t_v) / D(t_m) - (1 + r * tau)
```

Equivalently, the discount factor at maturity is determined by the
discount factor at the value date via `D(t_m) = D(t_v) / (1 + r * tau)`.
Negative rates are permitted — they have been quoted on EUR/CHF money
markets since 2014.

### FRA — `src/instruments/fra.rs`

A forward-rate agreement fixes today, for a forward accrual period
`[t_s, t_e]`, a simply-compounded forward rate `r`. The no-arbitrage
identity is structurally identical to the deposit's, with the period
sitting strictly in the future:

```
D(t_s) / D(t_e) = 1 + r * tau,    tau = dc.year_fraction(t_s, t_e)

residual = D(t_s) / D(t_e) - (1 + r * tau)
```

This identity reappears in the multi-curve framework as the
projection-leg definition for a single forward (Mercurio 2009, §3.1):
in the multi-curve setting `D` is replaced by the **projection** curve
`D_proj`, while discounting itself is moved to the OIS curve.

### STIR future — `src/instruments/future.rs`

A short-term interest-rate (STIR) future is quoted as a price `P`. The
implied quoted rate over the underlying period `[t_s, t_e]` is
`r_q = (100 - P) / 100`. Because a futures contract is daily margined,
its implied rate is not the no-arbitrage forward — it is the **futures
rate**, biased by a positive **convexity adjustment** `c`. The forward
rate is recovered by subtracting `c`:

```
r_fwd = (100 - P) / 100 - c
```

Once `r_fwd` is known the instrument pins the discount curve on the same
simply-compounded identity as the FRA:

```
residual = D(t_s) / D(t_e) - (1 + r_fwd * tau)
```

The convexity adjustment `c` is a **caller-supplied scalar in rate units**.
The model that produces it — typically a short-rate model (Hull-White,
Black-Karasinski) or an HJM / SABR construction calibrated to caps and
swaptions (Hull 2018, §6.3; Andersen & Piterbarg 2010, §6.3) — is out of
scope for this crate; passing `c = 0` disables the adjustment.

### Vanilla fixed-floating swap (single-curve) — `src/instruments/swap_fixed_float.rs`

A vanilla IRS exchanges a stream of fixed coupons against a floating
stream indexed off a single IBOR-style tenor. In the **single-curve**
framework — the same curve discounts cash flows and projects forward
rates — the floating-leg present value telescopes to the two end
discount factors.

*Derivation.* Let the float schedule be `t_0 < t_1 < ... < t_N` with
accruals `tau_i^float`. The forward rate over `[t_{i-1}, t_i]` implied by
the curve is

```
F_i = ( D(t_{i-1}) / D(t_i) - 1 ) / tau_i^float
```

so

```
PV_float = SUM_i  tau_i^float * F_i * D(t_i)
         = SUM_i  ( D(t_{i-1}) - D(t_i) )
         = D(t_0) - D(t_N)
```

(the same telescoping is given by Mercurio 2009 eq. (3.3)). The fixed-leg
PV under the same curve is

```
PV_fixed = rate * SUM_j  tau_j^fixed * D(t_j^fixed)
```

Equating the two legs yields the single-curve par-swap identity

```
rate * SUM_j tau_j^fixed * D(t_j^fixed)  =  D(t_0) - D(t_N)
```

and the residual

```
residual = PV_fixed - PV_float
         = rate * SUM_j tau_j^fixed * D(t_j^fixed) - (D(t_0) - D(t_N))
```

Zero at the bootstrap solution; positive when the quoted rate is above the
curve-implied par, negative when below. The two legs may use different
payment frequencies and accrual day-counts (e.g. annual 30/360 fixed
against semi-annual Act/360 float).

### OIS swap — `src/instruments/ois_swap.rs`

An OIS exchanges, on each period, a fixed coupon against the
daily-compounded realisation of an overnight rate over the same period.
The expected realised compounded overnight rate over `[t_{i-1}, t_i]` is
the simply-compounded OIS forward rate over the same interval (Mercurio
2009, §3.2). OIS is intrinsically a **single-curve** instrument — the OIS
curve both projects the float leg's forwards and discounts every cash
flow — so the float-leg telescoping argument above applies verbatim:

```
PV_float = SUM_i D(t_i) * tau_i * F_i
         = SUM_i ( D(t_{i-1}) - D(t_i) )
         = D(t_0) - D(t_N)
```

The fixed-leg PV is `PV_fixed = rate * SUM_i tau_i * D(t_i)`, and the par
OIS identity is

```
rate * SUM_i tau_i * D(t_i)  =  D(t_0) - D(t_N)
```

with residual `PV_fixed - PV_float`. This pins the OIS discount curve to
its quoted par OIS rates. Market convention pays once at maturity for OIS
tenors `<= 1Y` (a single bullet payment) and annually thereafter.

### Basis swap — `src/instruments/basis_swap.rs`

A basis swap exchanges two floating legs of different tenors (or
different currencies). Let leg `j in {a, b}` have payment schedule
`(t_0^j, ..., t_N^j)` and per-period accrual `tau_i^j`, and let `D` be
the OIS discount curve while `P_j` is the projection curve for the tenor
of leg `j`. Under no-arbitrage,

```
PV_float(leg_j) = SUM_i tau_i^j * F_j(t_{i-1}^j, t_i^j) * D(t_i^j)
F_j(t_{i-1}, t_i) = ( P_j(t_{i-1}) / P_j(t_i) - 1 ) / tau_i^j
```

The basis-swap quote is the spread `s` added to `leg_a` that equalises
the legs:

```
PV_float(leg_a) + s * A_a  =  PV_float(leg_b)
A_a = SUM_i tau_i^a * D(t_i^a)
```

so the spread is

```
s = ( PV_float(leg_b) - PV_float(leg_a) ) / A_a
```

Basis spreads pin a **tenor projection curve to the OIS discount curve**;
they are a multi-curve phenomenon and the residual is detailed in §7
below.

### Coupon bond — `src/instruments/bond.rs`

**Source:** Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate
Modeling, Volume I: Foundations and Vanilla Models*, Atlantic Financial
Press (2010), §5.1-§5.2; ISDA, *2006 ISDA Definitions*, §6 ("Fixed
Amounts and Floating Amounts").

A coupon-bearing bond settles at the issue date `t_0`, pays a fixed
coupon `coupon * tau_i * notional` on each scheduled coupon date
`t_1, ..., t_N` (with `tau_i = dc.year_fraction(t_{i-1}, t_i)` under the
bond's day-count), and repays the notional on the maturity date `t_N`.
The quoted market data are the **clean price** and the **accrued
interest** at settlement; the present-value identity is in the **dirty
price** `dirty = clean + accrued`:

```text
dirty_price = clean_price + accrued
            = coupon * SUM_i tau_i * D(t_i) * notional
              + notional * D(t_N)
```

The residual driven to zero by the bootstrap is the difference between
the present value of the cash flows and the quoted dirty price:

```text
residual = coupon_pv + principal_pv - dirty_price
         = coupon * SUM_i tau_i * D(t_i) * notional
           + notional * D(t_N)
           - (clean_price + accrued)
```

**Par bond at issue.** When the bond is quoted at par on its issue date
the clean price equals the notional and the accrued is zero, so the
identity collapses to

```text
coupon * SUM_i tau_i * D(t_i) + D(t_N) = 1,
```

structurally identical to the OIS-swap par equation: the coupon plays the
role of the fixed rate and the bond's coupon schedule plays the role of
the swap's fixed-leg schedule. The bootstrap engine pins the discount
factors at the coupon and maturity dates by driving the residual above to
zero against the in-progress curve; the accessors `coupon_pv` and
`principal_pv` expose the two PV components for re-pricing audit.

### Schedule generation — `src/instruments/schedule.rs`

A `SwapSchedule` is a precomputed sequence of contiguous accrual periods
`[d_0, d_1), [d_1, d_2), ..., [d_{n-1}, d_n)`. The regular generator
splits the total term into `n` equal periods of length
`12 / freq.periods_per_year()` months, generated forwards from `start`,
and rejects the schedule unless the `n`-th boundary falls exactly on
`maturity` (i.e. the schedule must be truly regular at the requested
frequency). For irregular schedules — stub first/last periods, modified-
following business-day adjustments — the caller composes the schedule
itself from `Date` arithmetic. Holiday calendars are jurisdiction-specific
and intentionally out-of-scope.

All instrument residuals above are implemented in `src/instruments/`.

---

## Interpolation methods — `src/interpolation/`

**Source:** Hagan, P. S. & West, G., *Interpolation methods for curve
construction*, *Applied Mathematical Finance* 13(2):89-129 (2006), §3;
Hagan, P. S. & West, G., *Methods for constructing a yield curve*,
*Wilmott Magazine*, May 2008, pp. 70-81; de Boor, C., *A Practical Guide
to Splines*, Revised Edition, Springer (2001), Chapter IV; Press, W. H.
et al., *Numerical Recipes*, 3rd ed., Cambridge University Press (2007),
§3.3-§3.4; Fritsch, F. N. & Carlson, R. E., *Monotone piecewise cubic
interpolation*, *SIAM J. Numer. Anal.* 17(2):238-246 (1980); Fritsch, F. N.
& Butland, J., *A method for constructing local monotone piecewise cubic
interpolants*, *SIAM J. Sci. Stat. Comput.* 5(2):300-304 (1984); Steffen,
M., *A simple method for monotonic interpolation in one dimension*,
*Astronomy & Astrophysics* 239:443-450 (1990); Hyman, J. M., *Accurate
monotonicity preserving cubic interpolation*, *SIAM J. Sci. Stat. Comput.*
4(4):645-654 (1983); Dougherty, R. L., Edelman, A. & Hyman, J. M.,
*Nonnegativity-, monotonicity-, or convexity-preserving cubic and quintic
Hermite interpolation*, *Mathematics of Computation* 52(186):471-494
(1989).

Yield-curve construction needs a rule for interpolating between knots.
Hagan & West (2006) §3 enumerate nine methods and discuss their
trade-offs; the crate ships eight (Method 0-2, Method 4 with three
boundary conditions, Method 7, plus Fritsch-Carlson, Steffen and Hyman
monotone Hermite variants). Every interpolator is a knot-based map
`(t_i, y_i) -> y(t)` exposing a uniform `build / eval / deriv` interface.

Throughout this section, knots are `(t_0, y_0), ..., (t_{n-1}, y_{n-1})`
with strictly increasing `t_i`. Define

```
h_i  = t_{i+1} - t_i                          (segment width, 0 <= i < n-1)
S_i  = (y_{i+1} - y_i) / h_i                  (secant slope on segment i)
```

`m_i` denotes a slope at knot `i` (used by the Hermite-cubic methods),
`M_i` denotes a second derivative at knot `i` (used by the C^2 cubic
spline).

### Linear — Hagan-West Method 0 — `src/interpolation/linear.rs`

Straight-line interpolation in the value domain. On a segment
`[t_lo, t_hi]` containing `t`, write `w = (t - t_lo) / (t_hi - t_lo)`:

```
y(t) = (1 - w) * y_lo + w * y_hi
```

The interpolant is C^0 everywhere and C^1 on the open interior of each
segment, with kinks at the knots in general. Applied directly to discount
factors it produces a step-discontinuous instantaneous forward, which is
the classical motivation for the smoother methods that follow (Hagan & West
2006, §3).

### Log-linear on `D` / piecewise-constant forward — Hagan-West Method 1 — `src/interpolation/log_linear.rs`, `src/interpolation/piecewise_constant_forward.rs`

Linear interpolation of `ln(y)` between knots:

```
y(t) = exp( (1 - w) * ln(y_lo) + w * ln(y_hi) ),    w = (t - t_lo)/(t_hi - t_lo)
```

The natural read of this method in the discount-factor domain is that
`ln D(t)` is linear on each segment. Because `ln D(t) = -z(t) * t` and the
identity is linear in `t` precisely when `z` is constant on the segment,
log-linear-on-`D` is equivalent to **piecewise-constant continuously-
compounded zero rate**. The segment zero rate reads

```
z = ( ln(y_lo) - ln(y_hi) ) / ( t_hi - t_lo )
```

Equivalently, the instantaneous forward `f(t) = -d/dt ln D(t)` is
piecewise constant equal to that segment `z`:

```
f_i = ( ln D_i - ln D_{i+1} ) / ( t_{i+1} - t_i )
D(t) = D_i * exp( -f_i * (t - t_i) ),    t_i <= t <= t_{i+1}
```

The three statements — log-linear on `D`, piecewise-constant `z`,
piecewise-constant `f` — describe the same `D(t)`. `LogLinear` and
`PiecewiseConstantForward` are two semantic views of one mathematical
object: `LogLinear` exposes `eval(t) = D(t)`; `PiecewiseConstantForward`
exposes the segment forward as a first-class quantity. Both require
`y > 0` at every knot (the natural invariant of the discount-factor
domain).

### Linear in continuously-compounded zero — Hagan-West Method 2 — `src/interpolation/linear_in_zero.rs`

Linear interpolation of the **zero rate** between knots, with `D` recovered
exponentially. For `t > 0`,

```
z(t)  =  -ln(D(t)) / t
z(t)  =  (1 - w) * z_lo + w * z_hi,    w = (t - t_lo)/(t_hi - t_lo)
D(t)  =  exp( -z(t) * t )
```

This is Hagan & West's "Method 2" and is identified in §4 of the 2006
paper as the **recommended default** for "good behaviour combined with
simplicity". The implied instantaneous forward is piecewise linear and
therefore continuous at the knots — a meaningful smoothness gain over
Method 1 at no algorithmic cost.

**Anchor convention.** The zero rate `z(t) = -ln D(t) / t` is undefined at
`t = 0` (a `0/0` limit). When the anchor knot `(t = 0, D = 1)` is present
we extend the first non-degenerate segment's zero rate to the anchor — set
`z_0 := z_1` — so that interpolation across the first segment is the
constant `z_1`. This gives `D(t) = exp(-z_1 * t)` on `[0, t_1]`, which
agrees with `D_0 = 1` at `t = 0` and `D_1` at `t = t_1`.

### Cubic spline — Hagan-West Method 4 — `src/interpolation/cubic_spline.rs`

A cubic spline is a piecewise cubic that is **C^2** (continuous through
second derivatives) at every interior knot. Parametrise the spline by its
second derivatives `M_i = y''(t_i)` at the knots. Continuity of the first
derivative across each interior knot yields the tridiagonal system
(Press et al. 2007 §3.3, de Boor 2001 §IV)

```
h_{i-1} * M_{i-1}
  + 2*(h_{i-1} + h_i) * M_i
  + h_i * M_{i+1}
  =  6 * ( (y_{i+1} - y_i)/h_i  -  (y_i - y_{i-1})/h_{i-1} )

for i = 1, ..., n-2.
```

Two boundary conditions close the system:

**Natural** — `M_0 = M_{n-1} = 0`. The unique C^2 spline minimising the
strain energy `INTEGRAL (y'')^2 dt` over all C^2 interpolants of the data
(de Boor 2001, Theorem IV.5).

**Clamped** — first derivative specified at each endpoint:
`y'(t_0) = first`, `y'(t_{n-1}) = last`. The boundary rows read

```
2*h_0       * M_0     + h_0       * M_1       = 6 * ( S_0       - first )
h_{n-2}     * M_{n-2} + 2*h_{n-2} * M_{n-1}   = 6 * ( last      - S_{n-2} )
```

**Not-a-knot** — the third derivative is continuous at `t_1` and
`t_{n-2}`, so the spline is a single cubic across the first two segments
and across the last two. Equivalently, `M_1` and `M_{n-2}` satisfy an
extrapolation relation derived from continuity of `M'`:

```
M_0     = (1 + h_0    /h_1     ) * M_1     - (h_0    /h_1    ) * M_2
M_{n-1} = (1 + h_{n-2}/h_{n-3} ) * M_{n-2} - (h_{n-2}/h_{n-3}) * M_{n-3}
```

substituted into the first / last interior row to give two boundary
rows in `M_1, M_2` and `M_{n-2}, M_{n-3}` respectively. Not-a-knot is the
QuantLib default cubic spline.

Once the `M_i` are solved (by the Thomas tridiagonal solver of §2), the
spline on segment `i` (`t in [t_i, t_{i+1}]`) is Press et al. (2007)
equation 3.3.3:

```
y(t)  = ( (t_{i+1} - t)^3 * M_i + (t - t_i)^3 * M_{i+1} ) / (6 * h_i)
      + ( y_i      / h_i  -  M_i      * h_i / 6 ) * (t_{i+1} - t)
      + ( y_{i+1}  / h_i  -  M_{i+1}  * h_i / 6 ) * (t - t_i)

y'(t) = -(t_{i+1} - t)^2 * M_i     / (2 * h_i)
      +  (t - t_i)^2     * M_{i+1} / (2 * h_i)
      + (y_{i+1} - y_i) / h_i
      + (M_i - M_{i+1}) * h_i / 6
```

### Hermite-Bessel — Hagan-West Method 7 — `src/interpolation/hermite_bessel.rs`

A **local** C^1 cubic-Hermite interpolant whose knot slopes `m_i` are the
slopes at `t_i` of the parabola passing through the three nearest knots
(de Boor 2001, Chapter IV). For interior `i`:

```
m_i = ( h_i * S_{i-1} + h_{i-1} * S_i ) / ( h_{i-1} + h_i )
```

The two boundary slopes use the standard three-point endpoint rule (the
slope at the end of the same parabola, evaluated one step away):

```
m_0     = ( (2*h_0 + h_1)         * S_0     - h_0     * S_1     ) / (h_0     + h_1)
m_{n-1} = ( (2*h_{n-2} + h_{n-3}) * S_{n-2} - h_{n-2} * S_{n-3} ) / (h_{n-3} + h_{n-2})
```

Once the slopes are in hand, segment `i` is evaluated with the cubic
Hermite basis. With local coordinate `u = (t - t_i) / h_i`,

```
H_00(u) =  2u^3 - 3u^2 + 1
H_10(u) =   u^3 - 2u^2 + u
H_01(u) = -2u^3 + 3u^2
H_11(u) =   u^3 -   u^2

y(t) = H_00(u) * y_i + h_i * H_10(u) * m_i
     + H_01(u) * y_{i+1} + h_i * H_11(u) * m_{i+1}
```

The interpolant is C^1 at every interior knot by construction (both
adjacent segments share `m_i` at `t = t_i`) and reproduces every quadratic
exactly on the interior — `m_i` is, by definition, the exact derivative of
the local parabola. Bessel-Hermite is **not** monotonicity-preserving in
general.

### Fritsch-Carlson monotone cubic — `src/interpolation/monotone_cubic.rs`

A C^1 piecewise-cubic Hermite interpolant whose slopes are chosen so that,
**when the input data is monotone**, the interpolant is monotone on every
segment (Fritsch & Carlson 1980). The slope at an interior knot uses the
Fritsch-Butland (1984) weighted harmonic mean — a refinement of the
original Fritsch-Carlson three-point rule that produces visibly smoother
slopes on real data:

```
if S_{i-1} * S_i <= 0:
    m_i = 0
else:
    m_i = 3 * (h_{i-1} + h_i)
        / ( (2*h_i + h_{i-1}) / S_{i-1} + (h_i + 2*h_{i-1}) / S_i )
```

Endpoint slopes use the same three-point formula as Hermite-Bessel,
followed by the standard PCHIP clamp (Dougherty-Edelman-Hyman 1989):

```
m_0 = ( (2*h_0 + h_1) * S_0 - h_0 * S_1 ) / (h_0 + h_1)
if sign(m_0) != sign(S_0):  m_0 = 0
if |m_0| > 3 * |S_0|:       m_0 = 3 * S_0
```

The monotonicity filter (Fritsch-Carlson 1980, §4): for each segment `i`
with `S_i != 0`, define

```
alpha = m_i / S_i,    beta = m_{i+1} / S_i
```

Fritsch & Carlson prove that the Hermite cubic on segment `i` is monotone
iff `(alpha, beta)` lies in a certain elliptic region in the closed
first quadrant. The standard sufficient condition uses the inscribed disc
of radius 3: if `alpha^2 + beta^2 > 9`, project both slopes radially:

```
tau     = 3 / sqrt(alpha^2 + beta^2)
m_i     := tau * alpha * S_i
m_{i+1} := tau * beta  * S_i
```

This preserves the slope signs while moving `(alpha, beta)` to the
boundary of the sufficient region. Evaluation then uses the cubic Hermite
basis above.

**Guarantee.** The interpolant is monotone on every segment whenever the
input data is monotone. With non-monotone data the filter still produces a
well-defined C^1 interpolant (zeroing the slope at every turning point);
no global monotonicity claim is made on non-monotone data.

### Steffen monotone — `src/interpolation/monotone_steffen.rs`

A **purely local** monotone Hermite cubic (Steffen 1990): the slope at
knot `i` depends only on the secants of the two adjacent segments. The
interior slope formula (Steffen 1990 equation 11) is

```
p_i = ( S_{i-1} * h_i + S_i * h_{i-1} ) / ( h_{i-1} + h_i )

if S_{i-1} * S_i <= 0:
    m_i = 0
else:
    m_i = ( sign(S_{i-1}) + sign(S_i) )
        * min( |S_{i-1}|, |S_i|, |p_i| / 2 )
```

The first arm zeroes the slope at turning points and at plateaux. The
second arm clamps the candidate slope to no more than twice the smaller
adjacent secant — a sufficient condition for monotonicity of the cubic
Hermite (Steffen 1990, §2; cf. Fritsch & Carlson 1980, Theorem 1).
When the two secants share a sign, `sign(S_{i-1}) + sign(S_i) = +/-2`,
restoring the factor that exactly reproduces a common-secant slope (the
linear-data limit). The divisor `2` in `|p_i|/2` comes from Steffen's
harmonic-mean slope bound.

Endpoint slopes are obtained by extrapolating the secant slopes (Steffen
1990, equations 12-13):

```
m_0     = S_0     + ( S_0     - S_1     ) * h_0     / ( h_0     + h_1 )
m_{n-1} = S_{n-2} + ( S_{n-2} - S_{n-3} ) * h_{n-2} / ( h_{n-3} + h_{n-2} )
```

followed by the same sign / magnitude limiter:

```
if sign(m_e) != sign(S_e):  m_e = 0
elif |m_e| > 2 * |S_e|:     m_e = 2 * S_e
```

Evaluation uses the cubic Hermite basis as before. Monotonicity preservation
is exact for monotone input; the locality (no global pre-pass, no spline
solve) is what distinguishes Steffen from Fritsch-Carlson and Hyman.

### Hyman 1983 monotonicity filter — `src/interpolation/monotone_hyman.rs`

A C^2 cubic spline is built through the knots (natural boundary by
default), the analytic slope of that spline is read off at every knot, and
each knot slope is **clamped to the local monotonicity envelope** before
re-evaluation as a piecewise cubic Hermite. The result is a C^1 piecewise
cubic that preserves the monotonicity of monotone input data while
retaining the smoothness profile of the underlying spline on intervals
where the spline is already monotone.

**Step 1 — base slopes.** Build a C^2 cubic spline (§ "Cubic spline"
above) through the knots and let `m_i = y'_spline(t_i)`. Because the base
spline is C^2 at every interior knot, the left- and right-derivative
agree there and `m_i` is unambiguous.

**Step 2 — interior filter** (`1 <= i <= n-2`):

```
if S_{i-1} * S_i > 0:
    m_i' = sign(S_{i-1}) * min( |m_i|, 3 * min(|S_{i-1}|, |S_i|) )
else:
    m_i' = 0                                    (turning point)
```

The clamp `|m_i'| <= 3 * min(|S_{i-1}|, |S_i|)` is the sufficient
monotonicity bound from Hyman (1983) §3 — a localised version of the
Fritsch-Carlson region: when both secants share a sign, the slope region
`|m_i| <= 3 * min(|S|, |S|)` lies inside the Fritsch-Carlson disc
`alpha^2 + beta^2 <= 9` for both adjacent segments.

**Step 3 — endpoint filter.** With only one adjacent secant the bound
degenerates to

```
if sign(m_0) == sign(S_0):
    m_0' = sign(S_0) * min(|m_0|, 3 * |S_0|)
else:
    m_0' = 0
```

and symmetrically at the right endpoint with `S_{n-2}` in place of `S_0`.

**Step 4 — evaluation.** Cubic Hermite on each segment using the filtered
slopes `m_i'` (same basis as Fritsch-Carlson / Steffen).

The filter only touches the slope at a knot — never the value — so the
result is **C^1 everywhere** but is **not C^2** at any knot where the
filter modifies the slope. On the segments where the unfiltered spline
slope already lies inside the envelope the filter is a no-op and the
interpolant coincides with the base spline.

The crate uses the Dougherty-Edelman-Hyman (1989) form of the clamp,
`|m_i'| <= 3 * min(|S_{i-1}|, |S_i|)`, rather than Hyman's original
`|m_i'| <= 3 * min(|m_i|, |S_{i-1}|, |S_i|)`. The two clamps coincide on
the typical case `|m_i| <= 3 * min(|S|, |S|)`; both are monotonicity-
preserving. The 1989 form is the version implemented by QuantLib's
`MonotonicCubicNaturalSpline` and the modern reference.

### Hagan–West Method 7 (monotone convex) — `src/interpolation/convex_monotone.rs`

**Source:** Hagan, P. S. & West, G., "Methods for constructing a yield
curve", *Wilmott Magazine*, May 2008, pp. 70-81 (Method 7); Hagan, P. S.
& West, G., "Interpolation methods for curve construction", *Applied
Mathematical Finance* 13(2):89-129 (2006), §3 (sequential bootstrap), §4
(the monotone-convex shape filter).

Method 7 of Hagan & West (2008) interpolates the **instantaneous forward
rate** `f(t) = -d/dt ln D(t)` piecewise, with a shape filter that
preserves the integral of `f` over each segment and keeps `f`
non-negative when the input discount factors are positive and monotone
non-increasing. The construction proceeds in four steps. Let the knots
be `(t_0, y_0), ..., (t_{n-1}, y_{n-1})` with strictly increasing `t_i`
and strictly positive `y_i`.

1. **Discrete forwards.** On each segment `[t_{i-1}, t_i]`,

   ```
   f_i = ( ln(y_{i-1}) - ln(y_i) ) / ( t_i - t_{i-1} ),    i = 1, ..., n-1.
   ```

2. **Instantaneous forwards at knots.** For an interior knot `i`, take
   the time-weighted midpoint of the two adjacent discrete forwards:

   ```
   fhat_i = (t_i - t_{i-1}) / (t_{i+1} - t_{i-1}) * f_{i+1}
          + (t_{i+1} - t_i) / (t_{i+1} - t_{i-1}) * f_i,
   ```

   for `1 <= i <= n-2`. The endpoint values are extrapolated linearly:

   ```
   fhat_0     = f_1     - 0.5 * (fhat_1     - f_1),
   fhat_{n-1} = f_{n-1} - 0.5 * (fhat_{n-2} - f_{n-1}).
   ```

3. **Monotonicity / convexity filter.** Each `fhat_i` is clipped to the
   positivity / convexity box so that the piecewise-quadratic segment
   forward is monotone on each segment:

   ```
   fhat_i_clipped = clamp( fhat_i, 0, 2 * min(f_i, f_{i+1}) )      (interior)
   fhat_0_clipped     = clamp( fhat_0,     0, 2 * f_1     )
   fhat_{n-1}_clipped = clamp( fhat_{n-1}, 0, 2 * f_{n-1} )
   ```

   Inside each segment, the basic quadratic ansatz with local coordinate
   `x = (t - t_i) / (t_{i+1} - t_i) in [0, 1]` is

   ```
   f(t) = g_0 * (1 - 4x + 3x^2) + g_1 * (-2x + 3x^2) + f_{i+1},
   ```

   with `g_0 = fhat_i - f_{i+1}` and `g_1 = fhat_{i+1} - f_{i+1}`. When
   `(2 g_0 + g_1)(g_0 + 2 g_1) > 0` the basic quadratic is non-monotone
   and is replaced by one of three two-piece shapes (HW 2008 §4) chosen
   by the sign and magnitude of `(g_0, g_1)`. Every alternative shape
   preserves the segment integral `INTEGRAL_0^1 g(x) dx = 0` by
   construction.

4. **Discount factor reconstruction.** On segment `i`,

   ```
   y(t) = y_i * exp( - INTEGRAL_{t_i}^{t} f(s) ds ),
   ```

   computed in closed form from the segment shape.

**Integral-preservation identity.** By construction every segment shape
satisfies

```
INTEGRAL_{t_i}^{t_{i+1}} f(s) ds = f_{i+1} * (t_{i+1} - t_i),
```

so `y(t_{i+1}) = y_i * exp(-f_{i+1} * (t_{i+1} - t_i)) = y_{i+1}`. The
knot values are reproduced exactly, and the interpolant is C^0 in the
discount-factor domain with a C^0 (piecewise-quadratic, possibly
piecewise-defined inside a single segment) instantaneous forward.

The method is **local**: the value at any `t` depends only on the four
neighbouring knots `(t_{i-1}, t_i, t_{i+1}, t_{i+2})`, so a perturbation
of a single market quote propagates only into the two adjacent segments
and the sequential bootstrap engine of §6 does not need to invoke its
outer iteration when this interpolator is used.

All ten interpolation methods above are implemented in
`src/interpolation/`.

---

## Curve views and conversions — `src/curves/`

**Source:** Hagan, P. S. & West, G., *Interpolation methods for curve
construction*, *Applied Mathematical Finance* 13(2):89-129 (2006), §2;
Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling*,
Volume I: Foundations and Vanilla Models, Atlantic Financial Press (2010),
§6.

A yield curve has four equivalent representations, connected by the
canonical identities (Hagan & West 2006, §2):

```
D(t)  =  exp(-z(t) * t)              (continuous compounding)
z(t)  = -ln D(t) / t                 (t > 0)
f(t)  = -d/dt ln D(t)                (instantaneous forward)
D(t)  =  exp( -INTEGRAL_0^t f(s) ds )
```

Any one of the four views fully determines the other three.

### Canonical `D` and the three views

The crate makes `D(t)` canonical and exposes `z`, `f`, and the par swap
rate as **views** over the same underlying `DiscountCurve`. The choice of
`D` as canonical is grounded in three observations:

1. `D(t)` is the only one whose value is directly observable in the market
   (the price of a zero-coupon bond) without an integral or a derivative.
2. `D(t)` is the only one whose anchor `D(0) = 1` is a trivial, convention-
   free identity. The zero rate `z(0)` is a `0/0` limit; the forward `f(0)`
   needs a one-sided derivative.
3. All bootstrap instruments price as discount-factor products. The
   bootstrap engine of §6 solves for `D` at each pillar.

Hagan & West (2006, §2) make the same choice: "the discount function `D(t)`
is the centrally interpolated object."

### Conversion formulas

The view types delegate to the parent `DiscountCurve`. Their evaluation
maps are:

```
ZeroCurve (continuous):     z(t)            = -ln D(t) / t
ZeroCurve (simple):         z(t)            =  ( 1/D(t) - 1 ) / t
ZeroCurve (periodic n):     z(t)            =  n * ( D(t)^(-1/(n*t)) - 1 )

ForwardCurve (instant):     f(t)            = -D'(t) / D(t)
ForwardCurve (simply):      L(t_1, t_2)     = ( D(t_1) / D(t_2) - 1 ) / tau

ParCurve (single-curve):    r_par(t_0, t_N) = ( D(t_0) - D(t_N) )
                                            / SUM_i tau_i * D(t_i)
```

where the par-rate `t_i` are the fixed-leg period end dates and `tau_i =
dc.year_fraction(t_{i-1}, t_i)` is the accrual under the leg's chosen
day-count. The simply-compounded forward `L(t_1, t_2)` uses
`tau = dc.year_fraction(d_1, d_2)` against the caller-supplied day-count
(not necessarily the curve's own).

For the instantaneous forward, `D'(t)` is computed from the interpolant's
analytic derivative when it exists (a `deriv` returning `Some`), and is
approximated by a centred finite difference otherwise (a `deriv` returning
`None`, e.g. for `PiecewiseConstantForward` at a knot).

### Round-trip identity at the nodes

The four views agree at every curve node. With `(t_i, D_i)` a node,

```
ZeroCurve::rate(t_i)        = Compounding.rate_from_discount(D_i, t_i)
                            = the continuous zero rate that maps D_i at t_i
ForwardCurve::forward_rate( t_i, t_j ) tau in dc
                            = ( D_i / D_j - 1 ) / tau                  (exact)
ParCurve::par_rate(t_0, t_N)
                            = ( D_0 - D_N ) / SUM_i tau_i D_{t_i}      (exact)
```

so every view round-trips to the canonical `D` at the nodes by
construction. Off-node, the views inherit the smoothness of the underlying
interpolant: a log-linear `D` produces a piecewise-constant `f` (a single-
sided derivative at the nodes), a `LinearInZero` interpolant produces a
continuous piecewise-linear `f`, the cubic spline produces a C^1 forward,
and so on.

### Knot validation

`DiscountCurve::new` validates that:

- at least two nodes are supplied;
- the first node is anchored at `(reference_date, 1.0)`;
- dates are strictly increasing;
- every discount factor is strictly positive and finite.

The chosen `Interpolation` method then validates its own additional
invariants (e.g. positivity for `LogLinear`, `t > 0` for `LinearInZero`).

The canonical curve and the four view types above are implemented in
`src/curves/`.

---

## Single-curve bootstrap — `src/bootstrap.rs`

**Source:** Hagan, P. S. & West, G., *Interpolation methods for curve
construction*, *Applied Mathematical Finance* 13(2):89-129 (2006), §3;
Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling*,
Volume I: Foundations and Vanilla Models, Atlantic Financial Press (2010),
§6.4.

The bootstrap engine constructs a `DiscountCurve` from a list of market
instruments by sequentially solving for the discount factor at each
instrument's pillar date so that the instrument re-prices to within
`tolerance` against the in-progress curve.

### Sequential algorithm

```
anchor:    (t_0, D_0) = (0, 1)

for k = 1, 2, ..., N:
    t_k          = daycount.year_fraction(reference_date, instruments[k-1].pillar())

    residual_fn(D):
        snapshot = CurveSnapshot over (t_0, ..., t_{k-1}, t_k)
                                with (D_0, ..., D_{k-1}, D)
        return instruments[k-1].residual(reference_date, snapshot)

    bracket(D)   = [D_guess * exp(-w), D_guess * exp(+w)]    w = config.bracket
    D_k          = brent_root(residual_fn, bracket, BrentConfig)

    append (t_k, D_k) to the running curve

return DiscountCurve::from_times_and_discounts(reference_date, daycount,
                                               times, discounts, method)
```

Instruments are ordered by pillar date (the bootstrap requires earlier
pillars to be solved first); ties on the pillar date are rejected. The
bracket initial guess `D_guess` is the previous-anchor flat-forward
extrapolation, which lands the bracket close to the true root on every
realistic market quote. If Brent fails to bracket the root, the bracket is
widened by doubling up to five times before the bootstrap raises
`BootstrapError::NoConvergence`.

The bootstrap engine consumes any `Instrument` enum variant uniformly via
the crate-private `InstrumentLike::residual` dispatch (§3 above for the
per-variant residual formulas).

### Outer iteration for non-local interpolators

For local interpolation methods — `Linear`, `LogLinear`, `LinearInZero`,
`PiecewiseConstantForward`, `MonotoneCubic`, `MonotoneSteffen` — the
value of the interpolant at one pillar depends only on the values at the
adjacent two or three pillars, and the single sequential pass solves the
bootstrap exactly. For non-local methods — the cubic spline, Hermite-
Bessel, and Hyman-filtered cubic — the global interpolant shifts as later
pillars are added, so the single pass leaves residuals on earlier
instruments. The engine handles this with an **outer iteration**: starting
from the single-pass solution, it re-solves each pillar against the latest
curve until the maximum nodal change `max_i |D_i^{new} - D_i^{old}|` drops
below `iter_tol`.

Convergence of the outer iteration on consistent market data follows from
the contractivity of the residual map under reasonable curve shapes; the
formal argument is laid out in Andersen & Piterbarg (2010, Vol. 1, §6.4)
and Hagan & West (2006, §3).

### Re-pricing certificate

On every successful `build`, every input instrument's residual against the
returned curve is `< config.tolerance`. This is the bootstrap's contract
with the caller: a curve returned by `Bootstrap::build` is one that
re-prices its inputs. The certificate is the audit-grade output —
a regulator or auditor can re-pull every quote, re-evaluate every residual,
and confirm the curve to the published tolerance.

Default configuration: `tolerance = 1e-12`, `max_iter = 100`,
`bracket = 0.5`, `iterative = true`, `iter_max = 8`, `iter_tol = 1e-14`.

---

## Multi-curve bootstrap — `src/multi_curve.rs`

**Source:** Bianchetti, M., *Two Curves, One Price: Pricing & Hedging
Interest Rate Derivatives Decoupling Forwarding and Discounting Yield
Curves*, *Risk Magazine*, August 2010, pp. 66-72; arXiv 0905.2770 (2009);
Mercurio, F., *Interest Rates and The Credit Crunch: New Formulas and
Market Models*, Bloomberg Portfolio Research Paper No. 2010-01-FRONTIERS
(February 2009); Ametrano, F. M. & Bianchetti, M., *Everything You Always
Wanted to Know About Multiple Interest Rate Curve Bootstrapping but Were
Afraid to Ask*, SSRN 2219548 (April 2013); Andersen, L. B. G. &
Piterbarg, V. V., *Interest Rate Modeling*, Volume I: Foundations and
Vanilla Models, Atlantic Financial Press (2010), §6.5-§6.6.

Post-2008, the discount and forward-projection roles of the yield curve
are separated. Cash flows are discounted on the **OIS** curve; IBOR-style
floating-leg forwards are projected from a **tenor-specific projection
curve**, one per fixing tenor (1M, 3M, 6M, 12M, ...). The basis between
projection curves of different tenors becomes a first-class market
observable.

### Algorithm

```
Step 1: D_OIS  <- Bootstrap(ois_instruments, ois_method)
        (built by the single-curve engine of §6)

Step 2: for each (tenor, instruments_tenor) in projection_instruments:
            D_proj_tenor  <- ProjectionBootstrap(instruments_tenor,
                                                 D_OIS,
                                                 projection_method)

Step 3: return MultiCurve { discount: D_OIS,
                            projection: [(tenor, D_proj_tenor), ...] }
```

The OIS curve is bootstrapped first because every projection-curve
instrument prices cash flows under the OIS discount and therefore needs
`D_OIS` already in hand. Each projection curve is bootstrapped
independently against the OIS-discount cash flows.

### Multi-curve pricing formulas

For a vanilla fixed-floating swap with float schedule
`[t_0, t_1, ..., t_N]` and float accruals `tau_i^float`, under a discount
curve `D_OIS` and a projection curve `D_proj` of the same tenor as the
float leg, the float-leg PV is

```
PV_float  =  SUM_i  tau_i^float * F_i * D_OIS(t_i)

F_i       =  ( D_proj(t_{i-1}) / D_proj(t_i) - 1 ) / tau_i^float
```

`F_i` is the simply-compounded forward rate implied by the projection
curve — the same simple-compounding identity as in the FRA (§3 above) but
read off `D_proj` rather than the discount curve. Unlike the single-curve
case (§6), the sum does **not** telescope: the discount factor multiplying
each period is `D_OIS(t_i)`, while the forward rate is built from
`D_proj`.

The fixed-leg PV uses the OIS curve only:

```
PV_fixed  =  rate * SUM_j  tau_j^fixed * D_OIS(t_j^fixed)
```

Equating the two legs yields the multi-curve par-swap equation:

```
rate * SUM_j tau_j^fixed * D_OIS(t_j^fixed)
    =  SUM_i tau_i^float * F_i * D_OIS(t_i)
```

so the multi-curve par swap rate is

```
r_par^MC  =  ( SUM_i tau_i^float * F_i * D_OIS(t_i) )
           / ( SUM_j tau_j^fixed * D_OIS(t_j^fixed) )
```

The residual passed to the projection-curve bootstrap is

```
residual  =  rate * SUM_j tau_j^fixed * D_OIS(t_j^fixed)
           - SUM_i tau_i^float * F_i * D_OIS(t_i)
```

zero at the bootstrap solution.

### Single-instrument projection residuals

For deposits, FRAs, and STIR futures driving a projection curve, the
residual is the single-curve identity evaluated on the projection curve
alone — these instruments pin a single forward rate which, by no
arbitrage in the multi-curve framework, is independent of the OIS discount
curve:

```
Deposit on projection curve:
    residual = D_proj(t_v) / D_proj(t_m) - (1 + rate * tau)

FRA on projection curve:
    residual = D_proj(t_s) / D_proj(t_e) - (1 + rate * tau)

Future on projection curve (forward rate r_fwd = r_quoted - convexity):
    residual = D_proj(t_s) / D_proj(t_e) - (1 + r_fwd * tau)
```

### Basis-swap deferral

Basis swaps pin the **relationship** between two projection curves and
require a joint solve: both projection curves move together to satisfy the
basis-swap residual. Joint solves across multiple projection curves are
deferred — they need a least-squares or Newton-style multi-curve solver
rather than the sequential anchor-by-anchor sweep used here. In the present
implementation, basis swaps are **rejected** as projection-curve
bootstrap instruments; they are supplied only after the projection curves
have been independently bootstrapped from their own tenor instruments, as
a re-pricing check that documents the inter-curve basis the resulting
projection curves imply. The Bianchetti (2010) and Ametrano-Bianchetti
(2013) papers discuss the joint-solve approach in detail.

The multi-curve engine is implemented in `src/multi_curve.rs`.

---

## Algorithm references

| Algorithm | Primary reference |
|---|---|
| Yield-curve bootstrap (single-curve) | Hagan, P. S. & West, G., "Interpolation methods for curve construction", *Applied Mathematical Finance* 13(2):89-129 (2006) |
| Yield-curve bootstrap (methods survey) | Hagan, P. S. & West, G., "Methods for constructing a yield curve", *Wilmott Magazine*, May 2008, pp. 70-81 |
| Monotone-convex interpolation (Method 7) | Hagan, P. S. & West, G., "Methods for constructing a yield curve", *Wilmott Magazine*, May 2008, pp. 70-81 (Method 7); "Interpolation methods for curve construction", *Applied Mathematical Finance* 13(2):89-129 (2006), §4 |
| Coupon-bond pricing and bootstrap | Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling, Volume I: Foundations and Vanilla Models*, Atlantic Financial Press (2010), §5.1-§5.2; ISDA, *2006 ISDA Definitions*, §6 |
| Multi-curve / OIS-discounted swap pricing | Bianchetti, M., "Two Curves, One Price", *Risk Magazine* (Aug 2010); arXiv 0905.2770 (2009) |
| Multi-curve / OIS-discounted reformulation | Mercurio, F., "Interest Rates and The Credit Crunch: New Formulas and Market Models", Bloomberg Portfolio Research Paper No. 2010-01-FRONTIERS (2009) |
| Multi-curve bootstrap (comprehensive treatment) | Ametrano, F. M. & Bianchetti, M., "Everything You Always Wanted to Know About Multiple Interest Rate Curve Bootstrapping but Were Afraid to Ask", SSRN 2219548 (2013) |
| OIS-discounted multi-curve bootstrap (textbook) | Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling, Volume I: Foundations and Vanilla Models*, Atlantic Financial Press (2010) |
| Day-count conventions | International Swaps and Derivatives Association, *2006 ISDA Definitions*, §4.16 |
| Act/Act (ICMA) day-count | International Capital Market Association, *Rule 251 — Actual/Actual (ICMA)*, ICMA Rulebook |
| Cubic spline (natural / clamped / not-a-knot) | de Boor, C., *A Practical Guide to Splines*, Revised Edition, Applied Mathematical Sciences vol. 27, Springer (2001), Chapter IV |
| Cubic spline (modern textbook) | Press, W. H., Teukolsky, S. A., Vetterling, W. T. & Flannery, B. P., *Numerical Recipes: The Art of Scientific Computing*, 3rd Edition, Cambridge University Press (2007) |
| Hermite-Bessel slopes | de Boor, C., *A Practical Guide to Splines*, Revised Edition, Springer (2001), Chapter IV |
| Fritsch-Carlson monotone cubic | Fritsch, F. N. & Carlson, R. E., "Monotone Piecewise Cubic Interpolation", *SIAM J. Numer. Anal.* 17(2):238-246 (1980) |
| Fritsch-Butland slope refinement | Fritsch, F. N. & Butland, J., "A method for constructing local monotone piecewise cubic interpolants", *SIAM J. Sci. Stat. Comput.* 5(2):300-304 (1984) |
| Steffen monotone cubic | Steffen, M., "A simple method for monotonic interpolation in one dimension", *Astronomy & Astrophysics* 239:443-450 (1990) |
| Hyman monotone filter | Hyman, J. M., "Accurate Monotonicity Preserving Cubic Interpolation", *SIAM J. Sci. Stat. Comput.* 4(4):645-654 (1983) |
| Hyman filter (1989 improvement) | Dougherty, R. L., Edelman, A. & Hyman, J. M., "Nonnegativity-, Monotonicity-, or Convexity-Preserving Cubic and Quintic Hermite Interpolation", *Mathematics of Computation* 52(186):471-494 (1989) |
| Brent root-finder | Brent, R. P., *Algorithms for Minimization Without Derivatives*, Prentice-Hall (1973) |
| Thomas tridiagonal solver | Thomas, L. H., *Elliptic Problems in Linear Differential Equations over a Network*, Watson Sci. Comput. Lab. Report, Columbia University (1949) |
| Gaussian elimination, Cholesky | Golub, G. H. & Van Loan, C. F., *Matrix Computations*, 4th Edition, Johns Hopkins University Press (2013), Chapters 3 and 4 |
| Numerical stability (LU and Cholesky) | Higham, N. J., *Accuracy and Stability of Numerical Algorithms*, 2nd Edition, SIAM (2002) |
| Calendar arithmetic | Hinnant, H., *chrono-Compatible Low-Level Date Algorithms*, <https://howardhinnant.github.io/date_algorithms.html> |
| Convexity adjustment for STIR futures | Hull, J. C., *Options, Futures, and Other Derivatives*, 10th Edition, Pearson (2018), §6.3 |

---

*Part of [Regit OS](https://www.regit.io) — the operating system for investment products. From Luxembourg.*
