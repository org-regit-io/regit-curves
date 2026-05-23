// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! C² cubic spline interpolation with natural, clamped, and not-a-knot
//! boundary conditions.
//!
//! A cubic spline interpolating `(t_i, y_i)` for `i = 0, ..., n-1` is a
//! piecewise cubic polynomial that is **C²** (continuous through second
//! derivatives) at every interior knot, and satisfies a chosen boundary
//! condition at the two endpoints. Following de Boor (2001) Chapter IV and
//! Press et al. (2007) §3.3, we parametrise the spline by its second
//! derivatives `M_i = y''(t_i)` at the knots.
//!
//! # Construction
//!
//! With segment widths `h_i = t_{i+1} - t_i`, requiring continuity of the
//! first derivative across each interior knot yields the tridiagonal system
//!
//! ```text
//!   h_{i-1} * M_{i-1}
//! + 2 * (h_{i-1} + h_i) * M_i
//! + h_i * M_{i+1}
//! = 6 * ((y_{i+1} - y_i) / h_i  -  (y_i - y_{i-1}) / h_{i-1})
//! ```
//!
//! for `i = 1, ..., n-2`. The boundary conditions close the system:
//!
//! - **Natural** — `M_0 = M_{n-1} = 0`. Reduces to a tridiagonal system of
//!   size `n - 2` over the interior second derivatives.
//! - **Clamped** — first derivative specified at each endpoint:
//!   `y'(t_0) = first`, `y'(t_{n-1}) = last`. Two boundary rows are added to
//!   the system (size `n`).
//! - **Not-a-knot** — the spline is a single cubic across the first two
//!   segments and across the last two; equivalently, the third derivative is
//!   continuous at `t_1` and `t_{n-2}`. Two boundary rows are added.
//!
//! Once the `M_i` are solved, the spline on segment `i` (`t in [t_i, t_{i+1}]`)
//! is — Press et al. (2007) eq. 3.3.3 —
//!
//! ```text
//! y(t) = ((t_{i+1} - t)^3 * M_i + (t - t_i)^3 * M_{i+1}) / (6 * h_i)
//!      + (y_i      / h_i  -  M_i      * h_i / 6) * (t_{i+1} - t)
//!      + (y_{i+1}  / h_i  -  M_{i+1}  * h_i / 6) * (t - t_i)
//! ```
//!
//! with first derivative
//!
//! ```text
//! y'(t) = -(t_{i+1} - t)^2 * M_i     / (2 * h_i)
//!       +  (t - t_i)^2     * M_{i+1} / (2 * h_i)
//!       + (y_{i+1} - y_i) / h_i
//!       + (M_i - M_{i+1}) * h_i / 6.
//! ```
//!
//! # Boundary-condition choice
//!
//! Not-a-knot is the default cubic spline in `QuantLib` and, more generally, is
//! the recommended choice for general-purpose interpolation when no
//! information about the endpoint slope is available — it avoids the
//! artificial linearisation imposed by the natural condition (`y'' = 0` at
//! the boundary). The natural spline remains the most widely used variant in
//! finance for its simplicity and the easy interpretation of its endpoint
//! curvature; it is also the unique cubic spline that minimises the strain
//! energy `\int (y'')^2 dt` over all C² interpolants of the data (de Boor
//! 2001, Theorem IV.5).
//!
//! # Degenerate cases
//!
//! - `n = 2`: a cubic spline through two points collapses to the straight
//!   line through them regardless of boundary condition. Implemented
//!   specially (no interior tridiagonal system to solve).
//! - `n = 3`: not-a-knot is a single cubic across the two segments. The
//!   construction still goes through the general code path.
//!
//! # Extrapolation
//!
//! Flat extrapolation in the value domain outside the knot range, matching
//! the rest of the interpolation family.
//!
//! # References
//!
//! - de Boor, C., *A Practical Guide to Splines*, Revised Edition, Springer
//!   (2001), Chapter IV.
//! - Press, W. H., Teukolsky, S. A., Vetterling, W. T. & Flannery, B. P.,
//!   *Numerical Recipes*, 3rd Edition, Cambridge University Press (2007),
//!   §3.3.
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §3.3 (Method 4).

use crate::errors::{CurveError, TypeError};
use crate::math::tridiag::thomas;

use super::Interpolator;

/// Boundary condition used when constructing a [`CubicSpline`].
///
/// All three classical conditions are exposed. **Not-a-knot is the default in
/// `QuantLib` and the recommended choice for general interpolation**; natural is
/// the most common variant in finance for its simplicity; clamped is the right
/// choice when an explicit endpoint slope is known a priori (e.g. matching a
/// short-rate at the curve anchor).
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::SplineBoundary;
///
/// let bc = SplineBoundary::NotAKnot;
/// assert_eq!(bc, SplineBoundary::NotAKnot);
/// let clamped = SplineBoundary::Clamped { first: 0.0, last: 0.0 };
/// assert!(matches!(clamped, SplineBoundary::Clamped { .. }));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplineBoundary {
    /// Natural spline: `y''(t_0) = y''(t_{n-1}) = 0`. The unique C² spline
    /// minimising the strain energy `\int (y'')^2 dt` over all C²
    /// interpolants (de Boor 2001, Theorem IV.5).
    Natural,
    /// Not-a-knot: the third derivative is continuous across `t_1` and
    /// `t_{n-2}`, so the spline is a single cubic across the first two and
    /// last two segments respectively. This is the `QuantLib` default.
    NotAKnot,
    /// Clamped spline: first derivatives at the endpoints are pinned to
    /// `first` and `last`.
    Clamped {
        /// Specified first derivative at `t_0`, i.e. `y'(t_0)`.
        first: f64,
        /// Specified first derivative at `t_{n-1}`, i.e. `y'(t_{n-1})`.
        last: f64,
    },
}

/// Piecewise cubic interpolant that is C² at every interior knot.
///
/// Constructed from a slice of `(t_i, y_i)` knots and a [`SplineBoundary`].
/// The pre-solved second derivatives at the knots are stored once at build
/// time; [`Interpolator::eval`] and [`Interpolator::deriv`] are then `O(log n)`
/// per call (binary search for the segment, constant-time polynomial
/// evaluation).
///
/// Flat-extrapolates in the value domain outside the knot range, matching the
/// rest of the [`crate::interpolation`] family.
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::{CubicSpline, Interpolator, SplineBoundary};
///
/// // Cubic spline interpolates a true cubic polynomial exactly.
/// // y = x^3 on five knots; check the value at an interior point.
/// let knots: Vec<(f64, f64)> = (0..5)
///     .map(|i| {
///         let x = f64::from(i);
///         (x, x * x * x)
///     })
///     .collect();
/// let spline = CubicSpline::new(&knots, SplineBoundary::NotAKnot).unwrap();
/// // y(2.5) = 2.5^3 = 15.625 exactly (up to round-off).
/// assert!((spline.eval(2.5) - 15.625).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct CubicSpline {
    /// Knot times, strictly increasing.
    times: Vec<f64>,
    /// Knot values `y_i`.
    values: Vec<f64>,
    /// Pre-solved second derivatives `M_i = y''(t_i)` at the knots.
    second: Vec<f64>,
    /// Boundary condition used to build this spline.
    boundary: SplineBoundary,
}

impl CubicSpline {
    /// Builds a cubic spline from a slice of `(t, y)` knots and a boundary
    /// condition.
    ///
    /// Validation:
    ///
    /// - `knots.len() >= 2`.
    /// - All `t` and `y` are finite.
    /// - Times are strictly increasing.
    /// - For [`SplineBoundary::Clamped`], `first` and `last` are finite.
    ///
    /// # Errors
    ///
    /// - [`CurveError::TooFewNodes`] if fewer than two knots are supplied.
    /// - [`CurveError::InvalidTime`] if any time is not finite.
    /// - [`CurveError::DuplicateNode`] if two consecutive times are equal.
    /// - [`CurveError::NodesNotIncreasing`] if times are not strictly
    ///   increasing.
    /// - [`CurveError::NonPositiveDiscount`] if any `y` is non-finite. The
    ///   variant name reflects the discount-curve use-case; here it signals
    ///   any non-finite knot value.
    /// - [`CurveError::Type`] wrapping [`TypeError::NonFinite`] if the clamped
    ///   boundary slopes are non-finite, or if the internal tridiagonal solve
    ///   reports a numerical failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::interpolation::{CubicSpline, SplineBoundary};
    /// use regit_curves::CurveError;
    ///
    /// assert!(
    ///     CubicSpline::new(&[(0.0, 1.0), (1.0, 0.95), (2.0, 0.90)], SplineBoundary::Natural)
    ///         .is_ok()
    /// );
    /// assert!(matches!(
    ///     CubicSpline::new(&[(0.0, 1.0)], SplineBoundary::Natural).unwrap_err(),
    ///     CurveError::TooFewNodes { found: 1 },
    /// ));
    /// ```
    pub fn new(knots: &[(f64, f64)], boundary: SplineBoundary) -> Result<Self, CurveError> {
        if knots.len() < 2 {
            return Err(CurveError::TooFewNodes { found: knots.len() });
        }
        if let SplineBoundary::Clamped { first, last } = boundary {
            if !first.is_finite() {
                return Err(CurveError::Type(TypeError::NonFinite {
                    name: "clamped boundary slope (first)",
                }));
            }
            if !last.is_finite() {
                return Err(CurveError::Type(TypeError::NonFinite {
                    name: "clamped boundary slope (last)",
                }));
            }
        }

        let n = knots.len();
        let mut times = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        for (i, &(t, y)) in knots.iter().enumerate() {
            if !t.is_finite() {
                return Err(CurveError::InvalidTime { t });
            }
            if !y.is_finite() {
                return Err(CurveError::NonPositiveDiscount {
                    at_index: i,
                    value: y,
                });
            }
            if i > 0 {
                let prev = times[i - 1];
                // Exact equality is the correct test here — a duplicate
                // grid time is a structural defect of the input, not a
                // numerical approximation. `clippy::float_cmp` flags this
                // by default; we suppress for this canonical use case.
                #[allow(clippy::float_cmp)]
                let is_duplicate = t == prev;
                if is_duplicate {
                    return Err(CurveError::DuplicateNode { t });
                }
                if t < prev {
                    return Err(CurveError::NodesNotIncreasing { at_index: i });
                }
            }
            times.push(t);
            values.push(y);
        }

        let second = solve_second_derivatives(&times, &values, boundary)?;

        Ok(Self {
            times,
            values,
            second,
            boundary,
        })
    }

    /// Returns the number of knots.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns `true` if the interpolant has no knots. Always `false` for a
    /// successfully constructed `CubicSpline` (which requires `>= 2` knots);
    /// retained for `clippy::len_without_is_empty`.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Boundary condition used at construction.
    #[must_use]
    #[inline]
    pub fn boundary(&self) -> SplineBoundary {
        self.boundary
    }

    /// Binary-searches the segment index `i` such that
    /// `times[i] <= t < times[i + 1]`. Returns `0` if `t <= times[0]` and
    /// `n - 2` if `t >= times[n - 1]` (so the result is always a valid
    /// segment index in `0..n - 1`).
    #[inline]
    fn locate(&self, t: f64) -> usize {
        let n = self.times.len();
        if t <= self.times[0] {
            return 0;
        }
        if t >= self.times[n - 1] {
            return n - 2;
        }
        let mut lo = 0_usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if self.times[mid] <= t {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

impl Interpolator for CubicSpline {
    /// Builds a cubic spline with the [`SplineBoundary::Natural`] boundary
    /// condition.
    ///
    /// The trait method has no way to carry a boundary choice; we pick the
    /// natural spline as the default for the trait path, since it has the
    /// fewest moving parts (no external slope input, no degenerate
    /// `n = 3` special-casing). Use [`CubicSpline::new`] directly to select
    /// not-a-knot or clamped.
    fn build(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
        Self::new(knots, SplineBoundary::Natural)
    }

    // The textbook spline letters `h, a, b, t` from Press et al. (2007) eq.
    // 3.3.3 are the canonical primary-source names; renaming them would harm
    // auditability against the cited reference.
    #[allow(clippy::many_single_char_names)]
    fn eval(&self, t: f64) -> f64 {
        let n = self.times.len();
        // Flat extrapolation outside the knot range.
        if t <= self.times[0] {
            return self.values[0];
        }
        if t >= self.times[n - 1] {
            return self.values[n - 1];
        }
        let i = self.locate(t);
        let t_lo = self.times[i];
        let t_hi = self.times[i + 1];
        let h = t_hi - t_lo;
        let m_lo = self.second[i];
        let m_hi = self.second[i + 1];
        let y_lo = self.values[i];
        let y_hi = self.values[i + 1];
        let a = t_hi - t;
        let b = t - t_lo;
        // Press et al. (2007) eq. 3.3.3.
        (a * a * a * m_lo + b * b * b * m_hi) / (6.0 * h)
            + (y_lo / h - m_lo * h / 6.0) * a
            + (y_hi / h - m_hi * h / 6.0) * b
    }

    // See note on `eval`.
    #[allow(clippy::many_single_char_names)]
    fn deriv(&self, t: f64) -> Option<f64> {
        let n = self.times.len();
        // Flat extrapolation -> zero derivative outside the knot range.
        if t < self.times[0] || t > self.times[n - 1] {
            return Some(0.0);
        }
        let i = self.locate(t);
        let t_lo = self.times[i];
        let t_hi = self.times[i + 1];
        let h = t_hi - t_lo;
        let m_lo = self.second[i];
        let m_hi = self.second[i + 1];
        let y_lo = self.values[i];
        let y_hi = self.values[i + 1];
        let a = t_hi - t;
        let b = t - t_lo;
        // Differentiating Press et al. (2007) eq. 3.3.3.
        Some(
            -(a * a) * m_lo / (2.0 * h)
                + (b * b) * m_hi / (2.0 * h)
                + (y_hi - y_lo) / h
                + (m_lo - m_hi) * h / 6.0,
        )
    }
}

/// Solves the tridiagonal system for the second derivatives `M_i` at the
/// knots, dispatching on the boundary condition.
///
/// Handles the `n = 2` degenerate case directly (straight line, all
/// `M_i = 0` regardless of boundary). For `n >= 3` the natural condition uses
/// a size-`n-2` tridiagonal solve (interior only); clamped and not-a-knot use
/// size-`n` tridiagonal systems with the two boundary rows folded in.
fn solve_second_derivatives(
    times: &[f64],
    values: &[f64],
    boundary: SplineBoundary,
) -> Result<Vec<f64>, CurveError> {
    let n = times.len();
    // n >= 2 enforced by caller.
    if n == 2 {
        // A spline through two points collapses to the line through them;
        // M_0 = M_1 = 0 regardless of boundary condition.
        return Ok(vec![0.0, 0.0]);
    }

    // Segment widths and slopes.
    let mut h = vec![0.0_f64; n - 1];
    let mut slope = vec![0.0_f64; n - 1];
    for i in 0..(n - 1) {
        h[i] = times[i + 1] - times[i];
        slope[i] = (values[i + 1] - values[i]) / h[i];
    }

    match boundary {
        SplineBoundary::Natural => solve_natural(&h, &slope, n),
        SplineBoundary::Clamped { first, last } => solve_clamped(&h, &slope, n, first, last),
        SplineBoundary::NotAKnot => solve_not_a_knot(&h, &slope, n),
    }
}

/// Solves for the interior second derivatives under the natural boundary
/// condition `M_0 = M_{n-1} = 0`. Tridiagonal system of size `n - 2`.
fn solve_natural(h: &[f64], slope: &[f64], n: usize) -> Result<Vec<f64>, CurveError> {
    // Interior count.
    let m = n - 2;
    if m == 0 {
        // n == 2 already short-circuited; n == 2 means m == 0. Safety.
        return Ok(vec![0.0; n]);
    }

    let mut sub = vec![0.0_f64; m];
    let mut diag = vec![0.0_f64; m];
    let mut sup = vec![0.0_f64; m];
    let mut rhs = vec![0.0_f64; m];
    for k in 0..m {
        // k corresponds to interior knot index i = k + 1.
        let i = k + 1;
        sub[k] = if k == 0 { 0.0 } else { h[i - 1] };
        diag[k] = 2.0 * (h[i - 1] + h[i]);
        sup[k] = if k == m - 1 { 0.0 } else { h[i] };
        rhs[k] = 6.0 * (slope[i] - slope[i - 1]);
    }
    let interior = thomas(&sub, &diag, &sup, &rhs).map_err(CurveError::from)?;

    let mut second = vec![0.0_f64; n];
    for (k, &m_k) in interior.iter().enumerate() {
        second[k + 1] = m_k;
    }
    Ok(second)
}

/// Solves for `M_i` under the clamped boundary condition with prescribed
/// endpoint slopes `y'(t_0) = first`, `y'(t_{n-1}) = last`. Tridiagonal system
/// of size `n`.
fn solve_clamped(
    h: &[f64],
    slope: &[f64],
    n: usize,
    first: f64,
    last: f64,
) -> Result<Vec<f64>, CurveError> {
    let mut sub = vec![0.0_f64; n];
    let mut diag = vec![0.0_f64; n];
    let mut sup = vec![0.0_f64; n];
    let mut rhs = vec![0.0_f64; n];

    // Boundary row at i = 0:
    //   2 * h_0 * M_0 + h_0 * M_1 = 6 * ((y_1 - y_0)/h_0 - first)
    //                              = 6 * (slope_0 - first).
    diag[0] = 2.0 * h[0];
    sup[0] = h[0];
    rhs[0] = 6.0 * (slope[0] - first);

    // Interior rows.
    for i in 1..(n - 1) {
        sub[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        sup[i] = h[i];
        rhs[i] = 6.0 * (slope[i] - slope[i - 1]);
    }

    // Boundary row at i = n - 1:
    //   h_{n-2} * M_{n-2} + 2 * h_{n-2} * M_{n-1} = 6 * (last - (y_{n-1} - y_{n-2})/h_{n-2})
    //                                            = 6 * (last - slope_{n-2}).
    let last_idx = n - 1;
    sub[last_idx] = h[last_idx - 1];
    diag[last_idx] = 2.0 * h[last_idx - 1];
    rhs[last_idx] = 6.0 * (last - slope[last_idx - 1]);

    thomas(&sub, &diag, &sup, &rhs).map_err(CurveError::from)
}

/// Solves for `M_i` under the not-a-knot boundary condition: the third
/// derivative is continuous at `t_1` and `t_{n-2}`. Reduces to a tridiagonal
/// system of size `n - 2` over the interior unknowns `(M_1, ..., M_{n-2})`,
/// then recovers `M_0` and `M_{n-1}` by linear extrapolation of `y''`.
///
/// Continuity of `y'''` at `t_1` means that `y''(t)` — which is piecewise
/// linear between knots — is linear across segments 0 and 1, with the same
/// slope `(M_2 - M_1)/h_1` on both. Therefore `M_0` is determined by
///
/// ```text
///   M_0 = ((h_0 + h_1) * M_1 - h_0 * M_2) / h_1,
/// ```
///
/// and symmetrically at the right end,
///
/// ```text
///   M_{n-1} = ((h_{n-3} + h_{n-2}) * M_{n-2} - h_{n-2} * M_{n-3}) / h_{n-3}.
/// ```
///
/// Substituting these expressions into the interior C¹ rows at `i = 1` and
/// `i = n - 2` removes `M_0` and `M_{n-1}` from the system, leaving a
/// tridiagonal system over `(M_1, ..., M_{n-2})` whose diagonal entries
/// `(h_{i-1} + h_i)(h_{i-1} + 2 h_i)` and `(h_{i-1} + h_i)(2 h_{i-1} + h_i)`
/// are strictly positive — the system is always well-posed (unlike
/// alternative reductions that produce `h_{i-1}^2 - h_i^2` on the diagonal
/// and become singular for uniform grids).
fn solve_not_a_knot(h: &[f64], slope: &[f64], n: usize) -> Result<Vec<f64>, CurveError> {
    if n == 3 {
        // Special case: with three knots, not-a-knot at both ends forces the
        // spline to be a single cubic across the two segments. The natural
        // closure is the unique quadratic through the three points (a cubic
        // with zero leading coefficient); that quadratic has constant second
        // derivative `M_0 = M_1 = M_2 = 2 (slope_1 - slope_0) / (h_0 + h_1)`,
        // i.e. twice its leading coefficient.
        let m_const = 2.0 * (slope[1] - slope[0]) / (h[0] + h[1]);
        return Ok(vec![m_const, m_const, m_const]);
    }

    // Reduced tridiagonal system over interior unknowns (M_1, ..., M_{n-2}).
    let m = n - 2;
    let mut sub = vec![0.0_f64; m];
    let mut diag = vec![0.0_f64; m];
    let mut sup = vec![0.0_f64; m];
    let mut rhs = vec![0.0_f64; m];

    // Row k = 0 corresponds to interior index i = 1. The standard interior
    // C¹ equation
    //   h_0 M_0 + 2 (h_0 + h_1) M_1 + h_1 M_2 = 6 (slope_1 - slope_0)
    // becomes, after substituting M_0 = ((h_0+h_1) M_1 - h_0 M_2)/h_1 and
    // multiplying by h_1,
    //   (h_0+h_1)(h_0 + 2 h_1) M_1 + (h_1^2 - h_0^2) M_2 = 6 h_1 (slope_1 - slope_0).
    diag[0] = (h[0] + h[1]) * (h[0] + 2.0 * h[1]);
    if m >= 2 {
        sup[0] = h[1] * h[1] - h[0] * h[0];
    }
    rhs[0] = 6.0 * h[1] * (slope[1] - slope[0]);

    // Standard interior rows for k = 1, ..., m - 2 (i.e. i = 2, ..., n - 3).
    if m >= 3 {
        for k in 1..(m - 1) {
            let i = k + 1;
            sub[k] = h[i - 1];
            diag[k] = 2.0 * (h[i - 1] + h[i]);
            sup[k] = h[i];
            rhs[k] = 6.0 * (slope[i] - slope[i - 1]);
        }
    }

    // Row k = m - 1 corresponds to interior index i = n - 2. The standard row
    //   h_{n-3} M_{n-3} + 2 (h_{n-3}+h_{n-2}) M_{n-2} + h_{n-2} M_{n-1}
    //     = 6 (slope_{n-2} - slope_{n-3})
    // becomes, after substituting
    //   M_{n-1} = ((h_{n-3}+h_{n-2}) M_{n-2} - h_{n-2} M_{n-3}) / h_{n-3}
    // and multiplying through by h_{n-3},
    //   (h_{n-3}^2 - h_{n-2}^2) M_{n-3} + (h_{n-3}+h_{n-2})(2 h_{n-3} + h_{n-2}) M_{n-2}
    //     = 6 h_{n-3} (slope_{n-2} - slope_{n-3}).
    let last_k = m - 1;
    let h_a = h[n - 3]; // h_{n-3}
    let h_b = h[n - 2]; // h_{n-2}
    if last_k >= 1 {
        sub[last_k] = h_a * h_a - h_b * h_b;
    }
    diag[last_k] = (h_a + h_b) * (2.0 * h_a + h_b);
    rhs[last_k] = 6.0 * h_a * (slope[n - 2] - slope[n - 3]);
    // For m == 1 (i.e. n == 3 is handled above; m == 1 actually corresponds
    // to n == 3 only), this branch is not reached; the diag is the sum of
    // the two row-0 and row-(m-1) constructions but they collapse to the
    // n == 3 special case.

    let interior = thomas(&sub, &diag, &sup, &rhs).map_err(CurveError::from)?;

    // Recover M_0 and M_{n-1} by linear extrapolation of y''.
    let m_1 = interior[0];
    let m_2 = if m >= 2 { interior[1] } else { m_1 };
    let m_0 = ((h[0] + h[1]) * m_1 - h[0] * m_2) / h[1];

    let m_nm2 = interior[last_k];
    let m_nm3 = if last_k >= 1 {
        interior[last_k - 1]
    } else {
        m_nm2
    };
    let m_nm1 = ((h[n - 3] + h[n - 2]) * m_nm2 - h[n - 2] * m_nm3) / h[n - 3];

    let mut second = vec![0.0_f64; n];
    second[0] = m_0;
    for (k, &v) in interior.iter().enumerate() {
        second[k + 1] = v;
    }
    second[n - 1] = m_nm1;
    Ok(second)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Construction & validation ───────────────────────────────────────

    #[test]
    fn rejects_empty() {
        let err = CubicSpline::new(&[], SplineBoundary::Natural).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 0 }));
    }

    #[test]
    fn rejects_single_knot() {
        let err = CubicSpline::new(&[(0.0, 1.0)], SplineBoundary::NotAKnot).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 1 }));
    }

    #[test]
    fn rejects_non_monotone_times() {
        let err = CubicSpline::new(
            &[(0.0, 1.0), (2.0, 0.9), (1.0, 0.95)],
            SplineBoundary::Natural,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CurveError::NodesNotIncreasing { at_index: 2 }
        ));
    }

    #[test]
    fn rejects_duplicate_times() {
        let err = CubicSpline::new(
            &[(0.0, 1.0), (1.0, 0.95), (1.0, 0.9)],
            SplineBoundary::Natural,
        )
        .unwrap_err();
        assert!(matches!(err, CurveError::DuplicateNode { .. }));
    }

    #[test]
    fn rejects_nan_value() {
        let err =
            CubicSpline::new(&[(0.0, 1.0), (1.0, f64::NAN)], SplineBoundary::Natural).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_nan_time() {
        let err =
            CubicSpline::new(&[(0.0, 1.0), (f64::NAN, 0.9)], SplineBoundary::Natural).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn rejects_inf_time() {
        let err = CubicSpline::new(&[(0.0, 1.0), (f64::INFINITY, 0.9)], SplineBoundary::Natural)
            .unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn rejects_nan_clamped_first() {
        let err = CubicSpline::new(
            &[(0.0, 1.0), (1.0, 0.9)],
            SplineBoundary::Clamped {
                first: f64::NAN,
                last: 0.0,
            },
        )
        .unwrap_err();
        assert!(matches!(err, CurveError::Type(TypeError::NonFinite { .. })));
    }

    #[test]
    fn rejects_nan_clamped_last() {
        let err = CubicSpline::new(
            &[(0.0, 1.0), (1.0, 0.9)],
            SplineBoundary::Clamped {
                first: 0.0,
                last: f64::INFINITY,
            },
        )
        .unwrap_err();
        assert!(matches!(err, CurveError::Type(TypeError::NonFinite { .. })));
    }

    // ─── Knot reproduction across all boundaries ──────────────────────────

    #[test]
    fn knot_reproduction_natural() {
        let knots = [(0.0, 1.0), (0.5, 0.97), (1.0, 0.95), (2.0, 0.90)];
        let spline = CubicSpline::new(&knots, SplineBoundary::Natural).unwrap();
        for &(t, y) in &knots {
            assert!(
                (spline.eval(t) - y).abs() < 1e-12,
                "natural: knot ({t}, {y}) -> {}",
                spline.eval(t)
            );
        }
    }

    #[test]
    fn knot_reproduction_not_a_knot() {
        let knots = [
            (0.0, 1.0),
            (0.5, 0.97),
            (1.0, 0.95),
            (2.0, 0.90),
            (3.5, 0.80),
        ];
        let spline = CubicSpline::new(&knots, SplineBoundary::NotAKnot).unwrap();
        for &(t, y) in &knots {
            assert!(
                (spline.eval(t) - y).abs() < 1e-12,
                "not-a-knot: knot ({t}, {y}) -> {}",
                spline.eval(t)
            );
        }
    }

    #[test]
    fn knot_reproduction_clamped() {
        let knots = [(0.0, 1.0), (0.5, 0.97), (1.0, 0.95), (2.0, 0.90)];
        let spline = CubicSpline::new(
            &knots,
            SplineBoundary::Clamped {
                first: -0.05,
                last: -0.02,
            },
        )
        .unwrap();
        for &(t, y) in &knots {
            assert!(
                (spline.eval(t) - y).abs() < 1e-12,
                "clamped: knot ({t}, {y}) -> {}",
                spline.eval(t)
            );
        }
    }

    // ─── Boundary-condition correctness ──────────────────────────────────

    #[test]
    fn natural_has_zero_second_derivative_at_endpoints() {
        let knots = [(0.0, 1.0), (0.5, 0.97), (1.0, 0.95), (2.0, 0.90)];
        let spline = CubicSpline::new(&knots, SplineBoundary::Natural).unwrap();
        // Direct check of stored M_0, M_{n-1} — should be identically zero.
        let n = spline.len();
        assert!(spline.second[0].abs() < 1e-15);
        assert!(spline.second[n - 1].abs() < 1e-15);
        // Cross-check via finite differences of the first derivative. The
        // truncation error of (f(t+h) - f(t))/h applied to y' is O(h * y'''),
        // and y''' on the first segment is (M_1 - M_0) / h_0 which is O(1)
        // for these knots — so a tolerance of about 10 * h suffices.
        let t0 = knots[0].0;
        let tn = knots[knots.len() - 1].0;
        let h = 1e-5;
        let d2_left = (spline.deriv(t0 + h).unwrap() - spline.deriv(t0).unwrap()) / h;
        let d2_right = (spline.deriv(tn).unwrap() - spline.deriv(tn - h).unwrap()) / h;
        assert!(d2_left.abs() < 1e-4, "d2 at left endpoint = {d2_left}");
        assert!(d2_right.abs() < 1e-4, "d2 at right endpoint = {d2_right}");
    }

    #[test]
    fn clamped_matches_specified_slopes_at_endpoints() {
        let first = 0.7_f64;
        let last = -1.3_f64;
        let knots = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.5), (3.0, 0.8)];
        let spline = CubicSpline::new(&knots, SplineBoundary::Clamped { first, last }).unwrap();
        let d0 = spline.deriv(0.0).unwrap();
        let dn = spline.deriv(3.0).unwrap();
        assert!(
            (d0 - first).abs() < 1e-12,
            "clamped first slope: got {d0}, want {first}"
        );
        assert!(
            (dn - last).abs() < 1e-12,
            "clamped last slope: got {dn}, want {last}"
        );
    }

    // ─── Cubic-polynomial reproduction ───────────────────────────────────

    #[test]
    fn not_a_knot_reproduces_cubic_exactly() {
        // y = x^3 on 5 knots. A not-a-knot spline collapses to the unique
        // cubic through any 4+ knots of a cubic polynomial — so the spline
        // equals x^3 everywhere.
        let knots: Vec<(f64, f64)> = (0..5)
            .map(|i| {
                let x = f64::from(i);
                (x, x * x * x)
            })
            .collect();
        let spline = CubicSpline::new(&knots, SplineBoundary::NotAKnot).unwrap();
        // Spot check the literal value at x = 2.5: 2.5^3 = 15.625.
        let v = spline.eval(2.5);
        assert!((v - 15.625).abs() < 1e-10, "y(2.5) = {v}, want 15.625");
        // A finer grid of points across the interior.
        for &t in &[0.25_f64, 0.75, 1.5, 2.0, 2.75, 3.1, 3.9] {
            let expected = t * t * t;
            let got = spline.eval(t);
            assert!(
                (got - expected).abs() < 1e-10,
                "y({t}) = {got}, want {expected}"
            );
        }
    }

    #[test]
    fn not_a_knot_three_knots_reproduces_quadratic() {
        // With only 3 knots, not-a-knot collapses to the unique quadratic
        // through the three points. y = x^2 + 1 on 3 knots → spline matches
        // x^2 + 1 everywhere.
        let knots = [(0.0, 1.0), (1.0, 2.0), (3.0, 10.0)];
        let spline = CubicSpline::new(&knots, SplineBoundary::NotAKnot).unwrap();
        for &t in &[0.0_f64, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            let expected = t * t + 1.0;
            let got = spline.eval(t);
            assert!(
                (got - expected).abs() < 1e-12,
                "y({t}) = {got}, want {expected}"
            );
        }
    }

    #[test]
    fn clamped_reproduces_cubic_with_exact_slopes() {
        // y = x^3, exact slopes y'(0) = 0, y'(4) = 48. Clamped spline with
        // these slopes recovers x^3 everywhere.
        let knots: Vec<(f64, f64)> = (0..5)
            .map(|i| {
                let x = f64::from(i);
                (x, x * x * x)
            })
            .collect();
        let spline = CubicSpline::new(
            &knots,
            SplineBoundary::Clamped {
                first: 0.0,
                last: 48.0,
            },
        )
        .unwrap();
        for &t in &[0.25_f64, 0.75, 1.5, 2.5, 3.1, 3.9] {
            let expected = t * t * t;
            let got = spline.eval(t);
            assert!(
                (got - expected).abs() < 1e-10,
                "clamped cubic: y({t}) = {got}, want {expected}"
            );
        }
    }

    // ─── n = 2 degenerate case ───────────────────────────────────────────

    #[test]
    fn two_knot_spline_is_linear_natural() {
        let knots = [(0.0, 1.0), (2.0, 0.0)];
        let spline = CubicSpline::new(&knots, SplineBoundary::Natural).unwrap();
        for &t in &[0.0, 0.5, 1.0, 1.5, 2.0] {
            let expected = 1.0 - t / 2.0;
            let got = spline.eval(t);
            assert!(
                (got - expected).abs() < 1e-15,
                "y({t}) = {got}, want {expected}"
            );
        }
    }

    #[test]
    fn two_knot_spline_is_linear_not_a_knot() {
        let knots = [(0.0, 1.0), (2.0, 0.0)];
        let spline = CubicSpline::new(&knots, SplineBoundary::NotAKnot).unwrap();
        // Slope is constant -0.5; check derivative & value at midpoint.
        let d = spline.deriv(1.0).unwrap();
        assert!((d + 0.5).abs() < 1e-15);
        assert!((spline.eval(1.0) - 0.5).abs() < 1e-15);
    }

    // ─── C^2 continuity at an interior knot ──────────────────────────────

    #[test]
    fn c2_continuity_at_interior_knot() {
        // The stored M_i is exactly y''(t_i); for a well-posed spline,
        // finite differencing y' on either side of an interior knot must
        // approach the same limit (= M_i).
        let knots = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.5), (3.0, 0.8), (4.5, 0.2)];
        let spline = CubicSpline::new(&knots, SplineBoundary::Natural).unwrap();
        let t_int = knots[2].0;
        let h = 1e-5;
        let d_left = (spline.deriv(t_int).unwrap() - spline.deriv(t_int - h).unwrap()) / h;
        let d_right = (spline.deriv(t_int + h).unwrap() - spline.deriv(t_int).unwrap()) / h;
        // Finite-difference truncation error is O(h * y''''); both one-sided
        // FDs share the same M_i at t_int, so the leading discrepancy is at
        // most h * (y''' jump), which for a true C^2 spline is zero up to
        // round-off — leaving O(h) truncation. A tolerance of 10 * h
        // accommodates this.
        assert!(
            (d_left - d_right).abs() < 1e-4,
            "C^2 mismatch at t={t_int}: left={d_left}, right={d_right}"
        );
    }

    // ─── Derivative correctness ──────────────────────────────────────────

    #[test]
    fn deriv_finite_difference_cubic() {
        // y = x^3 on 5 knots; not-a-knot reproduces it exactly. Then
        // deriv(t) should equal 3 t^2 to within numerical noise.
        let knots: Vec<(f64, f64)> = (0..5)
            .map(|i| {
                let x = f64::from(i);
                (x, x * x * x)
            })
            .collect();
        let spline = CubicSpline::new(&knots, SplineBoundary::NotAKnot).unwrap();
        for &t in &[0.5_f64, 1.5, 2.5, 3.5] {
            let expected = 3.0 * t * t;
            let got = spline.deriv(t).unwrap();
            assert!(
                (got - expected).abs() < 1e-10,
                "y'({t}) = {got}, want {expected}"
            );
        }
    }

    #[test]
    fn deriv_zero_in_extrapolation_region() {
        let knots = [(0.0, 1.0), (1.0, 0.95), (2.0, 0.9)];
        let spline = CubicSpline::new(&knots, SplineBoundary::Natural).unwrap();
        assert!(spline.deriv(-1.0).unwrap().abs() < 1e-15);
        assert!(spline.deriv(3.0).unwrap().abs() < 1e-15);
    }

    // ─── Extrapolation ───────────────────────────────────────────────────

    #[test]
    fn flat_extrapolation() {
        let knots = [(0.0, 1.0), (1.0, 0.95), (2.0, 0.90)];
        let spline = CubicSpline::new(&knots, SplineBoundary::Natural).unwrap();
        assert!((spline.eval(-100.0) - 1.0).abs() < 1e-15);
        assert!((spline.eval(100.0) - 0.90).abs() < 1e-15);
    }

    // ─── Trait & accessors ───────────────────────────────────────────────

    #[test]
    fn build_trait_method_returns_natural_default() {
        let knots = [(0.0, 1.0), (0.5, 0.97), (1.0, 0.95)];
        let via_trait = <CubicSpline as Interpolator>::build(&knots).unwrap();
        let direct = CubicSpline::new(&knots, SplineBoundary::Natural).unwrap();
        // Match at an off-knot evaluation; identical boundary => identical spline.
        assert!((via_trait.eval(0.25) - direct.eval(0.25)).abs() < 1e-15);
        assert_eq!(via_trait.boundary(), SplineBoundary::Natural);
    }

    #[test]
    fn len_and_is_empty() {
        let spline = CubicSpline::new(
            &[(0.0, 1.0), (1.0, 0.95), (2.0, 0.9)],
            SplineBoundary::Natural,
        )
        .unwrap();
        assert_eq!(spline.len(), 3);
        assert!(!spline.is_empty());
    }

    #[test]
    fn clone_yields_equivalent_interpolant() {
        let spline = CubicSpline::new(
            &[(0.0, 1.0), (0.5, 0.97), (1.0, 0.95)],
            SplineBoundary::NotAKnot,
        )
        .unwrap();
        let copy = spline.clone();
        assert!((spline.eval(0.25) - copy.eval(0.25)).abs() < 1e-15);
        assert_eq!(spline.boundary(), copy.boundary());
    }
}
