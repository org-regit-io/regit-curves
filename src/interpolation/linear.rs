// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Piecewise linear interpolation.
//!
//! `Linear` performs **straight-line interpolation in the value domain**
//! between knots — i.e. for `t` in a segment `[t_lo, t_hi]`,
//!
//! ```text
//! y(t) = (1 - w) * y_lo + w * y_hi,
//! with  w = (t - t_lo) / (t_hi - t_lo).
//! ```
//!
//! This is the simplest non-trivial interpolant — Hagan & West's "Method 0"
//! — and the natural baseline against which the more sophisticated methods
//! are measured. The interpolant is C⁰ everywhere and C¹ on the open interior
//! of each segment, with kinks at the knots in general. Applied directly to
//! discount factors it produces a non-smooth (piecewise-constant-then-step)
//! instantaneous forward — the classical motivation for the smoother methods
//! in Hagan & West (2006), §3.
//!
//! No positivity constraint is imposed on `y`: `Linear` interpolates any
//! finite real-valued field (rates, spreads, basis adjustments, capitalisation
//! factors, ...). The only `y`-side invariant is that every value be finite.
//!
//! # Invariants
//!
//! - At least two knots.
//! - Knot times strictly increasing.
//! - Knot times and values both finite (no `NaN`, no `±∞`).
//!
//! # Extrapolation
//!
//! Flat extrapolation outside the knot range — `eval(t) = y_0` for
//! `t <= t_0` and `eval(t) = y_{n-1}` for `t >= t_{n-1}`. The derivative
//! in the extrapolation region is therefore `0`. This matches the
//! conservative market default used elsewhere in the crate (cf.
//! [`LogLinear`](super::LogLinear)).
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §3 (Method 0 —
//!   linear on rates, or on any field). Identified as the simplest
//!   non-trivial interpolant; the paper observes that applying it directly
//!   to discount factors produces non-smooth forwards, motivating the more
//!   sophisticated methods (Methods 1-8).
//! - Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling*, Vol. 1,
//!   Atlantic Financial Press (2010), §6.2.

use crate::errors::CurveError;

use super::Interpolator;

/// Piecewise linear interpolant over a set of `(t, y)` knots.
///
/// Linearly interpolates `y` between adjacent knots — Hagan & West's
/// "Method 0". The interpolant is C⁰ everywhere and C¹ on the open interior
/// of each segment, with kinks at the knots in general.
///
/// No positivity constraint is imposed on `y`: any finite real value is
/// accepted. This makes `Linear` suitable for interpolating rates, spreads,
/// basis adjustments, capitalisation factors, or any field whose values are
/// not constrained to be positive.
///
/// Flat-extrapolates outside the knot range (eval returns `y_0` below the
/// first knot and `y_{n-1}` above the last).
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::{Interpolator, Linear};
///
/// let interp = Linear::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 1.5)]).unwrap();
/// // Knot reproduction.
/// assert!((interp.eval(0.0) - 1.0).abs() < 1e-15);
/// assert!((interp.eval(1.0) - 2.0).abs() < 1e-15);
/// // Midpoint of [0, 1]: average of endpoints.
/// assert!((interp.eval(0.5) - 1.5).abs() < 1e-15);
/// ```
#[derive(Debug, Clone)]
pub struct Linear {
    /// Knot times, strictly increasing.
    times: Vec<f64>,
    /// Knot values, one per knot time.
    values: Vec<f64>,
}

impl Linear {
    /// Builds a linear interpolant from a slice of `(t, y)` knots.
    ///
    /// Validation:
    ///
    /// - `knots.len() >= 2`.
    /// - `knots[i].0 < knots[i + 1].0` (strictly increasing times).
    /// - Every `t` is finite.
    /// - Every `y` is finite (no positivity requirement).
    ///
    /// Non-finite `y` values (`NaN`, `±∞`) are rejected via
    /// [`CurveError::NonPositiveDiscount`]. The variant name leans on the
    /// crate's discount-factor heritage but is reused here for the broader
    /// "invalid value at node" case — there is no separate `InvalidValue`
    /// variant, and the field semantics (`at_index`, `value`) carry the
    /// information a caller needs to diagnose the input.
    ///
    /// # Errors
    ///
    /// - [`CurveError::TooFewNodes`] if fewer than two knots are supplied.
    /// - [`CurveError::InvalidTime`] if any time is not finite.
    /// - [`CurveError::DuplicateNode`] if two consecutive times are equal.
    /// - [`CurveError::NodesNotIncreasing`] if times are not strictly
    ///   increasing.
    /// - [`CurveError::NonPositiveDiscount`] if any value is non-finite
    ///   (`NaN` or `±∞`).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::interpolation::Linear;
    /// use regit_curves::CurveError;
    ///
    /// assert!(Linear::new(&[(0.0, 1.0), (1.0, 2.0)]).is_ok());
    /// assert!(matches!(
    ///     Linear::new(&[(0.0, 1.0)]).unwrap_err(),
    ///     CurveError::TooFewNodes { found: 1 },
    /// ));
    /// // Negative values are accepted (no positivity constraint).
    /// assert!(Linear::new(&[(0.0, -1.0), (1.0, 1.0)]).is_ok());
    /// ```
    pub fn new(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
        if knots.len() < 2 {
            return Err(CurveError::TooFewNodes { found: knots.len() });
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
        Ok(Self { times, values })
    }

    /// Returns the number of knots.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns `true` if the interpolant has no knots. Always `false` for a
    /// successfully constructed `Linear` (which requires `>= 2` knots);
    /// retained for `clippy::len_without_is_empty`.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Binary-searches the segment index `i` such that
    /// `times[i] <= t < times[i + 1]`. Returns `0` if `t <= times[0]` and
    /// `n - 2` if `t >= times[n - 1]` (so the result is always a valid
    /// segment index in `0..n - 1`).
    #[inline]
    fn locate(&self, t: f64) -> usize {
        // Both bounds are inclusive in their respective branches; the
        // returned value is a segment index, never an endpoint index.
        let n = self.times.len();
        if t <= self.times[0] {
            return 0;
        }
        if t >= self.times[n - 1] {
            return n - 2;
        }
        // Standard half-open binary search: find the largest i with
        // times[i] <= t.
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

impl Interpolator for Linear {
    fn build(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
        Self::new(knots)
    }

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
        let w = (t - t_lo) / (t_hi - t_lo);
        (1.0 - w) * self.values[i] + w * self.values[i + 1]
    }

    fn deriv(&self, t: f64) -> Option<f64> {
        let n = self.times.len();
        // Flat extrapolation -> zero derivative outside the knot range.
        if t < self.times[0] || t > self.times[n - 1] {
            return Some(0.0);
        }
        // Inside the knot range: return the right-segment slope. `locate`
        // already returns the segment starting at-or-below `t`, so for an
        // interior knot this is the right-slope by construction.
        let i = self.locate(t);
        let t_lo = self.times[i];
        let t_hi = self.times[i + 1];
        Some((self.values[i + 1] - self.values[i]) / (t_hi - t_lo))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Construction & validation ───────────────────────────────────────

    #[test]
    fn rejects_empty() {
        let err = Linear::new(&[]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 0 }));
    }

    #[test]
    fn rejects_single_knot() {
        let err = Linear::new(&[(0.0, 1.0)]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 1 }));
    }

    #[test]
    fn rejects_non_monotone_times() {
        let err = Linear::new(&[(0.0, 1.0), (2.0, 0.9), (1.0, 0.95)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NodesNotIncreasing { at_index: 2 }
        ));
    }

    #[test]
    fn rejects_duplicate_times() {
        let err = Linear::new(&[(0.0, 1.0), (1.0, 0.95), (1.0, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::DuplicateNode { .. }));
    }

    #[test]
    fn accepts_negative_value() {
        // Linear has no positivity constraint — negative values are valid.
        let interp = Linear::new(&[(0.0, 1.0), (1.0, -0.5)]).unwrap();
        assert!((interp.eval(0.0) - 1.0).abs() < 1e-15);
        assert!((interp.eval(1.0) - (-0.5)).abs() < 1e-15);
    }

    #[test]
    fn accepts_zero_value() {
        // Zero is a valid `y` for plain Linear.
        let interp = Linear::new(&[(0.0, 1.0), (1.0, 0.0)]).unwrap();
        assert!((interp.eval(1.0) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn rejects_nan_value() {
        let err = Linear::new(&[(0.0, 1.0), (1.0, f64::NAN)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_inf_value() {
        let err = Linear::new(&[(0.0, 1.0), (1.0, f64::INFINITY)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_nan_time() {
        let err = Linear::new(&[(0.0, 1.0), (f64::NAN, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn rejects_inf_time() {
        let err = Linear::new(&[(0.0, 1.0), (f64::INFINITY, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    // ─── Evaluation correctness ──────────────────────────────────────────

    #[test]
    fn knot_reproduction_exact() {
        let knots = [(0.0, 1.0), (0.5, 0.97), (1.0, 0.95), (2.0, 0.90)];
        let interp = Linear::new(&knots).unwrap();
        for &(t, y) in &knots {
            let v = interp.eval(t);
            assert!((v - y).abs() < 1e-15, "knot ({t}, {y}) -> {v}");
        }
    }

    #[test]
    fn midpoint_identity() {
        // y(0) = 1.0, y(1) = 2.0 -> y(0.5) = 1.5 exactly.
        let interp = Linear::new(&[(0.0, 1.0), (1.0, 2.0)]).unwrap();
        let mid = interp.eval(0.5);
        #[allow(clippy::float_cmp)]
        let exact = mid == 1.5;
        assert!(exact, "expected exact 1.5, got {mid}");
    }

    #[test]
    fn midpoint_in_segment_is_average() {
        // For a midpoint of any segment, eval(mid) = avg of endpoints.
        let interp = Linear::new(&[(0.0, 1.0), (1.0, 0.95), (3.0, 0.85)]).unwrap();
        let mid = f64::midpoint(1.0, 3.0);
        let v = interp.eval(mid);
        let expected = f64::midpoint(0.95, 0.85);
        assert!((v - expected).abs() < 1e-15);
    }

    #[test]
    fn evaluation_matches_closed_form() {
        // Pick a non-trivial segment and check the closed-form linear blend
        // at several interior points.
        let interp = Linear::new(&[(1.0, 10.0), (3.0, 4.0)]).unwrap();
        for &t in &[1.0_f64, 1.25, 1.5, 2.0, 2.75, 3.0] {
            let w = (t - 1.0) / (3.0 - 1.0);
            let expected = (1.0 - w) * 10.0 + w * 4.0;
            let v = interp.eval(t);
            assert!(
                (v - expected).abs() < 1e-15,
                "t={t}: got {v}, want {expected}"
            );
        }
    }

    #[test]
    fn flat_extrapolation_left() {
        let interp = Linear::new(&[(0.5, 0.97), (1.0, 0.95)]).unwrap();
        assert!((interp.eval(0.0) - 0.97).abs() < 1e-15);
        assert!((interp.eval(-100.0) - 0.97).abs() < 1e-15);
        assert!((interp.eval(f64::NEG_INFINITY) - 0.97).abs() < 1e-15);
    }

    #[test]
    fn flat_extrapolation_right() {
        let interp = Linear::new(&[(0.0, 1.0), (1.0, 0.95)]).unwrap();
        assert!((interp.eval(2.0) - 0.95).abs() < 1e-15);
        assert!((interp.eval(100.0) - 0.95).abs() < 1e-15);
        assert!((interp.eval(f64::INFINITY) - 0.95).abs() < 1e-15);
    }

    // ─── Derivative correctness ──────────────────────────────────────────

    #[test]
    fn deriv_matches_segment_slope() {
        // On `[1.0, 3.0]` with values 10.0 and 4.0, the slope is -3.0.
        let interp = Linear::new(&[(1.0, 10.0), (3.0, 4.0)]).unwrap();
        let s = interp.deriv(2.0).unwrap();
        assert!((s - (-3.0)).abs() < 1e-15);
    }

    #[test]
    fn deriv_finite_difference_interior() {
        // Finite-difference cross-check on the interior of a segment.
        let knots = [(0.0, 1.0), (1.0, 1.2), (2.0, 0.8), (5.0, 0.5)];
        let interp = Linear::new(&knots).unwrap();
        let t = 1.5_f64;
        let dy_dt = interp.deriv(t).unwrap();
        let h = 1e-6_f64;
        let fd = (interp.eval(t + h) - interp.eval(t - h)) / (2.0 * h);
        assert!((dy_dt - fd).abs() < 1e-9, "analytic={dy_dt}, fd={fd}");
        // Analytic check: slope on segment [1, 2] is (0.8 - 1.2) / 1 = -0.4.
        let expected = -0.4_f64;
        assert!((dy_dt - expected).abs() < 1e-15);
    }

    #[test]
    fn deriv_zero_in_extrapolation_region() {
        let interp = Linear::new(&[(0.0, 1.0), (1.0, 0.95)]).unwrap();
        // Below the first knot.
        let d_left = interp.deriv(-1.0).unwrap();
        assert!((d_left - 0.0).abs() < 1e-15);
        // Above the last knot.
        let d_right = interp.deriv(2.0).unwrap();
        assert!((d_right - 0.0).abs() < 1e-15);
    }

    #[test]
    fn deriv_at_knot_returns_right_slope() {
        // The interpolant is C^0 but not C^1 in general; by convention,
        // `deriv` returns the right-slope at an interior knot.
        let knots = [(0.0, 1.0), (1.0, 0.95), (2.0, 0.80)];
        let interp = Linear::new(&knots).unwrap();
        let d_at_1 = interp.deriv(1.0).unwrap();
        // Right-slope at t=1: segment [1, 2] has slope (0.80 - 0.95) / 1.
        let expected_right = 0.80_f64 - 0.95_f64;
        assert!(
            (d_at_1 - expected_right).abs() < 1e-15,
            "deriv at knot = {d_at_1}, expected right-slope = {expected_right}",
        );
    }

    // ─── Trait & accessors ───────────────────────────────────────────────

    #[test]
    fn build_trait_method_equivalent_to_new() {
        let knots = [(0.0, 1.0), (1.0, 2.0)];
        let a = Linear::new(&knots).unwrap();
        let b = <Linear as Interpolator>::build(&knots).unwrap();
        assert!((a.eval(0.5) - b.eval(0.5)).abs() < 1e-15);
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn len_and_is_empty() {
        let interp = Linear::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 1.5)]).unwrap();
        assert_eq!(interp.len(), 3);
        assert!(!interp.is_empty());
    }

    #[test]
    fn clone_yields_equivalent_interpolant() {
        let interp = Linear::new(&[(0.0, 1.0), (1.0, 2.0)]).unwrap();
        let copy = interp.clone();
        assert!((interp.eval(0.5) - copy.eval(0.5)).abs() < 1e-15);
    }
}
