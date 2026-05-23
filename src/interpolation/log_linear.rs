// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Piecewise log-linear interpolation.
//!
//! `LogLinear` performs **linear interpolation of `ln(y)`** between knots —
//! i.e. for `t` in a segment `[t_lo, t_hi]`,
//!
//! ```text
//! y(t) = exp( (1 - w) * ln(y_lo) + w * ln(y_hi) ),
//! with  w = (t - t_lo) / (t_hi - t_lo).
//! ```
//!
//! Equivalently — and this is the natural way to read it in the discount-
//! factor domain — a piecewise-log-linear `D(t)` is the same as a
//! **piecewise-constant continuously-compounded zero rate** `z(t)`. From
//! `D(t) = exp(-z(t) * t)`, the identity `ln D(t) = -z(t) * t` is linear in
//! `t` precisely when `z` is constant on the segment. The slope on that
//! segment is `-z`, so the implied piecewise-constant zero rate on
//! `[t_lo, t_hi]` is
//!
//! ```text
//! z = ( ln(y_lo) - ln(y_hi) ) / ( t_hi - t_lo ).
//! ```
//!
//! This makes `LogLinear` the conservative market default for a discount
//! curve: it is monotonicity-preserving (a monotone `y` produces a monotone
//! `y(t)`), and the implied instantaneous forward `f(t) = -d/dt ln D(t)` is
//! piecewise constant and equal to the segment zero rate.
//!
//! # Invariants
//!
//! - At least two knots.
//! - Knot times strictly increasing.
//! - Knot values strictly positive and finite — the natural invariant of
//!   the discount-factor domain.
//!
//! # Extrapolation
//!
//! Flat extrapolation in `ln y` outside the knot range — i.e. flat in the
//! continuously-compounded zero rate, the conservative market default. See
//! Hagan & West (2006), §3.2.
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §3.2. Identifies
//!   log-linear-on-`D` as Method 1 (equivalent to piecewise-constant forward
//!   on a continuously-compounded zero curve).
//! - Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling*, Vol. 1,
//!   Atlantic Financial Press (2010), §6.2.

use crate::errors::CurveError;

use super::Interpolator;

/// Piecewise log-linear interpolant over a set of `(t, y)` knots.
///
/// Equivalent to **linear interpolation of `ln(y)`** between knots —
/// equivalent in turn to **piecewise constant continuously-compounded zero
/// rate** in the discount-factor domain, since `ln(D(t)) = -z * t` is linear
/// in `t` precisely when `z` is piecewise constant.
///
/// Requires `y > 0` at every knot; this is the natural invariant of the
/// discount-factor domain (where it is most commonly used).
///
/// Flat-extrapolates in `ln(y)` outside the knot range (i.e. flat in the
/// continuously-compounded zero rate, the conservative market default —
/// see Hagan & West (2006), §3.2).
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::{Interpolator, LogLinear};
///
/// let interp = LogLinear::new(&[(0.0, 1.0), (1.0, 0.95), (2.0, 0.90)]).unwrap();
/// // Knot reproduction.
/// assert!((interp.eval(0.0) - 1.0).abs() < 1e-15);
/// assert!((interp.eval(1.0) - 0.95).abs() < 1e-15);
/// // Midpoint of [0, 1]: exp(0.5 * ln(0.95)) = sqrt(0.95).
/// assert!((interp.eval(0.5) - 0.95_f64.sqrt()).abs() < 1e-15);
/// ```
#[derive(Debug, Clone)]
pub struct LogLinear {
    /// Knot times, strictly increasing.
    times: Vec<f64>,
    /// `ln(y_i)` for each knot.
    log_values: Vec<f64>,
}

impl LogLinear {
    /// Builds a log-linear interpolant from a slice of `(t, y)` knots.
    ///
    /// Validation:
    ///
    /// - `knots.len() >= 2`.
    /// - `knots[i].0 < knots[i + 1].0` (strictly increasing times).
    /// - Every `t` and `y` is finite.
    /// - Every `y > 0`.
    ///
    /// # Errors
    ///
    /// - [`CurveError::TooFewNodes`] if fewer than two knots are supplied.
    /// - [`CurveError::InvalidTime`] if any time is not finite.
    /// - [`CurveError::DuplicateNode`] if two consecutive times are equal.
    /// - [`CurveError::NodesNotIncreasing`] if times are not strictly
    ///   increasing.
    /// - [`CurveError::NonPositiveDiscount`] if any value is `<= 0` or
    ///   non-finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::interpolation::LogLinear;
    /// use regit_curves::CurveError;
    ///
    /// assert!(LogLinear::new(&[(0.0, 1.0), (1.0, 0.95)]).is_ok());
    /// assert!(matches!(
    ///     LogLinear::new(&[(0.0, 1.0)]).unwrap_err(),
    ///     CurveError::TooFewNodes { found: 1 },
    /// ));
    /// ```
    pub fn new(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
        if knots.len() < 2 {
            return Err(CurveError::TooFewNodes { found: knots.len() });
        }
        let n = knots.len();
        let mut times = Vec::with_capacity(n);
        let mut log_values = Vec::with_capacity(n);
        for (i, &(t, y)) in knots.iter().enumerate() {
            if !t.is_finite() {
                return Err(CurveError::InvalidTime { t });
            }
            if !y.is_finite() || y <= 0.0 {
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
            log_values.push(y.ln());
        }
        Ok(Self { times, log_values })
    }

    /// Returns the number of knots.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns `true` if the interpolant has no knots. Always `false` for a
    /// successfully constructed `LogLinear` (which requires `>= 2` knots);
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

impl Interpolator for LogLinear {
    fn build(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
        Self::new(knots)
    }

    fn eval(&self, t: f64) -> f64 {
        let n = self.times.len();
        // Flat extrapolation outside the knot range — in the log domain
        // this is flat in `ln y`, i.e. flat `y`.
        if t <= self.times[0] {
            return self.log_values[0].exp();
        }
        if t >= self.times[n - 1] {
            return self.log_values[n - 1].exp();
        }
        let i = self.locate(t);
        let t_lo = self.times[i];
        let t_hi = self.times[i + 1];
        let w = (t - t_lo) / (t_hi - t_lo);
        let ly = (1.0 - w) * self.log_values[i] + w * self.log_values[i + 1];
        ly.exp()
    }

    fn deriv(&self, t: f64) -> Option<f64> {
        let n = self.times.len();
        // Flat extrapolation -> zero derivative outside the knot range.
        if t < self.times[0] || t > self.times[n - 1] {
            return Some(0.0);
        }
        let i = self.locate(t);
        let t_lo = self.times[i];
        let t_hi = self.times[i + 1];
        let slope_log = (self.log_values[i + 1] - self.log_values[i]) / (t_hi - t_lo);
        let y = self.eval(t);
        // d/dt y(t) = y(t) * d/dt ln y(t).
        Some(y * slope_log)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Construction & validation ───────────────────────────────────────

    #[test]
    fn rejects_empty() {
        let err = LogLinear::new(&[]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 0 }));
    }

    #[test]
    fn rejects_single_knot() {
        let err = LogLinear::new(&[(0.0, 1.0)]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 1 }));
    }

    #[test]
    fn rejects_non_monotone_times() {
        let err = LogLinear::new(&[(0.0, 1.0), (2.0, 0.9), (1.0, 0.95)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NodesNotIncreasing { at_index: 2 }
        ));
    }

    #[test]
    fn rejects_duplicate_times() {
        let err = LogLinear::new(&[(0.0, 1.0), (1.0, 0.95), (1.0, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::DuplicateNode { .. }));
    }

    #[test]
    fn rejects_negative_value() {
        let err = LogLinear::new(&[(0.0, 1.0), (1.0, -0.5)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_zero_value() {
        let err = LogLinear::new(&[(0.0, 1.0), (1.0, 0.0)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_nan_value() {
        let err = LogLinear::new(&[(0.0, 1.0), (1.0, f64::NAN)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_nan_time() {
        let err = LogLinear::new(&[(0.0, 1.0), (f64::NAN, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn rejects_inf_time() {
        let err = LogLinear::new(&[(0.0, 1.0), (f64::INFINITY, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    // ─── Evaluation correctness ──────────────────────────────────────────

    #[test]
    fn knot_reproduction_exact() {
        let knots = [(0.0, 1.0), (0.5, 0.97), (1.0, 0.95), (2.0, 0.90)];
        let interp = LogLinear::new(&knots).unwrap();
        for &(t, y) in &knots {
            let v = interp.eval(t);
            assert!((v - y).abs() < 1e-15, "knot ({t}, {y}) -> {v}");
        }
    }

    #[test]
    fn midpoint_identity() {
        // y(0) = 1.0, y(1) = 0.95 -> y(0.5) = sqrt(0.95) exactly.
        let interp = LogLinear::new(&[(0.0, 1.0), (1.0, 0.95)]).unwrap();
        let mid = interp.eval(0.5);
        let expected = (0.5_f64 * 0.95_f64.ln()).exp();
        assert!((mid - expected).abs() < 1e-15);
        // Equivalent check: it's the geometric mean.
        assert!((mid - 0.95_f64.sqrt()).abs() < 1e-15);
        // Stated as a log-domain identity:
        assert!((mid.ln() - 0.5 * 0.95_f64.ln()).abs() < 1e-15);
    }

    #[test]
    fn midpoint_in_segment_log_average() {
        // For a midpoint of any segment, ln(eval(mid)) = avg of endpoints' ln.
        let interp = LogLinear::new(&[(0.0, 1.0), (1.0, 0.95), (3.0, 0.85)]).unwrap();
        let mid = f64::midpoint(1.0, 3.0);
        let v = interp.eval(mid).ln();
        let expected = f64::midpoint(0.95_f64.ln(), 0.85_f64.ln());
        assert!((v - expected).abs() < 1e-15);
    }

    #[test]
    fn monotone_decreasing_input_produces_monotone_output() {
        // Typical discount-factor-like knots: monotonically decreasing in t.
        let knots = [
            (0.0, 1.0),
            (0.25, 0.9875),
            (0.5, 0.9752),
            (1.0, 0.9512),
            (2.0, 0.9048),
            (5.0, 0.7788),
        ];
        let interp = LogLinear::new(&knots).unwrap();
        let mut prev = interp.eval(0.0);
        let mut t = 0.01_f64;
        while t <= 5.0 {
            let v = interp.eval(t);
            assert!(
                v <= prev + 1e-15,
                "non-monotone at t = {t}: prev={prev}, v={v}"
            );
            prev = v;
            t += 0.01;
        }
    }

    #[test]
    fn flat_extrapolation_left() {
        let interp = LogLinear::new(&[(0.5, 0.97), (1.0, 0.95)]).unwrap();
        assert!((interp.eval(0.0) - 0.97).abs() < 1e-15);
        assert!((interp.eval(-100.0) - 0.97).abs() < 1e-15);
        assert!((interp.eval(f64::NEG_INFINITY) - 0.97).abs() < 1e-15);
    }

    #[test]
    fn flat_extrapolation_right() {
        let interp = LogLinear::new(&[(0.0, 1.0), (1.0, 0.95)]).unwrap();
        assert!((interp.eval(2.0) - 0.95).abs() < 1e-15);
        assert!((interp.eval(100.0) - 0.95).abs() < 1e-15);
        assert!((interp.eval(f64::INFINITY) - 0.95).abs() < 1e-15);
    }

    // ─── Derivative correctness ──────────────────────────────────────────

    #[test]
    fn deriv_finite_difference_interior() {
        // On a flat-rate (continuously-compounded r) curve: y(t) = exp(-r*t).
        // Then d/dt y = -r * y(t). Check this against the segment slope.
        let r = 0.05_f64;
        let knots = [
            (0.0, 1.0),
            (1.0, (-r * 1.0).exp()),
            (2.0, (-r * 2.0).exp()),
            (5.0, (-r * 5.0).exp()),
        ];
        let interp = LogLinear::new(&knots).unwrap();
        let t = 1.5_f64;
        let dy_dt = interp.deriv(t).unwrap();
        let h = 1e-6_f64;
        let fd = (interp.eval(t + h) - interp.eval(t - h)) / (2.0 * h);
        assert!((dy_dt - fd).abs() < 1e-7, "analytic={dy_dt}, fd={fd}");
        // Analytic check: -r * y(t).
        let expected = -r * interp.eval(t);
        assert!((dy_dt - expected).abs() < 1e-12);
    }

    #[test]
    fn deriv_zero_in_extrapolation_region() {
        let interp = LogLinear::new(&[(0.0, 1.0), (1.0, 0.95)]).unwrap();
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
        // `deriv` returns the right-slope at an interior knot. Construct
        // a curve where left- and right-slopes differ and check that
        // `deriv` returns the right segment's slope.
        let knots = [(0.0, 1.0), (1.0, 0.95), (2.0, 0.80)];
        let interp = LogLinear::new(&knots).unwrap();
        let d_at_1 = interp.deriv(1.0).unwrap();
        // Right-slope at t=1: segment [1, 2] has slope_log = ln(0.80) - ln(0.95).
        // d/dt y(1) (right) = y(1) * slope_log = 0.95 * (ln(0.80) - ln(0.95)).
        let expected_right = 0.95_f64 * (0.80_f64.ln() - 0.95_f64.ln());
        assert!(
            (d_at_1 - expected_right).abs() < 1e-12,
            "deriv at knot = {d_at_1}, expected right-slope = {expected_right}",
        );
    }

    // ─── Trait & accessor ────────────────────────────────────────────────

    #[test]
    fn build_trait_method_equivalent_to_new() {
        let knots = [(0.0, 1.0), (1.0, 0.95)];
        let a = LogLinear::new(&knots).unwrap();
        let b = <LogLinear as Interpolator>::build(&knots).unwrap();
        assert!((a.eval(0.5) - b.eval(0.5)).abs() < 1e-15);
    }

    #[test]
    fn len_and_is_empty() {
        let interp = LogLinear::new(&[(0.0, 1.0), (1.0, 0.95), (2.0, 0.9)]).unwrap();
        assert_eq!(interp.len(), 3);
        assert!(!interp.is_empty());
    }

    #[test]
    fn clone_yields_equivalent_interpolant() {
        let interp = LogLinear::new(&[(0.0, 1.0), (1.0, 0.95)]).unwrap();
        let copy = interp.clone();
        assert!((interp.eval(0.5) - copy.eval(0.5)).abs() < 1e-15);
    }
}
