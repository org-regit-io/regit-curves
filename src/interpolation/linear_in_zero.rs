// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Piecewise linear interpolation on the continuously-compounded zero rate.
//!
//! `LinearInZero` interpolates **linearly between zero rates** `z_i` derived
//! from discount-factor knots `(t_i, D_i)`, then reconstructs the discount
//! factor as `D(t) = exp(-z(t) * t)`. For `t > 0`,
//!
//! ```text
//! z(t)  = -ln(D(t)) / t,
//! z(t)  = (1 - w) * z_lo + w * z_hi,    w = (t - t_lo) / (t_hi - t_lo),
//! D(t)  = exp(-z(t) * t).
//! ```
//!
//! This is Hagan & West (2006)'s **Method 2** — "linear on continuously-
//! compounded zero rates" — and is identified in §4 of the paper as the
//! recommended default for "good behaviour combined with simplicity". Unlike
//! piecewise-constant-forward (Method 1, equivalent to log-linear on `D`),
//! the implied forward rate is piecewise *linear* and therefore continuous at
//! the knots — a meaningful smoothness gain at no algorithmic cost.
//!
//! # Anchor convention at `t = 0`
//!
//! The zero rate `z(t) = -ln(D(t)) / t` is undefined at `t = 0` (a `0/0`
//! limit). When the anchor knot `(t = 0, D = 1)` is supplied, we follow
//! Hagan & West's convention and **extend the first non-degenerate segment's
//! zero rate to the anchor** — i.e. we set `z_0 = z_1` so that the
//! interpolation across the first segment is the constant `z_1`. This makes
//! `D(t) = exp(-z_1 * t)` on `[0, t_1]`, which agrees with `D_1` at `t = t_1`
//! and with `D_0 = 1` at `t = 0`, and matches the standard textbook handling.
//! The anchor convention is mathematically arbitrary at a single point but
//! consistent with how a desk would read the front of the curve.
//!
//! # Invariants
//!
//! - At least two knots.
//! - Knot times strictly increasing.
//! - All `t >= 0` (zero rates are not defined for `t < 0`).
//! - All `D > 0` and finite (the natural invariant of the discount-factor
//!   domain).
//! - If `t_0 = 0` then `D_0 = 1` (anchor).
//!
//! # Extrapolation
//!
//! Flat extrapolation **in `z(t)`** outside the knot range. Below the first
//! non-zero knot this means `z(t) = z_1`; above the last knot, `z(t) = z_n`.
//! In the discount-factor domain this translates to `D(t) = exp(-z_n * t)`
//! for `t > t_n` — i.e. the right-tail discount factor decays exponentially
//! at the last knot's zero rate.
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §3 "Method 2" and §4
//!   "Comparison of methods" (Method 2 identified as the recommended default).
//! - Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling*, Vol. 1,
//!   Atlantic Financial Press (2010), §6.2.

use crate::errors::CurveError;

use super::Interpolator;

/// Piecewise linear interpolation on the continuously-compounded zero rate.
///
/// Stores the knot times and the **derived zero rates** `z_i = -ln(D_i) / t_i`
/// (with the `t = 0` anchor handled by extending `z_1` back to the anchor —
/// see the module-level documentation). Evaluation reconstructs
/// `D(t) = exp(-z(t) * t)` where `z(t)` is piecewise linear in `t`.
///
/// This is Hagan & West (2006)'s **Method 2** — their recommended default
/// for "good behaviour combined with simplicity" (§4 of the paper).
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::{Interpolator, LinearInZero};
///
/// // Flat zero rate z = 0.04: D(t) = exp(-0.04 * t) at every knot.
/// let interp = LinearInZero::new(&[
///     (0.0, 1.0),
///     (1.0, (-0.04_f64 * 1.0).exp()),
///     (2.0, (-0.04_f64 * 2.0).exp()),
/// ]).unwrap();
/// // Knot reproduction.
/// assert!((interp.eval(1.0) - (-0.04_f64).exp()).abs() < 1e-15);
/// // Midpoint at t = 1.5: D = exp(-0.04 * 1.5) (z is constant 0.04).
/// let mid = interp.eval(1.5);
/// let expected = (-0.04_f64 * 1.5).exp();
/// assert!((mid - expected).abs() < 1e-15);
/// ```
#[derive(Debug, Clone)]
pub struct LinearInZero {
    /// Knot times, strictly increasing.
    times: Vec<f64>,
    /// Continuously-compounded zero rate at each knot. At an anchor knot
    /// `t = 0` the entry equals the first non-degenerate segment's zero rate
    /// (see module docs).
    zero_rates: Vec<f64>,
}

impl LinearInZero {
    /// Builds a `LinearInZero` interpolant from a slice of `(t, D)` knots.
    ///
    /// Validation:
    ///
    /// - `knots.len() >= 2`.
    /// - Every `t` is finite and `>= 0`.
    /// - Times strictly increasing.
    /// - Every `D > 0` and finite.
    /// - If `t_0 = 0` then `D_0 = 1` (anchor).
    ///
    /// The derived zero rates are stored internally. At an anchor knot
    /// (`t = 0`, `D = 1`) the zero rate is set to that of the next knot —
    /// extending the first non-degenerate segment's rate back to the anchor.
    ///
    /// # Errors
    ///
    /// - [`CurveError::TooFewNodes`] if fewer than two knots are supplied.
    /// - [`CurveError::InvalidTime`] if any time is not finite, or is negative.
    /// - [`CurveError::DuplicateNode`] if two consecutive times are equal.
    /// - [`CurveError::NodesNotIncreasing`] if times are not strictly
    ///   increasing.
    /// - [`CurveError::NonPositiveDiscount`] if any discount is `<= 0` or
    ///   non-finite.
    /// - [`CurveError::AnchorNotUnit`] if the first knot has `t = 0` but
    ///   `D != 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::interpolation::LinearInZero;
    /// use regit_curves::CurveError;
    ///
    /// assert!(LinearInZero::new(&[(0.0, 1.0), (1.0, 0.95)]).is_ok());
    /// assert!(matches!(
    ///     LinearInZero::new(&[(0.0, 1.0)]).unwrap_err(),
    ///     CurveError::TooFewNodes { found: 1 },
    /// ));
    /// ```
    pub fn new(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
        if knots.len() < 2 {
            return Err(CurveError::TooFewNodes { found: knots.len() });
        }
        let n = knots.len();
        let mut times = Vec::with_capacity(n);
        let mut zero_rates = Vec::with_capacity(n);
        // First pass: validate, collect times, collect well-defined zero
        // rates (sentinel `NaN` at an anchor `t = 0` knot, to be filled in
        // from the next knot's rate below).
        for (i, &(t, d)) in knots.iter().enumerate() {
            if !t.is_finite() || t < 0.0 {
                return Err(CurveError::InvalidTime { t });
            }
            if !d.is_finite() || d <= 0.0 {
                return Err(CurveError::NonPositiveDiscount {
                    at_index: i,
                    value: d,
                });
            }
            if i > 0 {
                let prev = times[i - 1];
                // Exact equality is the correct test here — a duplicate
                // grid time is a structural defect of the input, not a
                // numerical approximation.
                #[allow(clippy::float_cmp)]
                let is_duplicate = t == prev;
                if is_duplicate {
                    return Err(CurveError::DuplicateNode { t });
                }
                if t < prev {
                    return Err(CurveError::NodesNotIncreasing { at_index: i });
                }
            }
            // Anchor handling: if `t == 0`, require `D == 1`. The zero rate
            // there is undefined; mark with `NaN` and fill in below.
            #[allow(clippy::float_cmp)]
            let is_anchor = t == 0.0;
            let z_i = if is_anchor {
                #[allow(clippy::float_cmp)]
                let d_is_unit = d == 1.0;
                if !d_is_unit {
                    return Err(CurveError::AnchorNotUnit);
                }
                f64::NAN
            } else {
                -d.ln() / t
            };
            times.push(t);
            zero_rates.push(z_i);
        }
        // Second pass: replace any `NaN` anchor zero rate with the next
        // knot's rate (Hagan & West's convention — extend the first
        // non-degenerate segment's `z` back to the anchor). The anchor can
        // only legally be at index 0; we validated that `t > 0` for any
        // later knot, so at most `zero_rates[0]` is `NaN`.
        if zero_rates[0].is_nan() {
            zero_rates[0] = zero_rates[1];
        }
        Ok(Self { times, zero_rates })
    }

    /// Returns the number of knots.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns `true` if the interpolant has no knots. Always `false` for a
    /// successfully constructed `LinearInZero` (which requires `>= 2` knots);
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

    /// Returns the piecewise-linear interpolated zero rate at `t` (flat
    /// outside the knot range).
    #[inline]
    fn zero_rate_at(&self, t: f64) -> f64 {
        let n = self.times.len();
        if t <= self.times[0] {
            return self.zero_rates[0];
        }
        if t >= self.times[n - 1] {
            return self.zero_rates[n - 1];
        }
        let i = self.locate(t);
        let t_lo = self.times[i];
        let t_hi = self.times[i + 1];
        let w = (t - t_lo) / (t_hi - t_lo);
        (1.0 - w) * self.zero_rates[i] + w * self.zero_rates[i + 1]
    }
}

impl Interpolator for LinearInZero {
    fn build(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
        Self::new(knots)
    }

    fn eval(&self, t: f64) -> f64 {
        // `D(t) = exp(-z(t) * t)`. At `t = 0` this is `exp(0) = 1` regardless
        // of the (extended) anchor zero rate, which matches the anchor.
        let z = self.zero_rate_at(t);
        (-z * t).exp()
    }

    fn deriv(&self, t: f64) -> Option<f64> {
        let n = self.times.len();
        let discount = self.eval(t);
        let zero = self.zero_rate_at(t);
        // Outside the knot range `z` is flat, so `z'(t) = 0`.
        if t <= self.times[0] || t >= self.times[n - 1] {
            return Some(-zero * discount);
        }
        // Inside a segment, `z` is linear with slope `(z_hi - z_lo) / dt`.
        let idx = self.locate(t);
        let t_lo = self.times[idx];
        let t_hi = self.times[idx + 1];
        let slope_z = (self.zero_rates[idx + 1] - self.zero_rates[idx]) / (t_hi - t_lo);
        // d/dt D(t) = D(t) * d/dt (-z(t) * t) = D(t) * (-(z'(t) * t + z(t))).
        Some(discount * (-(slope_z * t + zero)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Construction & validation ───────────────────────────────────────

    #[test]
    fn rejects_empty() {
        let err = LinearInZero::new(&[]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 0 }));
    }

    #[test]
    fn rejects_single_knot() {
        let err = LinearInZero::new(&[(0.0, 1.0)]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 1 }));
    }

    #[test]
    fn rejects_non_monotone_times() {
        let err = LinearInZero::new(&[(0.0, 1.0), (2.0, 0.9), (1.0, 0.95)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NodesNotIncreasing { at_index: 2 }
        ));
    }

    #[test]
    fn rejects_duplicate_times() {
        let err = LinearInZero::new(&[(0.0, 1.0), (1.0, 0.95), (1.0, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::DuplicateNode { .. }));
    }

    #[test]
    fn rejects_negative_value() {
        let err = LinearInZero::new(&[(0.0, 1.0), (1.0, -0.5)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_zero_value() {
        let err = LinearInZero::new(&[(0.0, 1.0), (1.0, 0.0)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_nan_value() {
        let err = LinearInZero::new(&[(0.0, 1.0), (1.0, f64::NAN)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_nan_time() {
        let err = LinearInZero::new(&[(0.0, 1.0), (f64::NAN, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn rejects_inf_time() {
        let err = LinearInZero::new(&[(0.0, 1.0), (f64::INFINITY, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn rejects_negative_time() {
        let err = LinearInZero::new(&[(-0.5, 1.0), (1.0, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn rejects_anchor_not_unit_discount() {
        let err = LinearInZero::new(&[(0.0, 0.99), (1.0, 0.95)]).unwrap_err();
        assert!(matches!(err, CurveError::AnchorNotUnit));
    }

    // ─── Evaluation correctness ──────────────────────────────────────────

    #[test]
    fn knot_reproduction_exact() {
        // Discount knots from a varied zero-rate curve so the test is real.
        let knots = [
            (0.0_f64, 1.0_f64),
            (0.5, (-0.05_f64 * 0.5).exp()),
            (1.0, (-0.04_f64 * 1.0).exp()),
            (2.0, (-0.045_f64 * 2.0).exp()),
            (5.0, (-0.05_f64 * 5.0).exp()),
        ];
        let interp = LinearInZero::new(&knots).unwrap();
        for &(t, d) in &knots {
            let v = interp.eval(t);
            assert!((v - d).abs() < 1e-15, "knot ({t}, {d}) -> {v}");
        }
    }

    #[test]
    fn midpoint_constant_zero_rate() {
        // Knots: (0, 1), (1, exp(-0.04)), (2, exp(-0.08)). All implied
        // zero rates are 0.04, so the interpolated z is constant 0.04 and
        // D(1.5) = exp(-0.04 * 1.5) = exp(-0.06) exactly.
        let interp = LinearInZero::new(&[
            (0.0_f64, 1.0_f64),
            (1.0, (-0.04_f64 * 1.0).exp()),
            (2.0, (-0.04_f64 * 2.0).exp()),
        ])
        .unwrap();
        let v = interp.eval(1.5);
        let expected = (-0.04_f64 * 1.5).exp();
        assert!((v - expected).abs() < 1e-15, "got {v}, expected {expected}");
    }

    #[test]
    fn midpoint_varying_zero_rate() {
        // Knots: (1, exp(-0.05)), (2, exp(-0.04 * 2)) -> z_1 = 0.05, z_2 = 0.04.
        // Then z(1.5) = 0.045 and D(1.5) = exp(-0.045 * 1.5) = exp(-0.0675).
        let interp = LinearInZero::new(&[
            (1.0_f64, (-0.05_f64 * 1.0).exp()),
            (2.0, (-0.04_f64 * 2.0).exp()),
        ])
        .unwrap();
        let v = interp.eval(1.5);
        let expected = (-0.045_f64 * 1.5).exp();
        assert!(
            (v - expected).abs() < 1e-15,
            "midpoint z = 0.045 check: got {v}, expected {expected}",
        );
        // Sanity: numerical value of `exp(-0.0675)`, agreeing to four
        // decimal places (the test really only needs the analytic
        // identity above; this is a sanity floor on the magnitude).
        assert!(
            (v - 0.9348_f64).abs() < 1e-3,
            "exp(-0.0675) magnitude: got {v}",
        );
    }

    #[test]
    fn anchor_t_zero_extension() {
        // With anchor (0, 1) and (1, exp(-0.05)), the implied z_1 = 0.05 and
        // by convention z(0) = z_1 = 0.05. So on [0, 1] the curve is
        // exp(-0.05 * t) and at e.g. t = 0.25, D = exp(-0.0125).
        let interp = LinearInZero::new(&[(0.0_f64, 1.0_f64), (1.0, (-0.05_f64).exp())]).unwrap();
        // Anchor itself.
        assert!((interp.eval(0.0) - 1.0).abs() < 1e-15);
        // Interior point on the first segment.
        let v = interp.eval(0.25);
        let expected = (-0.05_f64 * 0.25).exp();
        assert!((v - expected).abs() < 1e-15);
    }

    #[test]
    fn flat_extrapolation_in_z_right() {
        // (1, exp(-0.04)), (2, exp(-0.05 * 2)): z_1 = 0.04, z_2 = 0.05.
        // Right of t = 2 the zero rate is flat at z_2 = 0.05, so
        // D(3) = exp(-0.05 * 3).
        let interp = LinearInZero::new(&[
            (1.0_f64, (-0.04_f64 * 1.0).exp()),
            (2.0, (-0.05_f64 * 2.0).exp()),
        ])
        .unwrap();
        let v = interp.eval(3.0);
        let expected = (-0.05_f64 * 3.0).exp();
        assert!((v - expected).abs() < 1e-15, "got {v}, expected {expected}");
        // Also at a much larger t.
        let v_large = interp.eval(20.0);
        let expected_large = (-0.05_f64 * 20.0).exp();
        assert!((v_large - expected_large).abs() < 1e-15);
    }

    #[test]
    fn flat_extrapolation_in_z_left() {
        // (1, exp(-0.04)), (2, exp(-0.05 * 2)): z_1 = 0.04. Left of t = 1
        // (but with no anchor) the zero rate is flat at z_1 = 0.04, so
        // D(0.5) = exp(-0.04 * 0.5).
        let interp = LinearInZero::new(&[
            (1.0_f64, (-0.04_f64 * 1.0).exp()),
            (2.0, (-0.05_f64 * 2.0).exp()),
        ])
        .unwrap();
        let v = interp.eval(0.5);
        let expected = (-0.04_f64 * 0.5).exp();
        assert!((v - expected).abs() < 1e-15, "got {v}, expected {expected}");
    }

    #[test]
    fn monotone_decreasing_input_produces_positive_output() {
        // For a sensible discount-factor curve, D(t) stays in (0, 1].
        let knots = [
            (0.0_f64, 1.0_f64),
            (0.25, (-0.05_f64 * 0.25).exp()),
            (0.5, (-0.05_f64 * 0.5).exp()),
            (1.0, (-0.05_f64 * 1.0).exp()),
            (2.0, (-0.05_f64 * 2.0).exp()),
            (5.0, (-0.05_f64 * 5.0).exp()),
        ];
        let interp = LinearInZero::new(&knots).unwrap();
        let mut t = 0.0_f64;
        while t <= 5.0 {
            let v = interp.eval(t);
            assert!(v > 0.0 && v <= 1.0 + 1e-15, "out of (0, 1] at t = {t}: {v}");
            t += 0.01;
        }
    }

    // ─── Derivative correctness ──────────────────────────────────────────

    #[test]
    fn deriv_finite_difference_interior() {
        // Knots imply z_1 = 0.05, z_2 = 0.04, z_3 = 0.045. Pick t in the
        // first segment.
        let knots = [
            (1.0_f64, (-0.05_f64 * 1.0).exp()),
            (2.0, (-0.04_f64 * 2.0).exp()),
            (4.0, (-0.045_f64 * 4.0).exp()),
        ];
        let interp = LinearInZero::new(&knots).unwrap();
        let t = 1.5_f64;
        let dy_dt = interp.deriv(t).unwrap();
        let h = 1e-6_f64;
        let fd = (interp.eval(t + h) - interp.eval(t - h)) / (2.0 * h);
        assert!((dy_dt - fd).abs() < 1e-7, "analytic={dy_dt}, fd={fd}");
    }

    #[test]
    fn deriv_extrapolation_uses_flat_zero_rate() {
        // (1, exp(-0.04)), (2, exp(-0.05 * 2)). For t > 2, z(t) = 0.05 (flat),
        // so D(t) = exp(-0.05 * t) and dD/dt = -0.05 * D(t).
        let interp = LinearInZero::new(&[
            (1.0_f64, (-0.04_f64 * 1.0).exp()),
            (2.0, (-0.05_f64 * 2.0).exp()),
        ])
        .unwrap();
        let t = 3.0_f64;
        let d = interp.deriv(t).unwrap();
        let expected = -0.05_f64 * interp.eval(t);
        assert!((d - expected).abs() < 1e-12);
    }

    // ─── Trait & accessor ────────────────────────────────────────────────

    #[test]
    fn build_trait_method_equivalent_to_new() {
        let knots = [(0.0_f64, 1.0_f64), (1.0, (-0.04_f64).exp())];
        let a = LinearInZero::new(&knots).unwrap();
        let b = <LinearInZero as Interpolator>::build(&knots).unwrap();
        assert!((a.eval(0.5) - b.eval(0.5)).abs() < 1e-15);
    }

    #[test]
    fn len_and_is_empty() {
        let interp = LinearInZero::new(&[
            (0.0_f64, 1.0_f64),
            (1.0, (-0.04_f64).exp()),
            (2.0, (-0.045_f64 * 2.0).exp()),
        ])
        .unwrap();
        assert_eq!(interp.len(), 3);
        assert!(!interp.is_empty());
    }

    #[test]
    fn clone_yields_equivalent_interpolant() {
        let interp = LinearInZero::new(&[(0.0_f64, 1.0_f64), (1.0, (-0.04_f64).exp())]).unwrap();
        let copy = interp.clone();
        assert!((interp.eval(0.5) - copy.eval(0.5)).abs() < 1e-15);
    }
}
