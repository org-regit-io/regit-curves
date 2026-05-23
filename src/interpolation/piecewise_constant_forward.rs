// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Piecewise-constant instantaneous-forward interpolation.
//!
//! `PiecewiseConstantForward` is the Hagan & West (2006) "Method 1"
//! interpolant viewed through the **instantaneous-forward** lens. Given a
//! discount-factor table `(t_i, D_i)`, it defines, on each segment
//! `(t_i, t_{i+1})`, a constant instantaneous forward rate
//!
//! ```text
//! f_i = ( ln(D_i) - ln(D_{i+1}) ) / ( t_{i+1} - t_i )
//! ```
//!
//! and reconstructs the discount factor on that segment as
//!
//! ```text
//! D(t) = D_i * exp( -f_i * (t - t_i) ),    t_i <= t <= t_{i+1}.
//! ```
//!
//! Mathematically this is **identical** to log-linear interpolation on `D`:
//! the three statements
//!
//! - log-linear on the discount factor `D`,
//! - piecewise constant continuously-compounded zero rate `z`,
//! - piecewise constant instantaneous forward `f`,
//!
//! all describe the same `D(t)`. The difference between this struct and
//! [`super::LogLinear`] is purely **semantic** — what `eval` and `deriv`
//! report and what auxiliary accessors expose:
//!
//! | Method                       | `eval(t)` | `deriv(t)`        | Extra            |
//! |------------------------------|-----------|-------------------|------------------|
//! | `LogLinear`                  | `D(t)`    | `dy/dt = -f * D`  | (none)           |
//! | `PiecewiseConstantForward`   | `D(t)`    | `-f(t) * D(t)`    | [`PiecewiseConstantForward::forward_rate`] |
//!
//! Both `deriv` values coincide pointwise away from knots; the distinction
//! exists because the forward-rate view makes the discontinuity at a knot
//! explicit and exposes the segment forward as a first-class quantity via
//! [`PiecewiseConstantForward::forward_rate`].
//!
//! # Invariants
//!
//! - At least two knots.
//! - Knot times strictly increasing and finite.
//! - Knot discount factors strictly positive and finite — the natural
//!   invariant of the discount-factor domain.
//!
//! # Extrapolation
//!
//! Flat extrapolation in the **forward rate**: for `t < t_0` the segment
//! forward `f_0` is reused (so `D(t)` continues with the same exponential
//! decay backwards from `t_0`), and for `t > t_{n-1}` the last segment
//! forward `f_{n-2}` is reused. The discount factor itself is continuous
//! everywhere — extrapolation simply extends the boundary segments'
//! exponential curves outwards.
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §3, "Method 1 —
//!   Linear on log of discount factors / piecewise-constant forwards".
//! - Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling*, Vol. 1,
//!   Atlantic Financial Press (2010), §6.2.

use crate::errors::CurveError;

use super::Interpolator;

/// Piecewise-constant instantaneous-forward interpolant over a set of
/// `(t, D)` discount-factor knots.
///
/// On each segment `(t_i, t_{i+1})`, the instantaneous forward
/// `f(t) = -d/dt ln D(t)` is held constant at
/// `f_i = (ln D_i - ln D_{i+1}) / (t_{i+1} - t_i)`. Discount factors between
/// knots follow `D(t) = D_i * exp(-f_i * (t - t_i))`.
///
/// Equivalent to [`super::LogLinear`] on the same knots in the `D` domain;
/// this type exposes the **forward-rate** view as a first-class quantity via
/// [`PiecewiseConstantForward::forward_rate`].
///
/// Requires `D > 0` at every knot; this is the natural invariant of the
/// discount-factor domain.
///
/// Flat-extrapolates in the **forward rate** outside the knot range — the
/// boundary segments' exponential curves are extended outwards. See the
/// module docs for the semantics.
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::{Interpolator, PiecewiseConstantForward};
///
/// // Two flat-forward segments at f0 = 0.04 and f1 = 0.06.
/// let knots = [
///     (0.0_f64, 1.0_f64),
///     (1.0_f64, (-0.04_f64).exp()),
///     (2.0_f64, (-0.10_f64).exp()),
/// ];
/// let interp = PiecewiseConstantForward::new(&knots).unwrap();
/// // Knot reproduction.
/// assert!((interp.eval(0.0) - 1.0).abs() < 1e-15);
/// // Mid-second-segment: D(1.5) = exp(-0.04 - 0.06 * 0.5) = exp(-0.07).
/// assert!((interp.eval(1.5) - (-0.07_f64).exp()).abs() < 1e-15);
/// // Segment forward rates.
/// assert!((interp.forward_rate(0.5) - 0.04).abs() < 1e-15);
/// assert!((interp.forward_rate(1.5) - 0.06).abs() < 1e-15);
/// ```
#[derive(Debug, Clone)]
pub struct PiecewiseConstantForward {
    /// Knot times, strictly increasing.
    times: Vec<f64>,
    /// Knot discount factors, strictly positive.
    discounts: Vec<f64>,
    /// Segment forwards: `forwards[i]` is the constant instantaneous forward
    /// on `(times[i], times[i + 1])`. Length `times.len() - 1`.
    forwards: Vec<f64>,
}

impl PiecewiseConstantForward {
    /// Builds a piecewise-constant-forward interpolant from a slice of
    /// `(t, D)` discount-factor knots.
    ///
    /// Validation:
    ///
    /// - `knots.len() >= 2`.
    /// - `knots[i].0 < knots[i + 1].0` (strictly increasing times).
    /// - Every `t` and `D` is finite.
    /// - Every `D > 0`.
    ///
    /// # Errors
    ///
    /// - [`CurveError::TooFewNodes`] if fewer than two knots are supplied.
    /// - [`CurveError::InvalidTime`] if any time is not finite.
    /// - [`CurveError::DuplicateNode`] if two consecutive times are equal.
    /// - [`CurveError::NodesNotIncreasing`] if times are not strictly
    ///   increasing.
    /// - [`CurveError::NonPositiveDiscount`] if any discount is `<= 0` or
    ///   non-finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::interpolation::PiecewiseConstantForward;
    /// use regit_curves::CurveError;
    ///
    /// assert!(PiecewiseConstantForward::new(&[(0.0, 1.0), (1.0, 0.95)]).is_ok());
    /// assert!(matches!(
    ///     PiecewiseConstantForward::new(&[(0.0, 1.0)]).unwrap_err(),
    ///     CurveError::TooFewNodes { found: 1 },
    /// ));
    /// ```
    pub fn new(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
        if knots.len() < 2 {
            return Err(CurveError::TooFewNodes { found: knots.len() });
        }
        let n = knots.len();
        let mut times = Vec::with_capacity(n);
        let mut discounts = Vec::with_capacity(n);
        for (i, &(t, d)) in knots.iter().enumerate() {
            if !t.is_finite() {
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
            discounts.push(d);
        }
        let mut forwards = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let dt = times[i + 1] - times[i];
            let f = (discounts[i].ln() - discounts[i + 1].ln()) / dt;
            forwards.push(f);
        }
        Ok(Self {
            times,
            discounts,
            forwards,
        })
    }

    /// Returns the number of knots.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns `true` if the interpolant has no knots. Always `false` for a
    /// successfully constructed `PiecewiseConstantForward` (which requires
    /// `>= 2` knots); retained for `clippy::len_without_is_empty`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::interpolation::PiecewiseConstantForward;
    ///
    /// let interp = PiecewiseConstantForward::new(&[(0.0, 1.0), (1.0, 0.95)]).unwrap();
    /// assert!(!interp.is_empty());
    /// ```
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Returns the instantaneous forward `f(t)` at `t`.
    ///
    /// On the interior, `f(t) = f_i` for `t` in `(t_i, t_{i+1}]`. At a knot
    /// `t = t_i` (interior), the right-segment forward `f_i` is returned —
    /// the convention is "right-continuous at knots". For
    /// `t <= t_0` the first-segment forward `f_0` is returned; for
    /// `t > t_{n-1}` the last-segment forward `f_{n-2}` is returned (flat
    /// extrapolation in the forward rate).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::interpolation::PiecewiseConstantForward;
    ///
    /// let knots = [
    ///     (0.0_f64, 1.0_f64),
    ///     (1.0_f64, (-0.04_f64).exp()),
    ///     (2.0_f64, (-0.10_f64).exp()),
    /// ];
    /// let interp = PiecewiseConstantForward::new(&knots).unwrap();
    /// assert!((interp.forward_rate(0.5) - 0.04).abs() < 1e-15);
    /// assert!((interp.forward_rate(1.5) - 0.06).abs() < 1e-15);
    /// // Flat extrapolation.
    /// assert!((interp.forward_rate(-1.0) - 0.04).abs() < 1e-15);
    /// assert!((interp.forward_rate(3.0) - 0.06).abs() < 1e-15);
    /// ```
    #[must_use]
    pub fn forward_rate(&self, t: f64) -> f64 {
        let n = self.times.len();
        // Flat in the forward rate outside the knot range.
        if t <= self.times[0] {
            return self.forwards[0];
        }
        if t > self.times[n - 1] {
            return self.forwards[n - 2];
        }
        // Interior: right-continuous at knots — for t in (t_i, t_{i+1}] we
        // return forwards[i]. `locate` returns the segment index `i` such
        // that `times[i] <= t < times[i + 1]`; the right-continuous
        // convention means we want `i - 1` when `t == times[i]` for an
        // interior knot. Equivalently: pick the smallest `i` with
        // `times[i + 1] >= t`.
        let i = self.locate_right_continuous(t);
        self.forwards[i]
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

    /// Same as `locate` but returns the **right-segment** at an interior
    /// knot: for `t == times[i]` with `0 < i < n - 1`, returns `i` (the
    /// segment to the right), not `i - 1`. This is the right-continuous
    /// convention used by [`PiecewiseConstantForward::forward_rate`].
    ///
    /// Endpoints behave as in `locate`: returns `0` for `t <= times[0]` and
    /// `n - 2` for `t >= times[n - 1]`.
    #[inline]
    fn locate_right_continuous(&self, t: f64) -> usize {
        // Caller guarantees t > times[0] when reaching the interior branch
        // in `forward_rate`; this method is correct in any case but its
        // semantics matter only on the open interval (times[0], times[n-1]].
        let i = self.locate(t);
        // If we landed exactly on an interior knot, jump to the right
        // segment. `locate` returns `i` for `t == times[i]`; we want `i`
        // unchanged when `times[i] < t`, and `i - 1 + 1 = i` is already the
        // right segment at the knot — so actually `locate` already returns
        // the right segment when `t` strictly exceeds `times[i]`. The
        // adjustment is only needed when `t == times[i]` and `i > 0`:
        // `locate` returns `i` in that case, which means segment
        // `(times[i], times[i + 1])` — already the right segment. No
        // adjustment is required.
        i
    }
}

impl Interpolator for PiecewiseConstantForward {
    fn build(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
        Self::new(knots)
    }

    fn eval(&self, t: f64) -> f64 {
        let n = self.times.len();
        // Flat-forward extrapolation outside the knot range — extend the
        // boundary segments' exponential decay outwards.
        if t <= self.times[0] {
            let f0 = self.forwards[0];
            return self.discounts[0] * (-f0 * (t - self.times[0])).exp();
        }
        if t >= self.times[n - 1] {
            let f_last = self.forwards[n - 2];
            return self.discounts[n - 1] * (-f_last * (t - self.times[n - 1])).exp();
        }
        let i = self.locate(t);
        let t_lo = self.times[i];
        let f_i = self.forwards[i];
        self.discounts[i] * (-f_i * (t - t_lo)).exp()
    }

    fn deriv(&self, t: f64) -> Option<f64> {
        // d/dt D(t) = -f(t) * D(t). The function is C^0 but not C^1 at
        // interior knots; by the right-continuous convention used by
        // `forward_rate`, the value at a knot is the right-segment slope.
        let f = self.forward_rate(t);
        let d = self.eval(t);
        Some(-f * d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Construction & validation ───────────────────────────────────────

    #[test]
    fn rejects_empty() {
        let err = PiecewiseConstantForward::new(&[]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 0 }));
    }

    #[test]
    fn rejects_single_knot() {
        let err = PiecewiseConstantForward::new(&[(0.0, 1.0)]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 1 }));
    }

    #[test]
    fn rejects_non_monotone_times() {
        let err =
            PiecewiseConstantForward::new(&[(0.0, 1.0), (2.0, 0.9), (1.0, 0.95)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NodesNotIncreasing { at_index: 2 }
        ));
    }

    #[test]
    fn rejects_duplicate_times() {
        let err =
            PiecewiseConstantForward::new(&[(0.0, 1.0), (1.0, 0.95), (1.0, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::DuplicateNode { .. }));
    }

    #[test]
    fn rejects_negative_discount() {
        let err = PiecewiseConstantForward::new(&[(0.0, 1.0), (1.0, -0.5)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_zero_discount() {
        let err = PiecewiseConstantForward::new(&[(0.0, 1.0), (1.0, 0.0)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_nan_discount() {
        let err = PiecewiseConstantForward::new(&[(0.0, 1.0), (1.0, f64::NAN)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_nan_time() {
        let err = PiecewiseConstantForward::new(&[(0.0, 1.0), (f64::NAN, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn rejects_inf_time() {
        let err = PiecewiseConstantForward::new(&[(0.0, 1.0), (f64::INFINITY, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    // ─── Evaluation correctness ──────────────────────────────────────────

    fn flat_forward_knots() -> [(f64, f64); 3] {
        // f0 = 0.04 on (0,1], f1 = 0.06 on (1,2].
        [
            (0.0, 1.0),
            (1.0, (-0.04_f64).exp()),
            (2.0, (-0.10_f64).exp()),
        ]
    }

    #[test]
    fn knot_reproduction_exact() {
        let knots = flat_forward_knots();
        let interp = PiecewiseConstantForward::new(&knots).unwrap();
        for &(t, d) in &knots {
            let v = interp.eval(t);
            assert!((v - d).abs() < 1e-15, "knot ({t}, {d}) -> {v}");
        }
    }

    #[test]
    fn forward_rate_segment_values() {
        // f0 = 0.04, f1 = 0.06.
        let interp = PiecewiseConstantForward::new(&flat_forward_knots()).unwrap();
        assert!((interp.forward_rate(0.5) - 0.04).abs() < 1e-15);
        assert!((interp.forward_rate(1.5) - 0.06).abs() < 1e-15);
    }

    #[test]
    fn forward_rate_flat_extrapolation() {
        let interp = PiecewiseConstantForward::new(&flat_forward_knots()).unwrap();
        // Left of t0: reuse f0.
        assert!((interp.forward_rate(-1.0) - 0.04).abs() < 1e-15);
        assert!((interp.forward_rate(f64::NEG_INFINITY) - 0.04).abs() < 1e-15);
        // Right of t_{n-1}: reuse f_{n-2}.
        assert!((interp.forward_rate(3.0) - 0.06).abs() < 1e-15);
        assert!((interp.forward_rate(f64::INFINITY) - 0.06).abs() < 1e-15);
    }

    #[test]
    fn eval_midsegment_matches_exponential_formula() {
        // D(1.5) = exp(-0.04) * exp(-0.06 * 0.5) = exp(-0.07).
        let interp = PiecewiseConstantForward::new(&flat_forward_knots()).unwrap();
        let v = interp.eval(1.5);
        let expected = (-0.07_f64).exp();
        assert!(
            (v - expected).abs() < 1e-15,
            "v = {v}, expected = {expected}"
        );
    }

    #[test]
    fn eval_first_segment_matches_exponential_formula() {
        // D(0.25) = 1 * exp(-0.04 * 0.25) = exp(-0.01).
        let interp = PiecewiseConstantForward::new(&flat_forward_knots()).unwrap();
        let v = interp.eval(0.25);
        let expected = (-0.01_f64).exp();
        assert!((v - expected).abs() < 1e-15);
    }

    #[test]
    fn eval_extrapolation_extends_boundary_segment() {
        let interp = PiecewiseConstantForward::new(&flat_forward_knots()).unwrap();
        // Left of t0: D(-0.5) = D(0) * exp(-f0 * (-0.5)) = exp(0.02).
        let v_left = interp.eval(-0.5);
        let expected_left = (0.02_f64).exp();
        assert!((v_left - expected_left).abs() < 1e-15);
        // Right of t_{n-1}: D(2.5) = D(2) * exp(-f1 * 0.5) = exp(-0.13).
        let v_right = interp.eval(2.5);
        let expected_right = (-0.13_f64).exp();
        assert!((v_right - expected_right).abs() < 1e-15);
    }

    #[test]
    fn monotone_decreasing_input_produces_monotone_output() {
        // Discount factors monotonically decreasing in t -> output monotone.
        let knots = [
            (0.0, 1.0),
            (0.25, 0.9875),
            (0.5, 0.9752),
            (1.0, 0.9512),
            (2.0, 0.9048),
            (5.0, 0.7788),
        ];
        let interp = PiecewiseConstantForward::new(&knots).unwrap();
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

    // ─── Derivative correctness ──────────────────────────────────────────

    #[test]
    fn deriv_equals_minus_f_times_d() {
        let interp = PiecewiseConstantForward::new(&flat_forward_knots()).unwrap();
        // Mid-first-segment.
        let t = 0.5_f64;
        let d = interp.eval(t);
        let f = interp.forward_rate(t);
        let dy = interp.deriv(t).unwrap();
        assert!((dy - (-f * d)).abs() < 1e-15);
        // Mid-second-segment.
        let t = 1.5_f64;
        let d = interp.eval(t);
        let f = interp.forward_rate(t);
        let dy = interp.deriv(t).unwrap();
        assert!((dy - (-f * d)).abs() < 1e-15);
    }

    #[test]
    fn deriv_finite_difference_interior() {
        let interp = PiecewiseConstantForward::new(&flat_forward_knots()).unwrap();
        let t = 0.5_f64;
        let dy = interp.deriv(t).unwrap();
        let h = 1e-6_f64;
        let fd = (interp.eval(t + h) - interp.eval(t - h)) / (2.0 * h);
        assert!((dy - fd).abs() < 1e-8, "analytic={dy}, fd={fd}");
    }

    #[test]
    fn deriv_in_extrapolation_region_is_flat_forward() {
        // Outside the knot range the forward stays flat -> deriv = -f * D.
        let interp = PiecewiseConstantForward::new(&flat_forward_knots()).unwrap();
        let t = -0.5_f64;
        let d = interp.eval(t);
        let f0 = 0.04_f64;
        let expected = -f0 * d;
        let dy = interp.deriv(t).unwrap();
        assert!((dy - expected).abs() < 1e-15);
        let t = 2.5_f64;
        let d = interp.eval(t);
        let f_last = 0.06_f64;
        let expected = -f_last * d;
        let dy = interp.deriv(t).unwrap();
        assert!((dy - expected).abs() < 1e-15);
    }

    // ─── Consistency with LogLinear ──────────────────────────────────────

    #[test]
    fn eval_matches_log_linear_on_same_knots_interior() {
        use super::super::LogLinear;
        let knots = [(0.0, 1.0), (0.5, 0.97), (1.0, 0.95), (2.0, 0.90)];
        let pcf = PiecewiseConstantForward::new(&knots).unwrap();
        let ll = LogLinear::new(&knots).unwrap();
        let mut t = 0.0_f64;
        while t <= 2.0 {
            let a = pcf.eval(t);
            let b = ll.eval(t);
            assert!((a - b).abs() < 1e-14, "t={t}: pcf={a}, ll={b}");
            t += 0.01;
        }
    }

    // ─── Trait & accessor ────────────────────────────────────────────────

    #[test]
    fn build_trait_method_equivalent_to_new() {
        let knots = flat_forward_knots();
        let a = PiecewiseConstantForward::new(&knots).unwrap();
        let b = <PiecewiseConstantForward as Interpolator>::build(&knots).unwrap();
        assert!((a.eval(1.5) - b.eval(1.5)).abs() < 1e-15);
    }

    #[test]
    fn len_and_is_empty() {
        let interp = PiecewiseConstantForward::new(&flat_forward_knots()).unwrap();
        assert_eq!(interp.len(), 3);
        assert!(!interp.is_empty());
    }

    #[test]
    fn clone_yields_equivalent_interpolant() {
        let interp = PiecewiseConstantForward::new(&flat_forward_knots()).unwrap();
        let copy = interp.clone();
        assert!((interp.eval(1.5) - copy.eval(1.5)).abs() < 1e-15);
        assert!((interp.forward_rate(0.5) - copy.forward_rate(0.5)).abs() < 1e-15);
    }
}
