// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Bessel-slope Hermite cubic interpolation.
//!
//! `HermiteBessel` is a **local C¹ cubic-Hermite interpolant** whose knot
//! slopes `m_i` are picked by the **Bessel formula** — the slope at `t_i` of
//! the parabola that passes through `(t_{i-1}, y_{i-1})`, `(t_i, y_i)`,
//! `(t_{i+1}, y_{i+1})`. With non-uniform spacing
//! `h_{i-1} = t_i - t_{i-1}`, `h_i = t_{i+1} - t_i`, and secant slopes
//! `S_{i-1} = (y_i - y_{i-1}) / h_{i-1}`, `S_i = (y_{i+1} - y_i) / h_i`,
//! the interior slope reads
//!
//! ```text
//! m_i = ( h_i * S_{i-1} + h_{i-1} * S_i ) / ( h_{i-1} + h_i ).
//! ```
//!
//! The two boundary slopes use the standard **three-point endpoint** rule
//! (the slope at the end of the same parabola, evaluated one step away):
//!
//! ```text
//! m_0     = ((2 h_0 + h_1) S_0     - h_0     S_1)     / (h_0 + h_1)
//! m_{n-1} = ((2 h_{n-2} + h_{n-3}) S_{n-2} - h_{n-2} S_{n-3}) / (h_{n-3} + h_{n-2})
//! ```
//!
//! For `n = 2` only one secant exists and both endpoint slopes degenerate to
//! `S_0`. For `n = 3` the interior slope `m_1` uses the Bessel formula and the
//! endpoint slopes use the three-point rule (which is exact here because there
//! is exactly one parabola through the three knots).
//!
//! Once the slopes are in hand, segment `i` is evaluated with the cubic
//! Hermite basis `H_{00}, H_{10}, H_{01}, H_{11}` on the local coordinate
//! `u = (t - t_i) / h_i`:
//!
//! ```text
//! H_{00}(u) =  2 u^3 - 3 u^2 + 1
//! H_{10}(u) =      u^3 - 2 u^2 + u
//! H_{01}(u) = -2 u^3 + 3 u^2
//! H_{11}(u) =      u^3 -     u^2
//!
//! y(t) = H_{00}(u) y_i + h_i H_{10}(u) m_i
//!      + H_{01}(u) y_{i+1} + h_i H_{11}(u) m_{i+1}.
//! ```
//!
//! The interpolant is C¹ at every interior knot by construction (both
//! adjacent segments share the same slope `m_i` at `t = t_i`) and reproduces
//! every quadratic exactly on the interior — `m_i` is, by definition, the
//! exact derivative of the local parabola.
//!
//! # Monotonicity
//!
//! Bessel-Hermite is **not** monotonicity-preserving in general: a strictly
//! monotone knot sequence can produce an interpolant with interior extrema.
//! It is monotone only when the data is already "monotone in slope" (the
//! Bessel slopes share the sign of every adjacent secant). For
//! monotonicity-preserving methods see [`MonotoneCubic`](super::MonotoneCubic)
//! (Fritsch & Carlson 1980), [`MonotoneSteffen`](super::MonotoneSteffen)
//! (Steffen 1990), and [`MonotoneHyman`](super::MonotoneHyman) (Hyman 1983).
//!
//! # Invariants
//!
//! - At least two knots.
//! - Knot times strictly increasing.
//! - Knot times and values both finite (no `NaN`, no `±∞`).
//!
//! # Extrapolation
//!
//! Flat extrapolation in the **value** domain outside the knot range —
//! `eval(t) = y_0` for `t <= t_0` and `eval(t) = y_{n-1}` for
//! `t >= t_{n-1}`. The slope information `m_0` / `m_{n-1}` is used only
//! inside the knot range; the derivative in the extrapolation region is `0`.
//!
//! # References
//!
//! - de Boor, C., *A Practical Guide to Splines*, Revised Edition,
//!   Springer-Verlag, Applied Mathematical Sciences vol. 27 (2001),
//!   Chapter IV — local cubic interpolation, including the Bessel scheme as
//!   the canonical local C¹ choice with three-point endpoint rule.
//! - Press, W. H., Teukolsky, S. A., Vetterling, W. T. & Flannery, B. P.,
//!   *Numerical Recipes*, 3rd Edition, Cambridge University Press (2007),
//!   §3.4 — cubic Hermite interpolation basis.

use crate::errors::CurveError;

use super::Interpolator;

/// Bessel-slope cubic-Hermite interpolant over a set of `(t, y)` knots.
///
/// Picks knot slopes `m_i` by the **Bessel formula** — the slope at `t_i`
/// of the parabola through the three nearest knots — and evaluates with the
/// standard cubic Hermite basis on each segment. Yields a **local, C¹**
/// interpolant that reproduces quadratics exactly at interior knots.
///
/// Bessel-Hermite is **not** monotonicity-preserving in general; for monotone
/// interpolants see [`MonotoneCubic`](super::MonotoneCubic),
/// [`MonotoneSteffen`](super::MonotoneSteffen), and
/// [`MonotoneHyman`](super::MonotoneHyman).
///
/// Flat-extrapolates outside the knot range (eval returns `y_0` below the
/// first knot and `y_{n-1}` above the last).
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::{HermiteBessel, Interpolator};
///
/// let interp = HermiteBessel::new(&[(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (3.0, 9.0)]).unwrap();
/// // Knot reproduction.
/// assert!((interp.eval(0.0) - 0.0).abs() < 1e-15);
/// assert!((interp.eval(2.0) - 4.0).abs() < 1e-15);
/// // Bessel reproduces the parabola y = t^2 exactly at an interior segment
/// // midpoint (the Bessel slope is the derivative of that very parabola).
/// assert!((interp.eval(1.5) - 1.5_f64.powi(2)).abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct HermiteBessel {
    /// Knot times, strictly increasing.
    times: Vec<f64>,
    /// Knot values, one per knot time.
    values: Vec<f64>,
    /// Bessel slopes `m_i` at each knot; same length as `times` / `values`.
    slopes: Vec<f64>,
}

impl HermiteBessel {
    /// Builds a Bessel-Hermite interpolant from a slice of `(t, y)` knots.
    ///
    /// Validation:
    ///
    /// - `knots.len() >= 2`.
    /// - `knots[i].0 < knots[i + 1].0` (strictly increasing times).
    /// - Every `t` is finite.
    /// - Every `y` is finite (no positivity requirement).
    ///
    /// Non-finite `y` values (`NaN`, `±∞`) are rejected via
    /// [`CurveError::NonPositiveDiscount`] — the variant name leans on the
    /// crate's discount-factor heritage but is reused for the broader
    /// "invalid value at node" case (matching `Linear`).
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
    /// use regit_curves::interpolation::HermiteBessel;
    /// use regit_curves::CurveError;
    ///
    /// assert!(HermiteBessel::new(&[(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]).is_ok());
    /// assert!(matches!(
    ///     HermiteBessel::new(&[(0.0, 1.0)]).unwrap_err(),
    ///     CurveError::TooFewNodes { found: 1 },
    /// ));
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
        let slopes = bessel_slopes(&times, &values);
        Ok(Self {
            times,
            values,
            slopes,
        })
    }

    /// Returns the number of knots.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns `true` if the interpolant has no knots. Always `false` for a
    /// successfully constructed `HermiteBessel` (which requires `>= 2`
    /// knots); retained for `clippy::len_without_is_empty`.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Returns the Bessel slope `m_i` at knot `i`, or `None` if `i` is out
    /// of range. Exposed primarily for testing and for callers that need
    /// the derivative information directly at the knots.
    #[must_use]
    #[inline]
    pub fn slope_at_knot(&self, i: usize) -> Option<f64> {
        self.slopes.get(i).copied()
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

/// Computes the Bessel-formula slopes `m_i` for every knot.
///
/// Invariants on the input: `times.len() == values.len() >= 2` and `times`
/// strictly increasing. The caller (`HermiteBessel::new`) is responsible for
/// enforcing both.
fn bessel_slopes(times: &[f64], values: &[f64]) -> Vec<f64> {
    let n = times.len();
    let mut slopes = Vec::with_capacity(n);
    if n == 2 {
        // Only one secant exists; both endpoint slopes degenerate to it.
        let s0 = (values[1] - values[0]) / (times[1] - times[0]);
        slopes.push(s0);
        slopes.push(s0);
        return slopes;
    }
    // n >= 3 — interior slopes use the Bessel formula; endpoint slopes use
    // the three-point rule.
    //
    // `h[i]` = times[i+1] - times[i], `s[i]` = (values[i+1] - values[i]) / h[i].
    let mut h = Vec::with_capacity(n - 1);
    let mut s = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let hi = times[i + 1] - times[i];
        h.push(hi);
        s.push((values[i + 1] - values[i]) / hi);
    }
    // Left endpoint: m_0 = ((2 h_0 + h_1) S_0 - h_0 S_1) / (h_0 + h_1).
    let m0 = ((2.0 * h[0] + h[1]) * s[0] - h[0] * s[1]) / (h[0] + h[1]);
    slopes.push(m0);
    // Interior knots: m_i = (h_i S_{i-1} + h_{i-1} S_i) / (h_{i-1} + h_i).
    for i in 1..n - 1 {
        let m = (h[i] * s[i - 1] + h[i - 1] * s[i]) / (h[i - 1] + h[i]);
        slopes.push(m);
    }
    // Right endpoint: m_{n-1} = ((2 h_{n-2} + h_{n-3}) S_{n-2}
    //                            - h_{n-2} S_{n-3}) / (h_{n-3} + h_{n-2}).
    let hn2 = h[n - 2];
    let hn3 = h[n - 3];
    let sn2 = s[n - 2];
    let sn3 = s[n - 3];
    let m_last = ((2.0 * hn2 + hn3) * sn2 - hn2 * sn3) / (hn3 + hn2);
    slopes.push(m_last);
    slopes
}

impl Interpolator for HermiteBessel {
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
        let dt = t_hi - t_lo;
        let u = (t - t_lo) / dt;
        let u2 = u * u;
        let u3 = u2 * u;
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;
        h00 * self.values[i]
            + dt * h10 * self.slopes[i]
            + h01 * self.values[i + 1]
            + dt * h11 * self.slopes[i + 1]
    }

    fn deriv(&self, t: f64) -> Option<f64> {
        let n = self.times.len();
        // Flat extrapolation -> zero derivative outside the knot range.
        if t < self.times[0] || t > self.times[n - 1] {
            return Some(0.0);
        }
        // Inside the knot range: differentiate the cubic Hermite basis on
        // the right-segment. At an interior knot this returns the right-
        // slope, but Bessel-Hermite is C¹ so left and right agree.
        let i = self.locate(t);
        let t_lo = self.times[i];
        let t_hi = self.times[i + 1];
        let dt = t_hi - t_lo;
        let u = (t - t_lo) / dt;
        let u2 = u * u;
        // d/du of the basis times du/dt = 1/dt:
        //   y'(t) = (6 u^2 - 6 u) (y_i - y_{i+1}) / dt
        //         + (3 u^2 - 4 u + 1) m_i
        //         + (3 u^2 - 2 u)     m_{i+1}.
        let term_endpoints = (6.0 * u2 - 6.0 * u) * (self.values[i] - self.values[i + 1]) / dt;
        let term_left = (3.0 * u2 - 4.0 * u + 1.0) * self.slopes[i];
        let term_right = (3.0 * u2 - 2.0 * u) * self.slopes[i + 1];
        Some(term_endpoints + term_left + term_right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Construction & validation ───────────────────────────────────────

    #[test]
    fn rejects_empty() {
        let err = HermiteBessel::new(&[]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 0 }));
    }

    #[test]
    fn rejects_single_knot() {
        let err = HermiteBessel::new(&[(0.0, 1.0)]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 1 }));
    }

    #[test]
    fn rejects_non_monotone_times() {
        let err = HermiteBessel::new(&[(0.0, 1.0), (2.0, 0.9), (1.0, 0.95)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NodesNotIncreasing { at_index: 2 }
        ));
    }

    #[test]
    fn rejects_duplicate_times() {
        let err = HermiteBessel::new(&[(0.0, 1.0), (1.0, 0.95), (1.0, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::DuplicateNode { .. }));
    }

    #[test]
    fn rejects_nan_time() {
        let err = HermiteBessel::new(&[(0.0, 1.0), (f64::NAN, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn rejects_inf_time() {
        let err = HermiteBessel::new(&[(0.0, 1.0), (f64::INFINITY, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn rejects_nan_value() {
        let err = HermiteBessel::new(&[(0.0, 1.0), (1.0, f64::NAN)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_inf_value() {
        let err = HermiteBessel::new(&[(0.0, 1.0), (1.0, f64::INFINITY)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn accepts_negative_value() {
        // No positivity constraint — negative values are valid.
        let interp = HermiteBessel::new(&[(0.0, 1.0), (1.0, -0.5), (2.0, 0.25)]).unwrap();
        assert!((interp.eval(0.0) - 1.0).abs() < 1e-14);
        assert!((interp.eval(1.0) - (-0.5)).abs() < 1e-14);
    }

    // ─── Evaluation correctness ──────────────────────────────────────────

    #[test]
    fn knot_reproduction_exact() {
        let knots = [
            (0.0, 1.0),
            (0.5, 0.97),
            (1.0, 0.95),
            (2.5, 0.90),
            (5.0, 0.80),
        ];
        let interp = HermiteBessel::new(&knots).unwrap();
        for &(t, y) in &knots {
            let v = interp.eval(t);
            assert!((v - y).abs() < 1e-14, "knot ({t}, {y}) -> {v}");
        }
    }

    #[test]
    fn linear_function_reproduced_exactly() {
        // y = 2 + 3 t — linear is in the polynomial family the Bessel scheme
        // is exact on (every Bessel slope collapses to 3, and a cubic Hermite
        // with slope 3 at each end of a segment reproduces the line).
        let knots: Vec<(f64, f64)> = [0.0, 0.7, 1.3, 2.5, 4.1, 5.0]
            .iter()
            .map(|&t| (t, 2.0 + 3.0 * t))
            .collect();
        let interp = HermiteBessel::new(&knots).unwrap();
        let mut t = 0.0_f64;
        while t <= 5.0 {
            let v = interp.eval(t);
            let expected = 2.0 + 3.0 * t;
            assert!(
                (v - expected).abs() < 1e-13,
                "t={t}, v={v}, expected={expected}"
            );
            t += 0.05;
        }
    }

    #[test]
    fn parabola_reproduced_at_interior_midpoints() {
        // y = t^2: the Bessel slope at an interior knot is the exact
        // derivative `2 t` of the local parabola. Inside an interior
        // segment bordered by two Bessel-slope knots, the cubic Hermite
        // therefore reproduces the parabola exactly.
        let knots: Vec<(f64, f64)> = [0.0, 1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|&t| (t, t * t))
            .collect();
        let interp = HermiteBessel::new(&knots).unwrap();
        // Test midpoints of the two interior segments: [1,2] and [2,3].
        // Both endpoints are interior knots so their Bessel slopes equal 2t.
        for &mid in &[1.5_f64, 2.5_f64] {
            let v = interp.eval(mid);
            let expected = mid * mid;
            assert!(
                (v - expected).abs() < 1e-13,
                "t={mid}, v={v}, expected={expected}"
            );
        }
    }

    #[test]
    fn cubic_polynomial_well_approximated() {
        // y = t^3 — Bessel is **not** exact on cubics; the deviation comes
        // from the second-order accuracy of the Bessel slope formula on a
        // genuine cubic (the local parabola whose derivative the slope
        // tracks differs from the cubic's tangent). Across 5 uniform knots
        // on `[0, 2]` the measured peak error on the **interior** segments
        // `[0.5, 1.5]` (away from the boundary segments, where the
        // three-point endpoint rule degrades accuracy further) is
        // approximately `1.2e-2`. The interpolant remains O(h²) accurate
        // on a smooth cubic — halve the step and the error drops by ~4x.
        let knots: Vec<(f64, f64)> = [0.0, 0.5, 1.0, 1.5, 2.0]
            .iter()
            .map(|&t| (t, t * t * t))
            .collect();
        let interp = HermiteBessel::new(&knots).unwrap();
        let mut max_err = 0.0_f64;
        // Restrict to the two interior segments [0.5, 1.5].
        let mut t = 0.5_f64;
        while t <= 1.5 {
            let v = interp.eval(t);
            let expected = t * t * t;
            let err = (v - expected).abs();
            if err > max_err {
                max_err = err;
            }
            t += 0.001;
        }
        assert!(max_err < 1.5e-2, "peak |y - t^3| on interior = {max_err}");
    }

    #[test]
    fn c1_continuity_at_interior_knots() {
        // Bessel-Hermite is C¹ by construction: at every interior knot the
        // derivative of segment `i - 1` evaluated at `u = 1` and the
        // derivative of segment `i` evaluated at `u = 0` both collapse to
        // the shared Bessel slope `m_i`. We verify this **exactly** by
        // checking that `deriv` taken from the right of the knot (which
        // `locate` resolves to segment `i`, `u = 0`) and the value obtained
        // by manually evaluating segment `i - 1`'s derivative at `u = 1`
        // both equal `m_i` to machine precision.
        let knots = [(0.0, 0.0), (0.5, 0.25), (1.4, 1.0), (2.6, 0.5), (4.1, -0.3)];
        let interp = HermiteBessel::new(&knots).unwrap();
        for i in 1..knots.len() - 1 {
            let (t, _) = knots[i];
            let m_i = interp.slope_at_knot(i).expect("interior knot");
            // Right segment, u = 0 → `deriv` at exactly `t`.
            let right = interp.deriv(t).expect("deriv defined on interior");
            // Left segment, u = 1: hand-evaluate the Hermite-basis
            // derivative formula. (Stepping `eval` slightly left to force
            // `locate` into the left segment would re-introduce truncation
            // error; this construction is exact.)
            let (t_lo, y_lo) = knots[i - 1];
            let (t_hi, y_hi) = knots[i];
            let h = t_hi - t_lo;
            // u = 1: (6 u² - 6 u) = 0; (3 u² - 4 u + 1) = 0;
            // (3 u² - 2 u) = 1. So derivative at u = 1 in segment i-1 is
            // m_i. We assemble the explicit calculation to also confirm
            // the formula matches our `deriv` code.
            let u = 1.0_f64;
            let u2 = u * u;
            let m_prev = interp.slope_at_knot(i - 1).expect("left knot");
            let left = (6.0 * u2 - 6.0 * u) * (y_lo - y_hi) / h
                + (3.0 * u2 - 4.0 * u + 1.0) * m_prev
                + (3.0 * u2 - 2.0 * u) * m_i;
            assert!(
                (left - m_i).abs() < 1e-14,
                "left u=1 derivative at t={t}: got {left}, m_i={m_i}",
            );
            assert!(
                (right - m_i).abs() < 1e-14,
                "right u=0 derivative at t={t}: got {right}, m_i={m_i}",
            );
            assert!(
                (left - right).abs() < 1e-14,
                "C^1 violated at t={t}: left={left}, right={right}",
            );
        }
    }

    #[test]
    fn deriv_consistent_with_finite_difference() {
        let knots = [
            (0.0, 1.0),
            (1.0, 0.95),
            (2.0, 0.85),
            (4.0, 0.70),
            (7.0, 0.55),
        ];
        let interp = HermiteBessel::new(&knots).unwrap();
        // Sample at midpoints of every interior segment.
        for i in 0..knots.len() - 1 {
            let mid = 0.5 * (knots[i].0 + knots[i + 1].0);
            let analytic = interp.deriv(mid).expect("deriv defined on interior");
            let h = 1e-6_f64;
            let fd = (interp.eval(mid + h) - interp.eval(mid - h)) / (2.0 * h);
            assert!(
                (analytic - fd).abs() < 1e-7,
                "segment {i}, mid={mid}: analytic={analytic}, fd={fd}",
            );
        }
    }

    #[test]
    fn flat_extrapolation_in_value() {
        let interp = HermiteBessel::new(&[(0.0, 1.0), (1.0, 0.95), (2.0, 0.85)]).unwrap();
        assert!((interp.eval(-1.0) - 1.0).abs() < 1e-14);
        assert!((interp.eval(-100.0) - 1.0).abs() < 1e-14);
        assert!((interp.eval(2.0) - 0.85).abs() < 1e-14);
        assert!((interp.eval(100.0) - 0.85).abs() < 1e-14);
        assert!((interp.eval(f64::INFINITY) - 0.85).abs() < 1e-14);
        assert!((interp.eval(f64::NEG_INFINITY) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn deriv_zero_in_extrapolation_region() {
        let interp = HermiteBessel::new(&[(0.0, 1.0), (1.0, 0.95), (2.0, 0.85)]).unwrap();
        let d_left = interp.deriv(-1.0).unwrap();
        let d_right = interp.deriv(3.0).unwrap();
        assert!((d_left - 0.0).abs() < 1e-15);
        assert!((d_right - 0.0).abs() < 1e-15);
    }

    #[test]
    fn two_knot_case_is_linear() {
        // n = 2: both endpoint slopes are the single secant. The cubic
        // Hermite with matching slopes at both ends of one segment is the
        // straight line through the two knots.
        let interp = HermiteBessel::new(&[(0.0, 1.0), (2.0, 5.0)]).unwrap();
        // Slope = (5 - 1) / 2 = 2; both stored slopes equal 2.
        assert!((interp.slope_at_knot(0).unwrap() - 2.0).abs() < 1e-15);
        assert!((interp.slope_at_knot(1).unwrap() - 2.0).abs() < 1e-15);
        // Linear reproduction.
        for &t in &[0.5_f64, 1.0, 1.5] {
            let v = interp.eval(t);
            let expected = 1.0 + 2.0 * t;
            assert!((v - expected).abs() < 1e-14, "t={t}, v={v}");
        }
    }

    #[test]
    fn three_knot_case_reproduces_parabola_exactly() {
        // n = 3 with y = t^2: interior Bessel slope at t_1 equals 2 t_1
        // exactly (the parabola through three of its own points has the
        // right derivative), and the three-point endpoint rule recovers
        // 2 t_0 and 2 t_2. With all three slopes correct, the Hermite cubic
        // reproduces the parabola exactly everywhere.
        let knots = [(-1.0, 1.0), (0.5, 0.25), (2.0, 4.0)];
        let interp = HermiteBessel::new(&knots).unwrap();
        // Slope at each knot equals 2 t_i.
        for (i, &(t, _)) in knots.iter().enumerate() {
            let m = interp.slope_at_knot(i).unwrap();
            assert!(
                (m - 2.0 * t).abs() < 1e-13,
                "knot {i}: slope={m}, expected={}",
                2.0 * t
            );
        }
        // Value reproduction at arbitrary interior points.
        for &t in &[-0.5_f64, 0.0, 0.8, 1.3, 1.9] {
            let v = interp.eval(t);
            assert!((v - t * t).abs() < 1e-13, "t={t}, v={v}");
        }
    }

    #[test]
    fn slope_at_knot_out_of_range_is_none() {
        let interp = HermiteBessel::new(&[(0.0, 1.0), (1.0, 0.95)]).unwrap();
        assert!(interp.slope_at_knot(0).is_some());
        assert!(interp.slope_at_knot(1).is_some());
        assert!(interp.slope_at_knot(2).is_none());
        assert!(interp.slope_at_knot(99).is_none());
    }

    // ─── Trait & accessor ────────────────────────────────────────────────

    #[test]
    fn build_trait_method_equivalent_to_new() {
        let knots = [(0.0, 1.0), (1.0, 0.95), (2.0, 0.85)];
        let a = HermiteBessel::new(&knots).unwrap();
        let b = <HermiteBessel as Interpolator>::build(&knots).unwrap();
        assert!((a.eval(0.5) - b.eval(0.5)).abs() < 1e-15);
        assert!((a.eval(1.7) - b.eval(1.7)).abs() < 1e-15);
    }

    #[test]
    fn len_and_is_empty() {
        let interp = HermiteBessel::new(&[(0.0, 1.0), (1.0, 0.95), (2.0, 0.9)]).unwrap();
        assert_eq!(interp.len(), 3);
        assert!(!interp.is_empty());
    }

    #[test]
    fn clone_yields_equivalent_interpolant() {
        let interp = HermiteBessel::new(&[(0.0, 1.0), (1.0, 0.95), (2.0, 0.85)]).unwrap();
        let copy = interp.clone();
        for &t in &[0.0_f64, 0.3, 0.5, 1.0, 1.4, 2.0] {
            assert!((interp.eval(t) - copy.eval(t)).abs() < 1e-15);
        }
    }
}
