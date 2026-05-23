// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Fritsch–Carlson monotone piecewise cubic interpolation.
//!
//! `MonotoneCubic` is a C¹ piecewise-cubic Hermite interpolant whose slopes
//! at each knot are chosen so that, **when the input data is monotone**, the
//! interpolant is monotone on every segment. The algorithm is due to Fritsch
//! & Carlson (1980); the slope formula used at interior knots is the
//! Fritsch–Butland (1984) weighted harmonic mean, a refinement of the
//! original Fritsch–Carlson three-point rule that produces visibly smoother
//! curves on real data. The endpoint slopes and the monotonicity filter are
//! the standard PCHIP construction (Dougherty, Edelman & Hyman 1989).
//!
//! # Algorithm
//!
//! Let the knots be `(t_0, y_0), ..., (t_{n-1}, y_{n-1})` with strictly
//! increasing `t_i`. Let `h_i = t_{i+1} - t_i` and the secant slope
//! `S_i = (y_{i+1} - y_i) / h_i`.
//!
//! 1. **Interior slopes (Fritsch–Butland 1984)** — for `1 <= i <= n-2`:
//!
//!    ```text
//!    if S_{i-1} * S_i <= 0:
//!        m_i = 0
//!    else:
//!        m_i = 3 * (h_{i-1} + h_i)
//!            / ( (2*h_i + h_{i-1}) / S_{i-1} + (h_i + 2*h_{i-1}) / S_i )
//!    ```
//!
//!    The weighted harmonic mean produces a slope with the same sign as the
//!    bracketing secants when they agree, and is `0` at a local extremum.
//!
//! 2. **Endpoint slopes (three-point formula, clamped)** — for `i = 0`:
//!
//!    ```text
//!    m_0 = ((2*h_0 + h_1) * S_0 - h_0 * S_1) / (h_0 + h_1)
//!    if sign(m_0) != sign(S_0):  m_0 = 0
//!    if |m_0| > 3 * |S_0|:       m_0 = 3 * S_0
//!    ```
//!
//!    The symmetric formula applies at `i = n-1`.
//!
//! 3. **Monotonicity filter (Fritsch–Carlson 1980, §4)** — for each segment
//!    `i` with `S_i != 0`, let `α = m_i / S_i`, `β = m_{i+1} / S_i`. If
//!    `(α, β)` is outside the disc `α² + β² <= 9`, scale both slopes back
//!    by `τ = 3 / sqrt(α² + β²)`:
//!
//!    ```text
//!    m_i     ← τ * α * S_i
//!    m_{i+1} ← τ * β * S_i
//!    ```
//!
//!    Fritsch & Carlson (1980) prove that `(α, β)` lying in the unit-circle-
//!    of-radius-3 intersected with the closed first quadrant is sufficient
//!    for the cubic Hermite to be monotone on the segment, and the projection
//!    above preserves the slope signs while moving `(α, β)` to the boundary
//!    of that region.
//!
//! 4. **Evaluation** — on segment `i` with `u = (t - t_i) / h_i`:
//!
//!    ```text
//!    y(t) = (2u³ − 3u² + 1) * y_i
//!         + (u³ − 2u² + u) * h_i * m_i
//!         + (−2u³ + 3u²)     * y_{i+1}
//!         + (u³ − u²)        * h_i * m_{i+1}
//!    ```
//!
//!    The first derivative is the analytic derivative of the same cubic.
//!
//! # Monotonicity guarantee
//!
//! The Fritsch–Carlson filter is a *segment-by-segment* monotonicity-
//! preservation step: **the interpolant is monotone on every segment iff the
//! input data is monotone**. If the input knots have a turning point, the
//! Fritsch–Butland slope formula sets `m_i = 0` at the extremum, so the
//! interpolant is locally flat there but otherwise smooth — the filter still
//! produces a well-defined C¹ interpolant; it simply makes no global
//! monotonicity claim when the data itself is non-monotone.
//!
//! # Invariants
//!
//! - At least two knots.
//! - Knot times strictly increasing.
//! - Knot times and values both finite (no `NaN`, no `±∞`).
//! - No positivity constraint on `y`.
//!
//! # Extrapolation
//!
//! Flat extrapolation outside the knot range — `eval(t) = y_0` for
//! `t <= t_0` and `eval(t) = y_{n-1}` for `t >= t_{n-1}`. The derivative in
//! the extrapolation region is therefore `0`. This matches the conservative
//! market default used elsewhere in the crate.
//!
//! # References
//!
//! - Fritsch, F. N. & Carlson, R. E., "Monotone piecewise cubic
//!   interpolation", *SIAM J. Numer. Anal.* 17(2):238-246 (1980). DOI
//!   10.1137/0717021. Necessary and sufficient conditions for monotonicity
//!   of a piecewise cubic Hermite interpolant; the slope-region filter
//!   `(α² + β² <= 9, α >= 0, β >= 0)` used in step 3.
//! - Fritsch, F. N. & Butland, J., "A method for constructing local monotone
//!   piecewise cubic interpolants", *SIAM J. Sci. Stat. Comput.* 5(2):300-304
//!   (1984). DOI 10.1137/0905021. The weighted-harmonic-mean slope formula
//!   used at interior knots in step 1 (a refinement of Fritsch–Carlson 1980
//!   §3 producing visibly smoother slopes on real data).
//! - Dougherty, R. L., Edelman, A. & Hyman, J. M., "Nonnegativity-,
//!   monotonicity-, or convexity-preserving cubic and quintic Hermite
//!   interpolation", *Math. Comp.* 52(186):471-494 (1989). DOI
//!   10.1090/S0025-5718-1989-0962209-1. Modern statement of the PCHIP
//!   endpoint-slope clamp used in step 2.

use crate::errors::CurveError;

use super::Interpolator;

/// Fritsch–Carlson monotone piecewise cubic Hermite interpolant.
///
/// Stores the knot times, knot values, and the Fritsch–Carlson-filtered
/// Hermite slope at each knot. Evaluation on each segment is the standard
/// cubic Hermite basis combination of `(y_i, y_{i+1}, m_i, m_{i+1})`; the
/// interpolant is C¹ everywhere by construction.
///
/// **Monotonicity guarantee:** the interpolant is monotone on every segment
/// iff the input knot values are monotone. With non-monotone input the
/// filter still produces a well-defined C¹ interpolant; it simply zeros the
/// slope at any turning point in the data.
///
/// Flat-extrapolates outside the knot range (eval returns `y_0` below the
/// first knot and `y_{n-1}` above the last).
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::{Interpolator, MonotoneCubic};
///
/// // A monotone-increasing knot set.
/// let interp = MonotoneCubic::new(&[
///     (0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (3.0, 9.0),
/// ]).unwrap();
/// // Knot reproduction.
/// assert!((interp.eval(0.0) - 0.0).abs() < 1e-15);
/// assert!((interp.eval(2.0) - 4.0).abs() < 1e-15);
/// // Monotone-increasing inputs produce a monotone-increasing interpolant.
/// assert!(interp.eval(0.5) <= interp.eval(1.5));
/// ```
#[derive(Debug, Clone)]
pub struct MonotoneCubic {
    /// Knot times, strictly increasing.
    times: Vec<f64>,
    /// Knot values, one per knot time.
    values: Vec<f64>,
    /// Fritsch–Carlson-filtered Hermite slopes `m_i`, one per knot.
    slopes: Vec<f64>,
}

impl MonotoneCubic {
    /// Builds a Fritsch–Carlson monotone cubic interpolant from a slice of
    /// `(t, y)` knots.
    ///
    /// Validation:
    ///
    /// - `knots.len() >= 2`.
    /// - `knots[i].0 < knots[i + 1].0` (strictly increasing times).
    /// - Every `t` is finite.
    /// - Every `y` is finite (no positivity requirement).
    ///
    /// On exactly two knots the interpolant reduces to linear interpolation
    /// on the single segment (both endpoint slopes equal the secant, which
    /// satisfies the Fritsch–Carlson region trivially).
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
    /// use regit_curves::interpolation::MonotoneCubic;
    /// use regit_curves::CurveError;
    ///
    /// assert!(MonotoneCubic::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]).is_ok());
    /// assert!(matches!(
    ///     MonotoneCubic::new(&[(0.0, 1.0)]).unwrap_err(),
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
        let slopes = compute_slopes(&times, &values);
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
    /// successfully constructed `MonotoneCubic` (which requires `>= 2`
    /// knots); retained for `clippy::len_without_is_empty`.
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

/// Computes Fritsch–Carlson-filtered Hermite slopes at each knot.
///
/// Sequence: Fritsch–Butland weighted-harmonic-mean for interior slopes,
/// three-point formula with PCHIP clamps for endpoint slopes, then the
/// Fritsch–Carlson monotonicity-region projection segment-by-segment.
fn compute_slopes(times: &[f64], values: &[f64]) -> Vec<f64> {
    let n = times.len();
    // n >= 2 by `new`'s precondition.
    if n == 2 {
        let s = (values[1] - values[0]) / (times[1] - times[0]);
        return vec![s, s];
    }

    // Step 1 — secant slopes on every segment.
    let mut secant = Vec::with_capacity(n - 1);
    let mut step = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let h = times[i + 1] - times[i];
        step.push(h);
        secant.push((values[i + 1] - values[i]) / h);
    }

    let mut m = vec![0.0_f64; n];

    // Step 2 — Fritsch–Butland weighted harmonic mean at interior knots.
    for i in 1..n - 1 {
        let s_left = secant[i - 1];
        let s_right = secant[i];
        if s_left * s_right <= 0.0 {
            m[i] = 0.0;
        } else {
            let h_left = step[i - 1];
            let h_right = step[i];
            let w_left = 2.0 * h_right + h_left;
            let w_right = h_right + 2.0 * h_left;
            m[i] = 3.0 * (h_left + h_right) / (w_left / s_left + w_right / s_right);
        }
    }

    // Step 3 — endpoint slopes via three-point formula, then PCHIP clamp.
    m[0] = endpoint_slope(step[0], step[1], secant[0], secant[1]);
    m[n - 1] = endpoint_slope(step[n - 2], step[n - 3], secant[n - 2], secant[n - 3]);

    // Step 4 — Fritsch–Carlson monotonicity-region projection on each
    // segment. We update slopes in place; each segment's projection only
    // tightens (reduces magnitude of) its two endpoint slopes, so the
    // projection of segment i never violates the projection of segment i-1.
    for i in 0..n - 1 {
        let s = secant[i];
        // A zero secant means the data is flat on this segment — Fritsch &
        // Carlson §4 require both endpoint slopes to be zero for the cubic
        // to be monotone (in fact constant) on that segment.
        if s == 0.0 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
            continue;
        }
        let alpha = m[i] / s;
        let beta = m[i + 1] / s;
        let radius_sq = alpha.mul_add(alpha, beta * beta);
        if radius_sq > 9.0 {
            let tau = 3.0 / radius_sq.sqrt();
            m[i] = tau * alpha * s;
            m[i + 1] = tau * beta * s;
        }
    }

    m
}

/// Three-point endpoint-slope formula with PCHIP sign clamp.
///
/// Given the segment step lengths `h_near` (segment touching the endpoint),
/// `h_far` (the next segment over) and the matching secant slopes `s_near`,
/// `s_far`, returns the endpoint slope. The raw three-point estimate is
/// clamped so that (a) its sign matches `s_near` and (b) its magnitude does
/// not exceed `3 * |s_near|` — both standard PCHIP boundary conditions.
fn endpoint_slope(h_near: f64, h_far: f64, s_near: f64, s_far: f64) -> f64 {
    let raw = ((2.0 * h_near + h_far) * s_near - h_near * s_far) / (h_near + h_far);
    // Sign mismatch with the adjacent secant -> zero the slope. (PCHIP's
    // "do no harm" rule at the boundary.)
    if raw * s_near <= 0.0 {
        return 0.0;
    }
    // Magnitude clamp: must lie within 3*|s_near| for the boundary segment
    // to satisfy the Fritsch-Carlson monotonicity region given the
    // (currently unknown) interior slope at the adjacent knot. The interior
    // filter handles the joint clamp, but applying this one early keeps the
    // boundary slope in the regime where the filter does not need to act.
    if raw.abs() > 3.0 * s_near.abs() {
        return 3.0 * s_near;
    }
    raw
}

impl Interpolator for MonotoneCubic {
    fn build(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
        Self::new(knots)
    }

    #[allow(clippy::many_single_char_names)] // u, u², u³ — Hermite-basis notation.
    fn eval(&self, t: f64) -> f64 {
        let last = self.times.len() - 1;
        // Flat extrapolation outside the knot range.
        if t <= self.times[0] {
            return self.values[0];
        }
        if t >= self.times[last] {
            return self.values[last];
        }
        let seg = self.locate(t);
        let step = self.times[seg + 1] - self.times[seg];
        let u = (t - self.times[seg]) / step;
        let u2 = u * u;
        let u3 = u2 * u;
        // Standard cubic Hermite basis polynomials.
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;
        h00 * self.values[seg]
            + h10 * step * self.slopes[seg]
            + h01 * self.values[seg + 1]
            + h11 * step * self.slopes[seg + 1]
    }

    #[allow(clippy::many_single_char_names)] // u, u² — Hermite-basis notation.
    fn deriv(&self, t: f64) -> Option<f64> {
        let last = self.times.len() - 1;
        // Flat extrapolation -> zero derivative outside the knot range.
        if t < self.times[0] || t > self.times[last] {
            return Some(0.0);
        }
        let seg = self.locate(t);
        let step = self.times[seg + 1] - self.times[seg];
        let u = (t - self.times[seg]) / step;
        let u2 = u * u;
        // Derivatives of the Hermite basis with respect to `u`.
        let dh00 = 6.0 * u2 - 6.0 * u;
        let dh10 = 3.0 * u2 - 4.0 * u + 1.0;
        let dh01 = -6.0 * u2 + 6.0 * u;
        let dh11 = 3.0 * u2 - 2.0 * u;
        // d/dt = (1/step) * d/du, so the `step * m_i` term loses its `step`.
        let dy_du = dh00 * self.values[seg]
            + dh10 * step * self.slopes[seg]
            + dh01 * self.values[seg + 1]
            + dh11 * step * self.slopes[seg + 1];
        Some(dy_du / step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Construction & validation ───────────────────────────────────────

    #[test]
    fn rejects_empty() {
        let err = MonotoneCubic::new(&[]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 0 }));
    }

    #[test]
    fn rejects_single_knot() {
        let err = MonotoneCubic::new(&[(0.0, 1.0)]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 1 }));
    }

    #[test]
    fn rejects_non_monotone_times() {
        let err = MonotoneCubic::new(&[(0.0, 1.0), (2.0, 0.9), (1.0, 0.95)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NodesNotIncreasing { at_index: 2 }
        ));
    }

    #[test]
    fn rejects_duplicate_times() {
        let err = MonotoneCubic::new(&[(0.0, 1.0), (1.0, 0.95), (1.0, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::DuplicateNode { .. }));
    }

    #[test]
    fn rejects_nan_value() {
        let err = MonotoneCubic::new(&[(0.0, 1.0), (1.0, f64::NAN)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_inf_value() {
        let err = MonotoneCubic::new(&[(0.0, 1.0), (1.0, f64::INFINITY)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_nan_time() {
        let err = MonotoneCubic::new(&[(0.0, 1.0), (f64::NAN, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    // ─── Knot reproduction ───────────────────────────────────────────────

    #[test]
    fn knot_reproduction_exact() {
        let knots = [(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (3.0, 9.0), (5.0, 25.0)];
        let interp = MonotoneCubic::new(&knots).unwrap();
        for &(t, y) in &knots {
            let v = interp.eval(t);
            assert!((v - y).abs() < 1e-12, "knot ({t}, {y}) -> {v}");
        }
    }

    #[test]
    fn two_knot_reduces_to_linear() {
        // With exactly two knots both slopes equal the secant; the cubic
        // Hermite combination then degenerates to the straight line.
        let interp = MonotoneCubic::new(&[(0.0, 1.0), (1.0, 3.0)]).unwrap();
        for &t in &[0.0_f64, 0.25, 0.5, 0.75, 1.0] {
            let expected = 1.0 + 2.0 * t;
            let v = interp.eval(t);
            assert!((v - expected).abs() < 1e-15, "t={t}: {v} vs {expected}");
        }
    }

    // ─── Monotonicity preservation ───────────────────────────────────────

    /// Deterministic, seedable LCG — enough randomness for a property-style
    /// test without pulling in a dev-dependency on `rand`. Numerical
    /// Recipes' "ranqd1" constants (Press et al. 2007 §7.1).
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        #[allow(clippy::cast_possible_truncation)] // Keep low 32 bits by design.
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 16) as u32
        }
        fn next_unit(&mut self) -> f64 {
            f64::from(self.next_u32()) / f64::from(u32::MAX)
        }
    }

    #[test]
    fn monotone_input_yields_monotone_output_thirty_sets() {
        let mut rng = Lcg::new(0x00C0_FFEE_u64);
        for set_idx in 0..30 {
            // Build 8-12 strictly increasing times with random non-negative
            // increments in (0, 1], and 8-12 strictly increasing values with
            // random non-negative increments in (0, 5].
            let n = 8 + (rng.next_u32() % 5) as usize;
            let mut times = Vec::with_capacity(n);
            let mut values = Vec::with_capacity(n);
            let mut t = 0.0_f64;
            let mut y = 0.0_f64;
            for _ in 0..n {
                times.push(t);
                values.push(y);
                // Add strictly positive increments to enforce both
                // strictness and monotonicity.
                t += 0.05 + rng.next_unit();
                y += 0.01 + 5.0 * rng.next_unit();
            }
            let knots: Vec<(f64, f64)> =
                times.iter().copied().zip(values.iter().copied()).collect();
            let interp = MonotoneCubic::new(&knots).unwrap();

            // Sample on a 200-point grid across the knot range.
            let t_lo = times[0];
            let t_hi = times[n - 1];
            let grid: u32 = 200;
            let mut prev = interp.eval(t_lo);
            for k in 1..=grid {
                let t = t_lo + (t_hi - t_lo) * f64::from(k) / f64::from(grid);
                let v = interp.eval(t);
                assert!(
                    v + 1e-12 >= prev,
                    "set {set_idx}: non-monotone at t={t}, prev={prev}, v={v}"
                );
                prev = v;
            }
        }
    }

    #[test]
    fn rpn15a_monotone_on_fine_grid() {
        // Hyman 1983 RPN15A — 9-point CDF-like data. Transcribed from
        // doc/RESEARCH.md §2.5. The Fritsch–Carlson interpolant must stay
        // monotone on a fine grid (and in particular satisfy f(11.0) <= 1).
        let knots = [
            (7.99, 0.0_f64),
            (8.09, 2.764_29e-5),
            (8.19, 4.374_98e-5),
            (8.70, 0.169_183),
            (9.20, 0.469_428),
            (10.00, 0.943_740),
            (12.00, 0.998_636),
            (15.00, 0.999_919),
            (20.00, 0.999_994),
        ];
        let interp = MonotoneCubic::new(&knots).unwrap();
        let mut prev = interp.eval(7.99);
        let mut t = 7.99_f64;
        let step = 0.01_f64;
        while t <= 20.0 {
            let v = interp.eval(t);
            assert!(
                v + 1e-12 >= prev,
                "non-monotone on RPN15A at t={t}: prev={prev}, v={v}"
            );
            prev = v;
            t += step;
        }
        // Monotonicity discriminator from RESEARCH.md §2.5: the filtered
        // interpolant satisfies f(11.0) <= 1.0 (where unfiltered cubic
        // splines overshoot to f(11) > 1).
        assert!(interp.eval(11.0) <= 1.0);
        // Knot reproduction stays exact on this fixture too.
        for &(t, y) in &knots {
            let v = interp.eval(t);
            assert!((v - y).abs() < 1e-12, "RPN15A knot ({t}, {y}) -> {v}");
        }
    }

    #[test]
    fn non_monotone_input_zeros_slope_at_turning_point() {
        // A single dip in an otherwise monotone series. The Fritsch–Butland
        // formula sets m_i = 0 at any turning point (where the bracketing
        // secants change sign), so the interpolant has a zero derivative
        // there.
        let knots = [
            (0.0, 0.0),
            (1.0, 2.0),
            (2.0, 1.0), // turning point
            (3.0, 4.0),
            (4.0, 8.0),
        ];
        let interp = MonotoneCubic::new(&knots).unwrap();
        // Knot reproduction even on non-monotone input.
        for &(t, y) in &knots {
            let v = interp.eval(t);
            assert!((v - y).abs() < 1e-12, "knot ({t}, {y}) -> {v}");
        }
        // Slope at the turning-point knot is exactly zero (left and right
        // derivative agree because we are C¹).
        let d = interp.deriv(2.0).unwrap();
        assert!(
            d.abs() < 1e-15,
            "expected zero slope at turning point, got {d}"
        );
    }

    // ─── Linear function reproduction ────────────────────────────────────

    #[test]
    fn reproduces_linear_function() {
        // y = 2 + 3*x at non-uniform knot spacing. A monotone cubic must
        // recover this exactly, since the linear function is a degenerate
        // cubic that already satisfies the Fritsch–Carlson region with
        // alpha = beta = 1.
        let f = |x: f64| 2.0 + 3.0 * x;
        let knots: Vec<(f64, f64)> = [0.0_f64, 0.5, 1.7, 3.1, 4.0, 6.0, 9.0]
            .iter()
            .map(|&x| (x, f(x)))
            .collect();
        let interp = MonotoneCubic::new(&knots).unwrap();
        for &t in &[0.1_f64, 0.7, 1.0, 2.5, 3.7, 5.2, 7.9] {
            let v = interp.eval(t);
            let expected = f(t);
            assert!(
                (v - expected).abs() < 1e-12,
                "t={t}: got {v}, want {expected}"
            );
        }
        // The derivative is the constant slope of the underlying line.
        for &t in &[0.3_f64, 1.8, 4.5, 7.0] {
            let d = interp.deriv(t).unwrap();
            assert!((d - 3.0).abs() < 1e-12, "t={t}: deriv {d}");
        }
    }

    // ─── C¹ continuity at interior knots ─────────────────────────────────

    #[test]
    fn c1_continuous_at_interior_knots() {
        // Compare the left- and right-side finite-difference derivative at
        // every interior knot. Since the interpolant is C¹ by construction,
        // the two must agree to numerical precision.
        let knots = [
            (0.0, 0.0_f64),
            (1.0, 1.5),
            (2.5, 3.0),
            (4.0, 7.0),
            (5.0, 12.0),
            (7.0, 13.0),
            (10.0, 14.5),
        ];
        let interp = MonotoneCubic::new(&knots).unwrap();
        // Use one-sided finite differences on each side of every interior
        // knot — that is the direct probe of C¹. A centered FD across a
        // knot would not be useful: the two cubic pieces meeting at a knot
        // share `(y, y')` but generally not `y''`, so the centered FD picks
        // up the f'' jump and is only O(h) accurate at the knot.
        let h = 1e-6_f64;
        for &(t, _) in &knots[1..knots.len() - 1] {
            let d_left = (interp.eval(t) - interp.eval(t - h)) / h;
            let d_right = (interp.eval(t + h) - interp.eval(t)) / h;
            assert!(
                (d_left - d_right).abs() < 1e-5,
                "C^1 mismatch at t={t}: left={d_left}, right={d_right}"
            );
            // The analytic derivative at the knot should agree with the
            // one-sided FD on the right segment (our `deriv` convention is
            // to return the right slope on segment boundaries; the locate
            // function maps `t = t_i` into segment `i` by construction).
            let analytic = interp.deriv(t).unwrap();
            assert!(
                (analytic - d_right).abs() < 1e-5,
                "deriv mismatch at t={t}: analytic={analytic}, fd_right={d_right}"
            );
        }
    }

    // ─── Extrapolation ───────────────────────────────────────────────────

    #[test]
    fn flat_extrapolation_both_sides() {
        let interp = MonotoneCubic::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)]).unwrap();
        assert!((interp.eval(-100.0) - 1.0).abs() < 1e-15);
        assert!((interp.eval(100.0) - 5.0).abs() < 1e-15);
        // And zero derivative in the extrapolation region.
        assert!((interp.deriv(-1.0).unwrap() - 0.0).abs() < 1e-15);
        assert!((interp.deriv(3.0).unwrap() - 0.0).abs() < 1e-15);
    }

    // ─── Trait & accessors ───────────────────────────────────────────────

    #[test]
    fn build_trait_method_equivalent_to_new() {
        let knots = [(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)];
        let a = MonotoneCubic::new(&knots).unwrap();
        let b = <MonotoneCubic as Interpolator>::build(&knots).unwrap();
        assert!((a.eval(0.7) - b.eval(0.7)).abs() < 1e-15);
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn len_and_is_empty() {
        let interp = MonotoneCubic::new(&[(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]).unwrap();
        assert_eq!(interp.len(), 3);
        assert!(!interp.is_empty());
    }

    #[test]
    fn clone_yields_equivalent_interpolant() {
        let interp = MonotoneCubic::new(&[(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]).unwrap();
        let copy = interp.clone();
        assert!((interp.eval(0.5) - copy.eval(0.5)).abs() < 1e-15);
    }

    // ─── Flat segment handling ───────────────────────────────────────────

    #[test]
    fn flat_segment_yields_flat_interpolant() {
        // A flat segment in the data forces both endpoint slopes to zero
        // (Fritsch–Carlson §4): the cubic on that segment is then the
        // constant function. This is the "constant data → constant
        // interpolant" identity.
        let knots = [(0.0, 1.0), (1.0, 2.0), (2.0, 2.0), (3.0, 5.0)];
        let interp = MonotoneCubic::new(&knots).unwrap();
        for &t in &[1.1_f64, 1.4, 1.7, 1.95] {
            let v = interp.eval(t);
            assert!((v - 2.0).abs() < 1e-12, "expected flat at t={t}, got {v}");
        }
    }
}
