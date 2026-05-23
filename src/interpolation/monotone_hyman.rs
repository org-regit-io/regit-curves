// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Hyman (1983) monotonicity-preserving filter applied to a cubic-spline base.
//!
//! `MonotoneHyman` builds a C² cubic spline through the knots, reads off the
//! analytic slope of that spline at every knot, and then **clamps each knot
//! slope to the local monotonicity envelope** of Hyman (1983) before
//! re-evaluating as a piecewise cubic Hermite. The result is a C¹ piecewise
//! cubic that **preserves the monotonicity of monotone input data** while
//! retaining the smoothness profile of the underlying spline on intervals
//! where the spline is already monotone.
//!
//! # Algorithm
//!
//! Let the knots be `(t_0, y_0), ..., (t_{n-1}, y_{n-1})` with strictly
//! increasing `t_i`, and let `h_i = t_{i+1} - t_i`,
//! `S_i = (y_{i+1} - y_i) / h_i` be the secant on segment `i`.
//!
//! 1. **Base slopes.** Build a C² cubic spline through the knots (natural
//!    boundary by default; see [`MonotoneHyman::with_boundary`]) and read its
//!    analytic derivative `m_i = y'(t_i)` at each knot. Because the base
//!    spline is C² at every interior knot, the left- and right-derivative
//!    agree there and `m_i` is unambiguous.
//!
//! 2. **Hyman filter at interior knots** (`1 <= i <= n-2`).
//!
//!    ```text
//!    if S_{i-1} * S_i > 0:   // data is locally monotone
//!        m_i' = sign(S_{i-1}) * min(|m_i|, 3 * min(|S_{i-1}|, |S_i|))
//!    else:                    // turning point
//!        m_i' = 0
//!    ```
//!
//!    The clamp `|m_i'| <= 3 * min(|S_{i-1}|, |S_i|)` is the sufficient
//!    monotonicity bound from Hyman (1983) §3 (a refinement of the
//!    Fritsch–Carlson 1980 region: when both secants share a sign, the slope
//!    region `|m_i| <= 3 * min(|S|, |S|)` lies inside the Fritsch–Carlson
//!    disc `α² + β² <= 9` for every adjacent segment).
//!
//! 3. **Hyman filter at endpoints.** With only one adjacent secant the
//!    bound degenerates to
//!
//!    ```text
//!    if sign(m_0) == sign(S_0):
//!        m_0' = sign(S_0) * min(|m_0|, 3 * |S_0|)
//!    else:
//!        m_0' = 0
//!    ```
//!
//!    and symmetrically at the right endpoint with `S_{n-2}` in place of
//!    `S_0`.
//!
//! 4. **Evaluation.** With the filtered slopes `m_i'` in hand the interpolant
//!    is the standard cubic Hermite on each segment. On segment `i` with
//!    `u = (t - t_i) / h_i`:
//!
//!    ```text
//!    y(t) = (2u³ − 3u² + 1) * y_i
//!         + (u³ − 2u² + u)  * h_i * m_i'
//!         + (-2u³ + 3u²)    * y_{i+1}
//!         + (u³ − u²)       * h_i * m_{i+1}'.
//!    ```
//!
//!    The first derivative is the analytic derivative of the same cubic.
//!
//! # Smoothness
//!
//! The base spline is C². The filter only touches the slope at a knot — it
//! never alters the value — so the resulting interpolant is **C¹ everywhere**
//! (the slopes `m_i'` are shared between the two segments meeting at knot `i`)
//! but is **not C²** at any knot where the filter modifies the slope. On the
//! segments where the unfiltered spline slope already lies inside the
//! monotonicity envelope the filter is a no-op and the interpolant coincides
//! with the base spline; in particular, the construction reproduces every
//! linear function exactly (the spline reproduces it; the filter accepts the
//! constant slope) and every cubic where the spline already satisfies the
//! envelope on every segment.
//!
//! # Filter form
//!
//! We use the **Dougherty–Edelman–Hyman (1989)** form of the clamp,
//! `|m_i'| <= 3 * min(|S_{i-1}|, |S_i|)`, rather than Hyman's original
//! `|m_i'| <= 3 * min(|m_i|, |S_{i-1}|, |S_i|)`. The two clamps coincide on
//! the typical case `|m_i| <= 3 * min(|S|, |S|)`; both are monotonicity-
//! preserving. The 1989 form is the version implemented by `QuantLib`'s
//! `MonotonicCubicNaturalSpline` and is the modern reference.
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
//! - Hyman, J. M., "Accurate monotonicity preserving cubic interpolation",
//!   *SIAM J. Sci. Stat. Comput.* 4(4):645–654 (1983). DOI 10.1137/0904045.
//!   The original filter applied to a C² cubic-spline base; §3 derives the
//!   sufficient monotonicity bound `|m_i| <= 3 * min(|S|, |S|)`.
//! - Dougherty, R. L., Edelman, A. & Hyman, J. M., "Nonnegativity-,
//!   monotonicity-, or convexity-preserving cubic and quintic Hermite
//!   interpolation", *Math. Comp.* 52(186):471–494 (1989). DOI
//!   10.1090/S0025-5718-1989-0962209-1. Modern statement of the clamp used
//!   here; resolves the degeneracy in Hyman's original strict-equality form.
//! - Fritsch, F. N. & Carlson, R. E., "Monotone piecewise cubic
//!   interpolation", *SIAM J. Numer. Anal.* 17(2):238–246 (1980). DOI
//!   10.1137/0717021. The monotonicity-region theorem the Hyman bound
//!   localises.

use crate::errors::CurveError;

use super::Interpolator;
use super::cubic_spline::{CubicSpline, SplineBoundary};

/// Hyman (1983) monotonicity-preserving filter applied to a cubic-spline base.
///
/// Constructs a C² cubic spline through the knots (natural boundary by
/// default), extracts its analytic slope at each knot, applies the Hyman
/// monotonicity clamp segment-by-segment, and evaluates as a cubic Hermite.
/// **Monotone input data produces a monotone interpolant by construction**;
/// non-monotone input is still interpolated, with the slope at every turning
/// point forced to zero.
///
/// The interpolant is C¹ everywhere by construction; it is **not** C² in
/// general (the filter breaks C² wherever it activates).
///
/// Flat-extrapolates outside the knot range (eval returns `y_0` below the
/// first knot and `y_{n-1}` above the last).
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::{Interpolator, MonotoneHyman};
///
/// // A monotone-increasing knot set: Hyman preserves monotonicity.
/// let interp =
///     MonotoneHyman::new(&[(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (3.0, 9.0)]).unwrap();
/// // Knot reproduction.
/// assert!((interp.eval(0.0) - 0.0).abs() < 1e-12);
/// assert!((interp.eval(2.0) - 4.0).abs() < 1e-12);
/// // Monotonicity preserved between knots.
/// assert!(interp.eval(0.5) <= interp.eval(1.5));
/// ```
#[derive(Debug, Clone)]
pub struct MonotoneHyman {
    /// Knot times, strictly increasing.
    times: Vec<f64>,
    /// Knot values, one per knot time.
    values: Vec<f64>,
    /// Hyman-filtered Hermite slopes `m_i'`, one per knot. Derived from the
    /// base spline's analytic slope and clamped to the monotonicity envelope
    /// of Hyman (1983).
    slopes: Vec<f64>,
}

impl MonotoneHyman {
    /// Builds a Hyman-filtered monotone cubic interpolant from a slice of
    /// `(t, y)` knots using the [`SplineBoundary::Natural`] cubic spline as
    /// the base.
    ///
    /// Validation:
    ///
    /// - `knots.len() >= 2`.
    /// - `knots[i].0 < knots[i + 1].0` (strictly increasing times).
    /// - Every `t` is finite.
    /// - Every `y` is finite (no positivity requirement).
    ///
    /// On exactly two knots the interpolant reduces to linear interpolation
    /// on the single segment (the natural spline through two points is the
    /// straight line; its slope is the common secant; the Hyman clamp is a
    /// no-op).
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
    /// - [`CurveError::Type`] wrapping a [`crate::TypeError::NonFinite`] if
    ///   the internal cubic-spline solve reports a numerical failure
    ///   (propagated from the tridiagonal solver).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::interpolation::MonotoneHyman;
    /// use regit_curves::CurveError;
    ///
    /// assert!(MonotoneHyman::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]).is_ok());
    /// assert!(matches!(
    ///     MonotoneHyman::new(&[(0.0, 1.0)]).unwrap_err(),
    ///     CurveError::TooFewNodes { found: 1 },
    /// ));
    /// ```
    pub fn new(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
        Self::with_boundary(knots, SplineBoundary::Natural)
    }

    /// Builds a Hyman-filtered monotone cubic interpolant using a specified
    /// cubic-spline boundary condition for the base spline.
    ///
    /// The choice of boundary only affects the base spline's endpoint slopes;
    /// the Hyman clamp then operates uniformly on all knot slopes. Use
    /// [`SplineBoundary::Natural`] (the default of [`MonotoneHyman::new`])
    /// for the conservative case, [`SplineBoundary::NotAKnot`] to match the
    /// `QuantLib` default base spline, or
    /// [`SplineBoundary::Clamped`] when explicit endpoint slopes are known.
    ///
    /// # Errors
    ///
    /// Same as [`MonotoneHyman::new`], with the additional possibility of
    /// [`CurveError::Type`] when the clamped boundary slopes are non-finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::interpolation::{Interpolator, MonotoneHyman, SplineBoundary};
    ///
    /// let interp = MonotoneHyman::with_boundary(
    ///     &[(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)],
    ///     SplineBoundary::NotAKnot,
    /// )
    /// .unwrap();
    /// assert!((interp.eval(1.0) - 1.0).abs() < 1e-12);
    /// ```
    pub fn with_boundary(
        knots: &[(f64, f64)],
        boundary: SplineBoundary,
    ) -> Result<Self, CurveError> {
        if knots.len() < 2 {
            return Err(CurveError::TooFewNodes { found: knots.len() });
        }
        let count = knots.len();
        let mut times = Vec::with_capacity(count);
        let mut values = Vec::with_capacity(count);
        for (idx, &(time, value)) in knots.iter().enumerate() {
            if !time.is_finite() {
                return Err(CurveError::InvalidTime { t: time });
            }
            if !value.is_finite() {
                return Err(CurveError::NonPositiveDiscount {
                    at_index: idx,
                    value,
                });
            }
            if idx > 0 {
                let prev = times[idx - 1];
                // Exact equality is the correct test here — a duplicate
                // grid time is a structural defect of the input, not a
                // numerical approximation. `clippy::float_cmp` flags this
                // by default; we suppress for this canonical use case.
                #[allow(clippy::float_cmp)]
                let is_duplicate = time == prev;
                if is_duplicate {
                    return Err(CurveError::DuplicateNode { t: time });
                }
                if time < prev {
                    return Err(CurveError::NodesNotIncreasing { at_index: idx });
                }
            }
            times.push(time);
            values.push(value);
        }

        // Step 1 — build the C² cubic-spline base and read its analytic
        // derivative at every knot. The spline-construction path performs the
        // same validation we just ran; passing the (already-validated) knot
        // slice through it carries the spline solver's error semantics
        // through the `?`.
        let base = CubicSpline::new(knots, boundary)?;
        let mut slopes: Vec<f64> = times
            .iter()
            .map(|&t| {
                // The base is C¹ everywhere; `deriv` always returns `Some`
                // inside the knot range. We pull a finite slope and fall back
                // to zero only if the spline reports an extrapolation-region
                // None (which never happens for `t_i` in `[t_0, t_{n-1}]`).
                base.deriv(t).unwrap_or(0.0)
            })
            .collect();

        // Step 2 — secant slopes S_i on every segment.
        let mut secants = Vec::with_capacity(count - 1);
        for idx in 0..count - 1 {
            let dt = times[idx + 1] - times[idx];
            secants.push((values[idx + 1] - values[idx]) / dt);
        }

        // Step 3 — apply the Hyman filter to each knot slope. Endpoints use
        // the single adjacent secant; interior knots use the two bracketing
        // secants.
        if count == 2 {
            // Single-segment case: both endpoints share the same secant.
            // The filter collapses both slopes to that common secant.
            slopes[0] = filter_endpoint(slopes[0], secants[0]);
            slopes[1] = filter_endpoint(slopes[1], secants[0]);
        } else {
            slopes[0] = filter_endpoint(slopes[0], secants[0]);
            slopes[count - 1] = filter_endpoint(slopes[count - 1], secants[count - 2]);
            for idx in 1..count - 1 {
                slopes[idx] = filter_interior(slopes[idx], secants[idx - 1], secants[idx]);
            }
        }

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
    /// successfully constructed `MonotoneHyman` (which requires `>= 2` knots);
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
        let count = self.times.len();
        if t <= self.times[0] {
            return 0;
        }
        if t >= self.times[count - 1] {
            return count - 2;
        }
        let mut lo = 0_usize;
        let mut hi = count - 1;
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

/// Hyman (1983) interior-knot filter.
///
/// `m` is the base-spline slope at the interior knot; `s_left`, `s_right` are
/// the two bracketing secants. Returns the filtered slope: zero at a turning
/// point (`s_left * s_right <= 0`), and the original slope clamped to
/// `3 * min(|s_left|, |s_right|)` with the secants' shared sign otherwise.
#[inline]
fn filter_interior(m: f64, s_left: f64, s_right: f64) -> f64 {
    if s_left * s_right <= 0.0 {
        // Turning point or plateau — zero the slope.
        return 0.0;
    }
    let envelope = 3.0 * s_left.abs().min(s_right.abs());
    let clamped = m.abs().min(envelope);
    clamped.copysign(s_left)
}

/// Hyman (1983) endpoint filter — single adjacent secant.
///
/// `m` is the base-spline slope at the endpoint; `s` is the adjacent secant
/// (`S_0` at the left endpoint, `S_{n-2}` at the right). Returns the filtered
/// slope: zero on sign mismatch with the secant, otherwise the original slope
/// clamped to `3 * |s|` with the secant's sign.
#[inline]
fn filter_endpoint(m: f64, s: f64) -> f64 {
    if m * s <= 0.0 {
        return 0.0;
    }
    let envelope = 3.0 * s.abs();
    let clamped = m.abs().min(envelope);
    clamped.copysign(s)
}

impl Interpolator for MonotoneHyman {
    fn build(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
        Self::new(knots)
    }

    fn eval(&self, t: f64) -> f64 {
        let count = self.times.len();
        if t <= self.times[0] {
            return self.values[0];
        }
        if t >= self.times[count - 1] {
            return self.values[count - 1];
        }
        let idx = self.locate(t);
        let t_lo = self.times[idx];
        let t_hi = self.times[idx + 1];
        let dt = t_hi - t_lo;
        let u = (t - t_lo) / dt;
        let u2 = u * u;
        let u3 = u2 * u;
        // Standard cubic Hermite basis.
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;
        h00 * self.values[idx]
            + h10 * dt * self.slopes[idx]
            + h01 * self.values[idx + 1]
            + h11 * dt * self.slopes[idx + 1]
    }

    fn deriv(&self, t: f64) -> Option<f64> {
        let count = self.times.len();
        // Flat extrapolation -> zero derivative outside the knot range.
        if t < self.times[0] || t > self.times[count - 1] {
            return Some(0.0);
        }
        let idx = self.locate(t);
        let t_lo = self.times[idx];
        let t_hi = self.times[idx + 1];
        let dt = t_hi - t_lo;
        let u = (t - t_lo) / dt;
        let u2 = u * u;
        // Derivatives of the cubic Hermite basis (chain rule absorbs the
        // 1/h factor on the value-weighted terms; the slope-weighted terms
        // contribute the polynomial directly since they were multiplied by
        // h in `eval`).
        let dh00 = (6.0 * u2 - 6.0 * u) / dt;
        let dh10 = 3.0 * u2 - 4.0 * u + 1.0;
        let dh01 = (-6.0 * u2 + 6.0 * u) / dt;
        let dh11 = 3.0 * u2 - 2.0 * u;
        Some(
            dh00 * self.values[idx]
                + dh10 * self.slopes[idx]
                + dh01 * self.values[idx + 1]
                + dh11 * self.slopes[idx + 1],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Construction & validation ───────────────────────────────────────

    #[test]
    fn rejects_empty() {
        let err = MonotoneHyman::new(&[]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 0 }));
    }

    #[test]
    fn rejects_single_knot() {
        let err = MonotoneHyman::new(&[(0.0, 1.0)]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 1 }));
    }

    #[test]
    fn rejects_non_monotone_times() {
        let err = MonotoneHyman::new(&[(0.0, 1.0), (2.0, 0.9), (1.0, 0.95)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NodesNotIncreasing { at_index: 2 }
        ));
    }

    #[test]
    fn rejects_duplicate_times() {
        let err = MonotoneHyman::new(&[(0.0, 1.0), (1.0, 0.95), (1.0, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::DuplicateNode { .. }));
    }

    #[test]
    fn rejects_nan_value() {
        let err = MonotoneHyman::new(&[(0.0, 1.0), (1.0, f64::NAN), (2.0, 2.0)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_inf_value() {
        let err = MonotoneHyman::new(&[(0.0, 1.0), (1.0, f64::INFINITY), (2.0, 2.0)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_nan_time() {
        let err = MonotoneHyman::new(&[(0.0, 1.0), (f64::NAN, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn rejects_inf_time() {
        let err = MonotoneHyman::new(&[(0.0, 1.0), (f64::INFINITY, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    // ─── Knot reproduction ───────────────────────────────────────────────

    #[test]
    fn knot_reproduction_exact() {
        let knots = [
            (0.0, 1.0),
            (0.5, 0.97),
            (1.0, 0.95),
            (2.0, 0.90),
            (5.0, 0.78),
        ];
        let interp = MonotoneHyman::new(&knots).unwrap();
        for &(t, y) in &knots {
            let v = interp.eval(t);
            assert!((v - y).abs() < 1e-12, "knot ({t}, {y}) -> {v}");
        }
    }

    // ─── Monotonicity preservation on random monotone data ───────────────

    /// Deterministic LCG — enough randomness for a property-style test
    /// without pulling in a dev-dependency on `rand`. Numerical Recipes'
    /// "ranqd1" constants (Press et al. 2007 §7.1). Matches the PRNG used
    /// in `monotone_cubic.rs::tests`.
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
                t += 0.05 + rng.next_unit();
                y += 0.01 + 5.0 * rng.next_unit();
            }
            let knots: Vec<(f64, f64)> =
                times.iter().copied().zip(values.iter().copied()).collect();
            let interp = MonotoneHyman::new(&knots).unwrap();

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

    // ─── Hyman 1983 RPN15A oracle ────────────────────────────────────────

    #[test]
    fn rpn15a_monotone_on_fine_grid_and_discriminator() {
        // Hyman 1983 RPN15A — 9-point CDF-like data. Transcribed from
        // doc/RESEARCH.md §2.5. The Hyman-filtered interpolant must:
        //   (a) reproduce the knot values exactly;
        //   (b) stay monotone on a fine grid;
        //   (c) satisfy f(11.0) <= 1.0 — the qualitative oracle that
        //       discriminates Hyman from the unfiltered natural cubic
        //       spline (which overshoots to f(11.0) > 1.0 there).
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
        let interp = MonotoneHyman::new(&knots).unwrap();

        // Knot reproduction.
        for &(t, y) in &knots {
            let v = interp.eval(t);
            assert!((v - y).abs() < 1e-12, "RPN15A knot ({t}, {y}) -> {v}");
        }

        // Monotonicity on a fine grid.
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
        // interpolant satisfies f(11.0) <= 1.0 (where an unfiltered cubic
        // spline overshoots to f(11) > 1.0).
        let v11 = interp.eval(11.0);
        assert!(v11 <= 1.0, "Hyman filter failed at x=11.0: f={v11}");
    }

    #[test]
    fn rpn15a_filter_strictly_below_one_at_eleven() {
        // Sharper version of the §2.5 discriminator: the Hyman filter is
        // not merely monotonicity-preserving in the weak sense — the
        // interpolated value at x = 11.0 lies between the bracketing knots
        // (0.943_740 and 0.998_636), with safe headroom from `1.0`.
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
        let interp = MonotoneHyman::new(&knots).unwrap();
        let v = interp.eval(11.0);
        assert!(
            (0.943_740..=0.998_636).contains(&v),
            "Hyman filter at x=11.0: got {v}, expected in [0.943_740, 0.998_636]"
        );
    }

    // ─── Non-monotone input: turning point zeros the slope ───────────────

    #[test]
    fn non_monotone_input_zeros_slope_at_turning_point() {
        // A single dip in an otherwise increasing series. At the dip knot
        // the bracketing secants have opposite signs (positive on the left,
        // negative on the right), so the Hyman filter forces the slope to
        // exactly zero there.
        let knots = [
            (0.0, 0.0),
            (1.0, 2.0),
            (2.0, 1.0), // turning point
            (3.0, 4.0),
            (4.0, 8.0),
        ];
        let interp = MonotoneHyman::new(&knots).unwrap();
        for &(t, y) in &knots {
            let v = interp.eval(t);
            assert!((v - y).abs() < 1e-12, "knot ({t}, {y}) -> {v}");
        }
        let d = interp.deriv(2.0).unwrap();
        assert!(
            d.abs() < 1e-15,
            "expected zero slope at turning point, got {d}"
        );
    }

    // ─── Linear-function reproduction ────────────────────────────────────

    #[test]
    fn reproduces_linear_function() {
        // y = 2 + 3*x at non-uniform knot spacing. The cubic-spline base
        // recovers any linear function exactly (M_i = 0 for all i); the
        // Hyman clamp accepts the common slope (|m| = |S| <= 3 |S|), so
        // the filtered slopes equal 3 at every knot and the interpolant
        // is the underlying line.
        let f = |x: f64| 2.0 + 3.0 * x;
        let knots: Vec<(f64, f64)> = [0.0_f64, 0.5, 1.7, 3.1, 4.0, 6.0, 9.0]
            .iter()
            .map(|&x| (x, f(x)))
            .collect();
        let interp = MonotoneHyman::new(&knots).unwrap();
        for &t in &[0.1_f64, 0.7, 1.0, 2.5, 3.7, 5.2, 7.9] {
            let v = interp.eval(t);
            let expected = f(t);
            assert!(
                (v - expected).abs() < 1e-12,
                "t={t}: got {v}, want {expected}"
            );
        }
        // The filtered slope at every knot is the constant 3.
        for &m in &interp.slopes {
            assert!((m - 3.0).abs() < 1e-12, "slope = {m}");
        }
        // Derivative is the constant slope.
        for &t in &[0.3_f64, 1.8, 4.5, 7.0] {
            let d = interp.deriv(t).unwrap();
            assert!((d - 3.0).abs() < 1e-12, "t={t}: deriv {d}");
        }
    }

    // ─── Plateau (constant input) ────────────────────────────────────────

    #[test]
    fn plateau_stays_flat() {
        // Constant data: every secant is zero, so the interior filter
        // (S_{i-1} * S_i = 0 <= 0) and the endpoint filter (m*s = 0 <= 0)
        // both force every slope to zero. The cubic Hermite collapses to
        // the constant function on every segment.
        let interp = MonotoneHyman::new(&[(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]).unwrap();
        for &m in &interp.slopes {
            #[allow(clippy::float_cmp)]
            let is_zero = m == 0.0;
            assert!(is_zero, "expected zero slope, got {m}");
        }
        for t in [0.0_f64, 0.1, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0, 2.5, 3.0] {
            let v = interp.eval(t);
            assert!((v - 1.0).abs() < 1e-15, "t={t}: v={v}");
        }
    }

    // ─── C¹ continuity at interior knots ─────────────────────────────────

    #[test]
    fn c1_continuous_at_interior_knots() {
        // Compare the one-sided finite-difference derivative on each side of
        // every interior knot. Since the slopes m_i' are shared between the
        // segments meeting at knot i, the interpolant is C¹ and the two
        // one-sided FDs must agree to numerical precision.
        let knots = [
            (0.0, 0.0_f64),
            (1.0, 1.5),
            (2.5, 3.0),
            (4.0, 7.0),
            (5.0, 12.0),
            (7.0, 13.0),
            (10.0, 14.5),
        ];
        let interp = MonotoneHyman::new(&knots).unwrap();
        let h = 1e-6_f64;
        for &(t, _) in &knots[1..knots.len() - 1] {
            let d_left = (interp.eval(t) - interp.eval(t - h)) / h;
            let d_right = (interp.eval(t + h) - interp.eval(t)) / h;
            assert!(
                (d_left - d_right).abs() < 1e-5,
                "C^1 mismatch at t={t}: left={d_left}, right={d_right}"
            );
        }
    }

    // ─── Two-knot degenerate case ────────────────────────────────────────

    #[test]
    fn two_knot_reduces_to_linear() {
        // With exactly two knots, the natural base spline collapses to the
        // straight line; the Hyman clamp at both endpoints reduces to the
        // common secant slope. The cubic Hermite is then exactly linear.
        let interp = MonotoneHyman::new(&[(0.0, 1.0), (2.0, 5.0)]).unwrap();
        for &t in &[0.0_f64, 0.25, 0.5, 1.0, 1.5, 2.0] {
            let expected = 1.0 + 2.0 * t;
            let v = interp.eval(t);
            assert!(
                (v - expected).abs() < 1e-15,
                "t={t}: got {v}, want {expected}"
            );
        }
        // Derivative is the constant secant slope.
        let d = interp.deriv(1.0).unwrap();
        assert!((d - 2.0).abs() < 1e-15);
    }

    // ─── Filter clamp activation ─────────────────────────────────────────

    #[test]
    fn filter_clamps_to_envelope_when_spline_overshoots() {
        // Construct a sharply-rising knot set that the unfiltered natural
        // cubic spline would interpolate with a slope at the central knot
        // exceeding 3 * min(|S_left|, |S_right|). The Hyman filter must
        // clamp it to that envelope.
        //
        // Choose three knots with very different secant magnitudes: a small
        // step then a larger one. The natural spline's M_1 is large; its
        // analytic slope at the right knot can exceed 3 * S_right.
        let knots = [(0.0, 0.0), (0.1, 0.5), (1.0, 0.6)];
        let interp = MonotoneHyman::new(&knots).unwrap();
        // Secants: S_0 = 5.0, S_1 = 0.5/0.9 ≈ 0.5556.
        let s0 = 5.0_f64;
        let s1 = 0.1_f64 / 0.9;
        let envelope_mid = 3.0 * s0.min(s1);
        assert!(
            interp.slopes[1].abs() <= envelope_mid + 1e-12,
            "interior slope = {}, envelope = {envelope_mid}",
            interp.slopes[1]
        );
        // And the slopes have the secants' (positive) sign.
        for &m in &interp.slopes {
            assert!(m >= 0.0, "slope = {m}");
        }
    }

    // ─── Boundary toggle ─────────────────────────────────────────────────

    #[test]
    fn with_boundary_not_a_knot_runs() {
        // Smoke-test the `with_boundary` constructor with the not-a-knot
        // base spline. The interpolant still reproduces the knots and stays
        // monotone on a monotone fixture.
        let knots = [
            (0.0, 0.0_f64),
            (1.0, 1.0),
            (2.0, 4.0),
            (3.0, 9.0),
            (4.0, 16.0),
        ];
        let interp = MonotoneHyman::with_boundary(&knots, SplineBoundary::NotAKnot).unwrap();
        for &(t, y) in &knots {
            let v = interp.eval(t);
            assert!((v - y).abs() < 1e-12, "knot ({t}, {y}) -> {v}");
        }
        let mut prev = interp.eval(0.0);
        let mut t = 0.0_f64;
        while t <= 4.0 {
            let v = interp.eval(t);
            assert!((v + 1e-12) >= prev, "non-monotone at t={t}");
            prev = v;
            t += 0.01;
        }
    }

    // ─── Extrapolation ───────────────────────────────────────────────────

    #[test]
    fn flat_extrapolation_left_right() {
        let interp = MonotoneHyman::new(&[(0.5, 0.97), (1.0, 0.95), (2.0, 0.90)]).unwrap();
        assert!((interp.eval(0.0) - 0.97).abs() < 1e-15);
        assert!((interp.eval(-100.0) - 0.97).abs() < 1e-15);
        assert!((interp.eval(3.0) - 0.90).abs() < 1e-15);
        assert!((interp.eval(100.0) - 0.90).abs() < 1e-15);
        // Zero derivative outside the knot range.
        assert!((interp.deriv(-1.0).unwrap() - 0.0).abs() < 1e-15);
        assert!((interp.deriv(5.0).unwrap() - 0.0).abs() < 1e-15);
    }

    // ─── Trait & accessors ───────────────────────────────────────────────

    #[test]
    fn build_trait_method_equivalent_to_new() {
        let knots = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)];
        let a = MonotoneHyman::new(&knots).unwrap();
        let b = <MonotoneHyman as Interpolator>::build(&knots).unwrap();
        assert!((a.eval(0.5) - b.eval(0.5)).abs() < 1e-15);
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn len_and_is_empty() {
        let interp = MonotoneHyman::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 1.5)]).unwrap();
        assert_eq!(interp.len(), 3);
        assert!(!interp.is_empty());
    }

    #[test]
    fn clone_yields_equivalent_interpolant() {
        let interp = MonotoneHyman::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)]).unwrap();
        let copy = interp.clone();
        assert!((interp.eval(0.5) - copy.eval(0.5)).abs() < 1e-15);
    }
}
