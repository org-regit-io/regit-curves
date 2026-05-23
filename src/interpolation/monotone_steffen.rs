// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Steffen (1990) monotone cubic interpolation.
//!
//! `MonotoneSteffen` is a **local, explicit, monotone** Hermite cubic
//! interpolant. On every segment the curve is the standard cubic Hermite
//! polynomial built from the two endpoint values and two endpoint slopes;
//! the slopes are chosen by Steffen's slope-limited weighted average of the
//! two neighbouring secants so that monotonicity of monotone input data is
//! preserved exactly. Unlike Fritsch–Carlson the slope formula is **purely
//! local**: the slope at knot `i` depends only on the secants of segments
//! `[i-1, i]` and `[i, i+1]`, with no global pre-pass or filter.
//!
//! # Construction
//!
//! Let `h_i = t_{i+1} - t_i` and `S_i = (y_{i+1} - y_i) / h_i` for
//! `i = 0, ..., n - 2`. Steffen's interior-knot slope formula (his eq. 11) is
//!
//! ```text
//! p_i = ( S_{i-1} * h_i + S_i * h_{i-1} ) / ( h_{i-1} + h_i )         (1)
//!
//! if S_{i-1} * S_i <= 0:
//!     m_i = 0
//! else:
//!     m_i = ( sign(S_{i-1}) + sign(S_i) )
//!         * min( |S_{i-1}|, |S_i|, |p_i| / 2 )                        (2)
//! ```
//!
//! The first arm zeroes the slope at turning points and at plateaux; the
//! second arm clamps the candidate slope to no more than twice the smaller
//! secant in magnitude — a sufficient condition for monotonicity of the
//! cubic Hermite (Steffen 1990, §2; cf. Fritsch & Carlson 1980, theorem 1).
//! When the two secants share a sign, `sign(S_{i-1}) + sign(S_i) = ±2`,
//! restoring the factor of two that exactly reproduces a common-secant
//! slope (the linear-data limit). The divisor `2` in `|p_i| / 2` comes
//! from Steffen's harmonic-mean slope bound: see his eq. (11) and
//! accompanying discussion.
//!
//! Endpoint slopes are obtained by extrapolating the secant slopes
//! (Steffen §3, equations 12 and 13):
//!
//! ```text
//! m_0     = S_0    + ( S_0    - S_1   ) * h_0    / ( h_0 + h_1     )  (3)
//! m_{n-1} = S_{n-2} + ( S_{n-2} - S_{n-3} ) * h_{n-2} / ( h_{n-3} + h_{n-2} )  (4)
//! ```
//!
//! followed by Steffen's endpoint sign / magnitude limiter:
//!
//! ```text
//! if sign(m_e) != sign(S_e):  m_e = 0
//! elif |m_e| > 2 * |S_e|:     m_e = 2 * S_e
//! ```
//!
//! where `S_e` is the adjacent secant (`S_0` for the left endpoint,
//! `S_{n-2}` for the right). For `n = 2` the extrapolation degenerates;
//! both endpoint slopes are set to the unique secant `S_0` and the result
//! is a linear interpolant.
//!
//! # Evaluation
//!
//! On segment `i` with `u = (t - t_i) / h_i, h = h_i`, the cubic Hermite is
//!
//! ```text
//! y(t) = (2u^3 - 3u^2 + 1) * y_i
//!      + (u^3 - 2u^2 + u)  * h * m_i
//!      + (-2u^3 + 3u^2)    * y_{i+1}
//!      + (u^3 - u^2)       * h * m_{i+1}.
//! ```
//!
//! The first derivative follows in closed form:
//!
//! ```text
//! y'(t) = (6u^2 - 6u) / h * y_i + (3u^2 - 4u + 1) * m_i
//!       + (-6u^2 + 6u) / h * y_{i+1} + (3u^2 - 2u) * m_{i+1}.
//! ```
//!
//! Because the slopes `m_i` are shared across the segments meeting at knot
//! `i`, the interpolant is C¹ at every interior knot.
//!
//! # Invariants
//!
//! - At least two knots.
//! - Knot times strictly increasing.
//! - Knot times and values both finite (no `NaN`, no `±∞`).
//!
//! Unlike `LogLinear` there is **no positivity constraint** — Steffen
//! interpolates any finite real-valued field. Monotonicity preservation is
//! the structural guarantee; if the input data is not monotone the limiter
//! still applies (slopes are zeroed at turning points), and the interpolant
//! tracks the data without spurious overshoot.
//!
//! # Extrapolation
//!
//! Flat extrapolation outside the knot range — `eval(t) = y_0` for
//! `t <= t_0` and `eval(t) = y_{n-1}` for `t >= t_{n-1}`. The derivative
//! in the extrapolation region is therefore `0`. This matches the
//! conservative market default used elsewhere in the crate.
//!
//! # Relation to GSL
//!
//! GNU Scientific Library ships the same method as `gsl_interp_steffen`
//! (`interpolation/steffen.c`); the two implementations agree numerically
//! within `f64` epsilon on the same inputs, which makes GSL a convenient
//! cross-oracle.
//!
//! # References
//!
//! - Steffen, M., "A simple method for monotonic interpolation in one
//!   dimension", *Astronomy & Astrophysics* 239:443–450 (1990). NASA ADS
//!   bibcode 1990A&A...239..443S. §2 derives the interior-knot slope
//!   formula (his equation 11); §3 derives the endpoint formulas (his
//!   equations 12-13) and the endpoint limiter.
//! - Fritsch, F. N. & Carlson, R. E., "Monotone piecewise cubic
//!   interpolation", *SIAM Journal on Numerical Analysis* 17(2):238–246
//!   (1980), theorem 1 — the monotonicity bound `|m_i| <= 3 * |S|` that
//!   underlies Steffen's harmonic-mean slope clamp.
//! - GNU Scientific Library, `interpolation/steffen.c` — independent
//!   implementation of the same method; useful as a cross-oracle.

use crate::errors::CurveError;

use super::Interpolator;

/// Steffen (1990) monotone cubic interpolant over a set of `(t, y)` knots.
///
/// A **local, explicit, monotone** Hermite cubic — each interior slope is a
/// slope-limited weighted average of the two neighbouring secants, with no
/// global pre-pass. Monotonic input data produces a monotonic interpolant
/// by construction; at turning points and plateaux the slope is forced to
/// zero (the limiter rule). The interpolant is C¹ at every interior knot.
///
/// No positivity constraint is imposed on `y` — any finite real value is
/// accepted. Flat-extrapolates outside the knot range (eval returns `y_0`
/// below the first knot and `y_{n-1}` above the last).
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::{Interpolator, MonotoneSteffen};
///
/// let interp = MonotoneSteffen::new(&[(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (3.0, 9.0)]).unwrap();
/// // Knot reproduction.
/// assert!((interp.eval(0.0) - 0.0).abs() < 1e-15);
/// assert!((interp.eval(2.0) - 4.0).abs() < 1e-15);
/// // Monotonicity preserved between knots.
/// assert!(interp.eval(0.5) >= 0.0);
/// assert!(interp.eval(0.5) <= 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct MonotoneSteffen {
    /// Knot times, strictly increasing.
    times: Vec<f64>,
    /// Knot values, one per knot time.
    values: Vec<f64>,
    /// Hermite slope at every knot — one per knot, computed at construction
    /// from Steffen's interior formula (2) and endpoint formulas (3)/(4).
    slopes: Vec<f64>,
}

impl MonotoneSteffen {
    /// Builds a Steffen monotone cubic interpolant from a slice of `(t, y)`
    /// knots.
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
    /// use regit_curves::interpolation::MonotoneSteffen;
    /// use regit_curves::CurveError;
    ///
    /// assert!(MonotoneSteffen::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]).is_ok());
    /// assert!(matches!(
    ///     MonotoneSteffen::new(&[(0.0, 1.0)]).unwrap_err(),
    ///     CurveError::TooFewNodes { found: 1 },
    /// ));
    /// ```
    pub fn new(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
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

        // Secant slopes S_i = (y_{i+1} - y_i) / h_i, h_i = t_{i+1} - t_i.
        // Length count - 1.
        let mut secants = Vec::with_capacity(count - 1);
        for idx in 0..count - 1 {
            let dt = times[idx + 1] - times[idx];
            secants.push((values[idx + 1] - values[idx]) / dt);
        }

        let mut slopes = vec![0.0_f64; count];

        if count == 2 {
            // Degenerate case: a single segment is interpolated linearly
            // by setting both endpoint slopes to the unique secant. The
            // resulting cubic collapses to a straight line.
            slopes[0] = secants[0];
            slopes[1] = secants[0];
            return Ok(Self {
                times,
                values,
                slopes,
            });
        }

        // Interior slopes (Steffen eq. 11).
        for idx in 1..count - 1 {
            let h_left = times[idx] - times[idx - 1];
            let h_right = times[idx + 1] - times[idx];
            let s_left = secants[idx - 1];
            let s_right = secants[idx];
            // Weighted candidate slope p_i.
            let p_cand = (s_left * h_right + s_right * h_left) / (h_left + h_right);
            slopes[idx] = if s_left * s_right <= 0.0 {
                // Turning point or plateau — limiter forces zero slope.
                0.0
            } else {
                // Both secants share a sign. The Steffen sum-of-signs
                // factor is ±2 here; the slope is therefore
                //
                //   m_i = 2 * sign(S) * min(|S_{i-1}|, |S_i|, |p_i| / 2).
                //
                // On equal-secant (linear) data this collapses to the
                // common secant S — Steffen is exact on linear inputs.
                let abs_left = s_left.abs();
                let abs_right = s_right.abs();
                let half_p = p_cand.abs() * 0.5;
                let m_abs = abs_left.min(abs_right).min(half_p);
                2.0 * m_abs.copysign(s_left)
            };
        }

        // Left endpoint (Steffen eq. 12).
        let h0 = times[1] - times[0];
        let h1 = times[2] - times[1];
        let m0_extrap = secants[0] + (secants[0] - secants[1]) * h0 / (h0 + h1);
        slopes[0] = limit_endpoint(m0_extrap, secants[0]);

        // Right endpoint (Steffen eq. 13).
        let h_last = times[count - 1] - times[count - 2];
        let h_prev = times[count - 2] - times[count - 3];
        let m_last_extrap = secants[count - 2]
            + (secants[count - 2] - secants[count - 3]) * h_last / (h_prev + h_last);
        slopes[count - 1] = limit_endpoint(m_last_extrap, secants[count - 2]);

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
    /// successfully constructed `MonotoneSteffen` (which requires `>= 2`
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

/// Steffen's endpoint sign / magnitude limiter (§3, formula 12-13 tail).
///
/// `m` is the extrapolated candidate slope at the endpoint; `s` is the
/// adjacent secant (`S_0` at the left endpoint, `S_{n-2}` at the right).
/// Zeroes the slope on sign mismatch; clamps to `2 * s` on overshoot.
#[inline]
fn limit_endpoint(m: f64, s: f64) -> f64 {
    if m * s <= 0.0 {
        // Either signs disagree, or the candidate slope is zero — fall
        // back to a flat endpoint to preserve monotonicity.
        0.0
    } else if m.abs() > 2.0 * s.abs() {
        2.0 * s
    } else {
        m
    }
}

impl Interpolator for MonotoneSteffen {
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
        let err = MonotoneSteffen::new(&[]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 0 }));
    }

    #[test]
    fn rejects_single_knot() {
        let err = MonotoneSteffen::new(&[(0.0, 1.0)]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 1 }));
    }

    #[test]
    fn rejects_non_monotone_times() {
        let err = MonotoneSteffen::new(&[(0.0, 1.0), (2.0, 0.9), (1.0, 0.95)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NodesNotIncreasing { at_index: 2 }
        ));
    }

    #[test]
    fn rejects_duplicate_times() {
        let err = MonotoneSteffen::new(&[(0.0, 1.0), (1.0, 0.95), (1.0, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::DuplicateNode { .. }));
    }

    #[test]
    fn rejects_nan_value() {
        let err = MonotoneSteffen::new(&[(0.0, 1.0), (1.0, f64::NAN), (2.0, 2.0)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_inf_value() {
        let err =
            MonotoneSteffen::new(&[(0.0, 1.0), (1.0, f64::INFINITY), (2.0, 2.0)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_nan_time() {
        let err = MonotoneSteffen::new(&[(0.0, 1.0), (f64::NAN, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn rejects_inf_time() {
        let err = MonotoneSteffen::new(&[(0.0, 1.0), (f64::INFINITY, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    // ─── Evaluation correctness ──────────────────────────────────────────

    #[test]
    fn knot_reproduction_exact() {
        let knots = [
            (0.0, 1.0),
            (0.5, 0.97),
            (1.0, 0.95),
            (2.0, 0.90),
            (5.0, 0.78),
        ];
        let interp = MonotoneSteffen::new(&knots).unwrap();
        for &(t, y) in &knots {
            let v = interp.eval(t);
            assert!((v - y).abs() < 1e-14, "knot ({t}, {y}) -> {v}");
        }
    }

    #[test]
    fn two_knot_linear() {
        // n = 2: both endpoint slopes are the secant; the cubic collapses
        // to a straight line.
        let interp = MonotoneSteffen::new(&[(0.0, 1.0), (2.0, 5.0)]).unwrap();
        // Midpoint should be (1 + 5) / 2 = 3.0 exactly.
        let v = interp.eval(1.0);
        assert!((v - 3.0).abs() < 1e-15);
        // The derivative is the constant secant slope.
        let d = interp.deriv(0.5).unwrap();
        assert!((d - 2.0).abs() < 1e-15);
    }

    #[test]
    fn linear_data_is_reproduced_exactly() {
        // y = 2 + 3 * x at integer knots; Steffen must be exact on linear
        // data (the slope formula collapses to the common secant slope).
        let knots: Vec<(f64, f64)> = (0..6)
            .map(|i| {
                let x = f64::from(i);
                (x, 2.0 + 3.0 * x)
            })
            .collect();
        let interp = MonotoneSteffen::new(&knots).unwrap();
        // All interior slopes should equal 3.0 (within fp epsilon).
        for &m in &interp.slopes {
            assert!((m - 3.0).abs() < 1e-13, "slope = {m}");
        }
        // Evaluate at non-knot points and verify the straight-line value.
        for t in [0.25_f64, 0.5, 1.7, 2.3, 3.6, 4.9] {
            let v = interp.eval(t);
            let expected = 2.0 + 3.0 * t;
            assert!((v - expected).abs() < 1e-13, "t={t}: v={v}");
        }
    }

    #[test]
    fn plateau_stays_flat() {
        // Constant data must be reproduced exactly: the limiter forces
        // every slope to zero (S_{i-1} * S_i = 0 <= 0), so the cubic
        // collapses to y(t) = 1.0 on [0, 2].
        let interp = MonotoneSteffen::new(&[(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]).unwrap();
        for &m in &interp.slopes {
            #[allow(clippy::float_cmp)]
            let is_zero = m == 0.0;
            assert!(is_zero, "expected zero slope, got {m}");
        }
        for t in [0.0_f64, 0.1, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0] {
            let v = interp.eval(t);
            assert!((v - 1.0).abs() < 1e-15, "t={t}: v={v}");
        }
    }

    #[test]
    fn turning_point_zero_slope() {
        // Up then down: at the apex the two secants have opposite signs,
        // so Steffen's limiter sets the slope to exactly zero.
        let interp = MonotoneSteffen::new(&[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]).unwrap();
        // Slope at the interior knot (i = 1) should be zero.
        #[allow(clippy::float_cmp)]
        let apex_zero = interp.slopes[1] == 0.0;
        assert!(apex_zero, "apex slope = {}", interp.slopes[1]);
    }

    #[test]
    fn monotonicity_preserved_on_random_monotone_data() {
        // Deterministic xorshift-based PRNG — no external dependency.
        // Build 30 strictly-increasing knot sets with strictly-increasing
        // values, then verify the interpolant is monotone on a fine grid.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let next = |s: &mut u64| -> f64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            // Map the top 53 bits of `*s` into (0, 1] by interpreting them
            // as the mantissa of an f64 in the range [1, 2) and subtracting
            // one. This avoids the `u64 -> f64` precision-loss cast.
            let mantissa = *s >> 11;
            let bits = (1023_u64 << 52) | mantissa;
            (f64::from_bits(bits) - 1.0) + 1e-9
        };

        for trial in 0_u32..30 {
            let count = 5 + (trial as usize % 6); // sizes 5..=10
            let mut times = Vec::with_capacity(count);
            let mut values = Vec::with_capacity(count);
            let mut t = 0.0_f64;
            let mut y = 0.0_f64;
            for _ in 0..count {
                t += next(&mut state) + 0.1;
                y += next(&mut state) + 0.05;
                times.push(t);
                values.push(y);
            }
            let knots: Vec<(f64, f64)> = times
                .iter()
                .zip(values.iter())
                .map(|(&a, &b)| (a, b))
                .collect();
            let interp = MonotoneSteffen::new(&knots).unwrap();

            // Evaluate on a fine grid spanning the full range.
            let t0 = knots[0].0;
            let t_end = knots[count - 1].0;
            let steps = 500_u32;
            let mut prev = interp.eval(t0);
            for k in 1..=steps {
                let frac = f64::from(k) / f64::from(steps);
                let tt = t0 + frac * (t_end - t0);
                let v = interp.eval(tt);
                assert!(
                    v + 1e-12 >= prev,
                    "trial {trial}: non-monotone at t={tt}: prev={prev}, v={v}"
                );
                prev = v;
            }
        }
    }

    #[test]
    fn exponential_growth_preserves_monotonicity() {
        // A specific monotone-but-difficult fixture: exponential growth
        // followed by an inflection. The classical failure mode of an
        // unguarded cubic spline is overshoot at the early knots; Steffen
        // must not exhibit it.
        let knots: Vec<(f64, f64)> = (0..10)
            .map(|i| {
                let x = f64::from(i) * 0.5;
                (x, x.exp())
            })
            .collect();
        let interp = MonotoneSteffen::new(&knots).unwrap();
        let t_end = knots[knots.len() - 1].0;
        let mut prev = interp.eval(0.0);
        let mut t = 0.0_f64;
        while t <= t_end {
            let v = interp.eval(t);
            assert!(
                v + 1e-12 >= prev,
                "non-monotone at t={t}: prev={prev}, v={v}"
            );
            prev = v;
            t += 0.005;
        }
    }

    #[test]
    fn flat_extrapolation_left_right() {
        let interp = MonotoneSteffen::new(&[(0.5, 0.97), (1.0, 0.95), (2.0, 0.90)]).unwrap();
        assert!((interp.eval(0.0) - 0.97).abs() < 1e-15);
        assert!((interp.eval(-100.0) - 0.97).abs() < 1e-15);
        assert!((interp.eval(3.0) - 0.90).abs() < 1e-15);
        assert!((interp.eval(100.0) - 0.90).abs() < 1e-15);
    }

    // ─── Derivative correctness & C¹ continuity ──────────────────────────

    #[test]
    fn deriv_finite_difference_interior() {
        let knots = [(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (3.0, 9.0), (4.0, 16.0)];
        let interp = MonotoneSteffen::new(&knots).unwrap();
        let t = 1.5_f64;
        let dy_dt = interp.deriv(t).unwrap();
        let h = 1e-6_f64;
        let fd = (interp.eval(t + h) - interp.eval(t - h)) / (2.0 * h);
        assert!((dy_dt - fd).abs() < 1e-6, "analytic={dy_dt}, fd={fd}");
    }

    #[test]
    fn c1_continuity_at_interior_knots() {
        // Approach an interior knot from the left and right; finite
        // differences must agree because the Hermite slopes are shared
        // across segments.
        let knots = [(0.0, 0.0), (1.0, 1.0), (2.5, 3.0), (4.0, 4.5), (6.0, 5.0)];
        let interp = MonotoneSteffen::new(&knots).unwrap();
        let h = 1e-7_f64;
        for &(t, _) in &knots[1..knots.len() - 1] {
            let d_left = (interp.eval(t) - interp.eval(t - h)) / h;
            let d_right = (interp.eval(t + h) - interp.eval(t)) / h;
            assert!(
                (d_left - d_right).abs() < 1e-5,
                "C^1 broken at t={t}: left={d_left}, right={d_right}"
            );
        }
    }

    #[test]
    fn deriv_zero_in_extrapolation_region() {
        let interp = MonotoneSteffen::new(&[(0.0, 1.0), (1.0, 0.95), (2.0, 0.90)]).unwrap();
        let d_left = interp.deriv(-1.0).unwrap();
        assert!((d_left - 0.0).abs() < 1e-15);
        let d_right = interp.deriv(3.0).unwrap();
        assert!((d_right - 0.0).abs() < 1e-15);
    }

    // ─── Trait & accessors ───────────────────────────────────────────────

    #[test]
    fn build_trait_method_equivalent_to_new() {
        let knots = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)];
        let a = MonotoneSteffen::new(&knots).unwrap();
        let b = <MonotoneSteffen as Interpolator>::build(&knots).unwrap();
        assert!((a.eval(0.5) - b.eval(0.5)).abs() < 1e-15);
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn len_and_is_empty() {
        let interp = MonotoneSteffen::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 1.5)]).unwrap();
        assert_eq!(interp.len(), 3);
        assert!(!interp.is_empty());
    }

    #[test]
    fn clone_yields_equivalent_interpolant() {
        let interp = MonotoneSteffen::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)]).unwrap();
        let copy = interp.clone();
        assert!((interp.eval(0.5) - copy.eval(0.5)).abs() < 1e-15);
    }
}
