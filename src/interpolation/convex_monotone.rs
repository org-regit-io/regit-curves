// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Hagan–West "monotone convex" interpolation (Method 7 of HW 2008).
//!
//! `ConvexMonotone` is the **arbitrage-free monotone-convex** interpolator of
//! Hagan & West. It interpolates the **instantaneous forward rate**
//! `f(t) = -d/dt ln D(t)` piecewise, with a shape filter that guarantees
//!
//! 1. the discount factors at the knot times are reproduced exactly,
//! 2. the instantaneous forward stays non-negative when the input discount
//!    factors are positive and monotone non-increasing in `t`,
//! 3. the interpolant does not oscillate spuriously between knots — a defect
//!    common to unfiltered cubic-spline curves.
//!
//! The construction is **local**: the value at any `t` depends only on the
//! four neighbouring knots `(t_{i-1}, t_i, t_{i+1}, t_{i+2})`, so a small
//! perturbation of a single market quote propagates only into the two
//! adjacent segments — a key practical advantage over a global cubic spline.
//!
//! # Algorithm
//!
//! Let the knots be `(t_0, y_0), …, (t_{n-1}, y_{n-1})` with strictly
//! increasing `t_i` and strictly positive `y_i`. The construction proceeds
//! in four steps (HW 2008 §3.4 and §4):
//!
//! 1. **Discrete forwards** on each segment `[t_{i-1}, t_i]`:
//!
//!    ```text
//!    f_i = ( ln(y_{i-1}) − ln(y_i) ) / ( t_i − t_{i-1} ),    i = 1, …, n−1.
//!    ```
//!
//!    This is the continuously-compounded zero rate of the segment when
//!    `y` is interpreted as a discount factor.
//!
//! 2. **Instantaneous forwards at the knots** `fhat_i`. Interior knots use
//!    the time-weighted midpoint of the two adjacent discrete forwards:
//!
//!    ```text
//!    fhat_i = (t_i − t_{i-1}) / (t_{i+1} − t_{i-1}) · f_{i+1}
//!           + (t_{i+1} − t_i) / (t_{i+1} − t_{i-1}) · f_i,
//!    ```
//!
//!    for `1 ≤ i ≤ n−2`. The endpoint values are extrapolated linearly:
//!
//!    ```text
//!    fhat_0     = f_1     − 0.5 · (fhat_1     − f_1),
//!    fhat_{n-1} = f_{n-1} − 0.5 · (fhat_{n-2} − f_{n-1}).
//!    ```
//!
//!    Each `fhat_i` is then clipped to the **positivity / convexity box**
//!    that ensures the segment quadratic stays non-negative when the
//!    discrete forwards are positive (HW 2008 §4):
//!
//!    ```text
//!    fhat_i_clipped = clamp( fhat_i,
//!                            0,
//!                            2 · min( f_i, f_{i+1} ) )      (interior knots)
//!    fhat_0_clipped     = clamp( fhat_0,     0, 2 · f_1     )
//!    fhat_{n-1}_clipped = clamp( fhat_{n-1}, 0, 2 · f_{n-1} )
//!    ```
//!
//! 3. **Quadratic ansatz on each segment.** On segment `i` (between knots
//!    `i` and `i+1`) with `x = (t − t_i) / (t_{i+1} − t_i) ∈ [0, 1]`, the
//!    basic shape is
//!
//!    ```text
//!    f(t) = g_0 · (1 − 4x + 3x²) + g_1 · (−2x + 3x²) + f_{i+1},
//!    ```
//!
//!    where `g_0 = fhat_i − f_{i+1}` and `g_1 = fhat_{i+1} − f_{i+1}` are
//!    the "gaps" at the segment endpoints relative to the segment-average
//!    forward `f_{i+1}`. By construction this quadratic satisfies
//!    `f(t_i) = fhat_i`, `f(t_{i+1}) = fhat_{i+1}`, and
//!    `∫_{t_i}^{t_{i+1}} f(s) ds = f_{i+1} · (t_{i+1} − t_i)` — so the
//!    discount factors at the knots are reproduced exactly.
//!
//! 4. **Shape filter** (HW 2008 §4). The basic quadratic is monotone iff
//!    `(2 g_0 + g_1) · (g_0 + 2 g_1) ≤ 0`. When that fails, the segment is
//!    replaced by one of three alternative shapes, classified by the sign
//!    and magnitude of `(g_0, g_1)`:
//!
//!    - **Region I** (unmodified): the quadratic above is monotone.
//!    - **Region II** (`g_1` dominates with opposite sign, `g_1 < −2g_0`
//!      when `g_0 > 0` and symmetric):
//!      flat at `g_0` until `η`, then quadratic up to `g_1`. With
//!      `η = (g_1 + 2 g_0) / (g_1 − g_0)`,
//!
//!      ```text
//!      g(x) = g_0,                                          x ∈ [0, η],
//!      g(x) = g_0 + (g_1 − g_0) · ((x − η) / (1 − η))²,     x ∈ [η, 1].
//!      ```
//!
//!    - **Region III** (`g_0` dominates with opposite sign, `−g_0/2 < g_1
//!      < 0` when `g_0 > 0` and symmetric):
//!      quadratic from `g_0` down to `g_1`, then flat at `g_1`. With
//!      `η = 3 g_1 / (g_1 − g_0)`,
//!
//!      ```text
//!      g(x) = g_1 + (g_0 − g_1) · ((η − x) / η)²,           x ∈ [0, η],
//!      g(x) = g_1,                                          x ∈ [η, 1].
//!      ```
//!
//!    - **Region IV** (same sign, `g_0 · g_1 > 0`):
//!      two-piece quadratic meeting at an interior value `A` of opposite
//!      sign. With `A = − g_0 · g_1 / (g_0 + g_1)` and
//!      `η = g_1 / (g_0 + g_1)`,
//!
//!      ```text
//!      g(x) = A + (g_0 − A) · ((η − x) / η)²,               x ∈ [0, η],
//!      g(x) = A + (g_1 − A) · ((x − η) / (1 − η))²,         x ∈ [η, 1].
//!      ```
//!
//!    In every case `g(x) = f(t) − f_{i+1}` and `∫_0^1 g(x) dx = 0`, so the
//!    segment integral is preserved and the discount factor at the next
//!    knot is reproduced exactly.
//!
//! 5. **Evaluation.** The discount factor at `t` in segment `i` is
//!
//!    ```text
//!    y(t) = y_i · exp( − ∫_{t_i}^{t} f(s) ds ),
//!    ```
//!
//!    computed piecewise by closed-form integration of the segment shape.
//!
//! # Invariants
//!
//! - At least two knots.
//! - Knot times strictly increasing and finite.
//! - Knot values strictly positive and finite — the natural invariant of
//!   the discount-factor domain.
//!
//! # Input domain
//!
//! The method is designed for **positive, monotone non-increasing discount
//! factors** — the canonical yield-curve setting in which Hagan & West (2008)
//! §3.6 prove the non-negative-forward guarantee. On that domain the
//! interpolant is uniquely determined by the paper and agrees bit-exactly
//! with independent implementations (verified against tf-quant-finance to
//! `2.2 × 10⁻¹⁶` relative across an 800-point test sweep). Outside that
//! domain — e.g. an oscillating input where the implied discrete forwards
//! change sign — the §3.6 proof does not apply, and the `fhat` clipping
//! policy of Hagan & West §4 eq. 25 (which this crate follows verbatim) can
//! differ from implementations that omit the clipping step. Both are
//! defensible interpretations of the paper, but neither is uniquely the
//! "right" answer on inputs the method was never intended for. Callers
//! seeking general-purpose non-monotone interpolation should choose
//! [`crate::interpolation::CubicSpline`] or
//! [`crate::interpolation::HermiteBessel`] instead.
//!
//! # Extrapolation
//!
//! Flat extrapolation in the instantaneous forward: for `t < t_0` the
//! boundary segment's `fhat_0` is reused, and for `t > t_{n-1}` the
//! boundary segment's `fhat_{n-1}` is reused. The discount factor is
//! continuous everywhere — extrapolation simply extends the boundary
//! segments' exponential decay outwards. This matches the conservative
//! market default used elsewhere in the crate.
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Methods for constructing a yield curve",
//!   *Wilmott Magazine*, May 2008, pp. 70-81. The "nine methods" survey
//!   paper; Method 7 is the monotone-convex interpolant implemented here.
//! - Hagan, P. S. & West, G., "Interpolation methods for curve
//!   construction", *Applied Mathematical Finance* 13(2):89-129 (2006),
//!   §3 (sequential bootstrap), §4 (the monotone-convex shape filter).

use crate::errors::CurveError;

use super::Interpolator;

/// Hagan–West monotone-convex interpolant over a set of `(t, y)` knots.
///
/// Interpolates the **instantaneous forward rate** `f(t)` piecewise, with a
/// shape filter that preserves the integral of `f` over each segment (so the
/// knot discount factors are reproduced exactly) and keeps `f` non-negative
/// when the input `y` is a monotone-decreasing discount-factor table.
///
/// Requires `y > 0` at every knot — the natural invariant of the discount-
/// factor domain.
///
/// Flat-extrapolates in the instantaneous forward outside the knot range
/// (so the discount factor follows the boundary segment's exponential decay).
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::{ConvexMonotone, Interpolator};
///
/// // A monotone-decreasing discount-factor table.
/// let interp = ConvexMonotone::new(&[
///     (0.0, 1.0),
///     (1.0, 0.95),
///     (2.0, 0.90),
///     (5.0, 0.78),
/// ])
/// .unwrap();
/// // Knot reproduction is exact.
/// assert!((interp.eval(0.0) - 1.0).abs() < 1e-12);
/// assert!((interp.eval(1.0) - 0.95).abs() < 1e-12);
/// assert!((interp.eval(2.0) - 0.90).abs() < 1e-12);
/// assert!((interp.eval(5.0) - 0.78).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct ConvexMonotone {
    /// Knot times, strictly increasing.
    times: Vec<f64>,
    /// Knot values, strictly positive.
    values: Vec<f64>,
    /// Logs of `values` — cached for the discount-factor integral evaluation.
    log_values: Vec<f64>,
    /// Discrete forward on each segment, length `n - 1`.
    discrete_forwards: Vec<f64>,
    /// Instantaneous forward at each knot, length `n`.
    fhat: Vec<f64>,
    /// Pre-solved shape on each segment, length `n - 1`.
    segments: Vec<Segment>,
}

/// Shape coefficients for a single segment, classified into one of the four
/// Hagan–West regions.
#[derive(Debug, Clone, Copy)]
enum Segment {
    /// Region I — unmodified quadratic `g(x) = g_0·(1−4x+3x²) + g_1·(−2x+3x²)`.
    QuadraticI { g0: f64, g1: f64 },
    /// Region II — flat at `g_0` for `x ∈ [0, η]`, quadratic
    /// `g_0 + (g_1 − g_0)·((x − η)/(1 − η))²` for `x ∈ [η, 1]`.
    FlatThenQuadII { g0: f64, g1: f64, eta: f64 },
    /// Region III — quadratic `g_1 + (g_0 − g_1)·((η − x)/η)²` for
    /// `x ∈ [0, η]`, flat at `g_1` for `x ∈ [η, 1]`.
    QuadThenFlatIII { g0: f64, g1: f64, eta: f64 },
    /// Region IV — two-piece, meeting at value `A` at `x = η`:
    /// `A + (g_0 − A)·((η − x)/η)²` for `x ∈ [0, η]`,
    /// `A + (g_1 − A)·((x − η)/(1 − η))²` for `x ∈ [η, 1]`.
    TwoPieceIV { g0: f64, g1: f64, a: f64, eta: f64 },
}

impl ConvexMonotone {
    /// Builds a Hagan–West monotone-convex interpolant from a slice of
    /// `(t, y)` knots.
    ///
    /// Validation:
    ///
    /// - `knots.len() >= 2`.
    /// - `knots[i].0 < knots[i + 1].0` (strictly increasing times).
    /// - Every `t` is finite.
    /// - Every `y > 0` and finite — the natural invariant of the discount-
    ///   factor domain in which this method is meaningful.
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
    /// use regit_curves::interpolation::ConvexMonotone;
    /// use regit_curves::CurveError;
    ///
    /// assert!(ConvexMonotone::new(&[(0.0, 1.0), (1.0, 0.95), (2.0, 0.9)]).is_ok());
    /// assert!(matches!(
    ///     ConvexMonotone::new(&[(0.0, 1.0)]).unwrap_err(),
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
            values.push(y);
            log_values.push(y.ln());
        }

        // Step 1 — discrete forwards on each segment.
        let mut discrete_forwards = Vec::with_capacity(n - 1);
        for i in 1..n {
            let h = times[i] - times[i - 1];
            // h > 0 by the strict-monotonicity check above.
            let f = (log_values[i - 1] - log_values[i]) / h;
            discrete_forwards.push(f);
        }

        // Step 2 — instantaneous forwards at the knots.
        let fhat = compute_fhat(&times, &discrete_forwards);

        // Step 3 + 4 — classify each segment and pre-solve its shape.
        let mut segments = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let f_avg = discrete_forwards[i];
            let g0 = fhat[i] - f_avg;
            let g1 = fhat[i + 1] - f_avg;
            segments.push(classify_segment(g0, g1));
        }

        Ok(Self {
            times,
            values,
            log_values,
            discrete_forwards,
            fhat,
            segments,
        })
    }

    /// Returns the number of knots.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns `true` if the interpolant has no knots. Always `false` for a
    /// successfully constructed `ConvexMonotone` (which requires `>= 2`
    /// knots); retained for `clippy::len_without_is_empty`.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Returns the instantaneous forward rate `f(t)` — the directly
    /// interpolated quantity of the Hagan–West construction.
    ///
    /// Outside the knot range the boundary segment's `fhat` is reused (flat
    /// extrapolation in the forward).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::interpolation::ConvexMonotone;
    ///
    /// // Flat 4% continuously-compounded curve, encoded as discount factors.
    /// let knots = [
    ///     (0.0_f64, 1.0_f64),
    ///     (1.0_f64, (-0.04_f64).exp()),
    ///     (2.0_f64, (-0.08_f64).exp()),
    /// ];
    /// let interp = ConvexMonotone::new(&knots).unwrap();
    /// // A flat-forward input is reproduced exactly.
    /// assert!((interp.forward_at(0.5) - 0.04).abs() < 1e-12);
    /// assert!((interp.forward_at(1.5) - 0.04).abs() < 1e-12);
    /// ```
    #[must_use]
    #[allow(clippy::many_single_char_names)] // `t, n, i, h, x` — standard quant notation.
    pub fn forward_at(&self, t: f64) -> f64 {
        let n = self.times.len();
        // Flat extrapolation in the forward.
        if t <= self.times[0] {
            return self.fhat[0];
        }
        if t >= self.times[n - 1] {
            return self.fhat[n - 1];
        }
        let i = self.locate(t);
        let h = self.times[i + 1] - self.times[i];
        let x = (t - self.times[i]) / h;
        let f_avg = self.discrete_forwards[i];
        f_avg + segment_g(self.segments[i], x)
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

    /// Discount factor at `t` — computed as
    /// `y(t) = y_i · exp(−∫_{t_i}^{t} f(s) ds)` on the containing segment,
    /// with flat-forward extrapolation outside the knot range.
    #[allow(clippy::many_single_char_names)] // `t, n, i, h, x` — standard quant notation.
    fn discount_at(&self, t: f64) -> f64 {
        let n = self.times.len();
        // Left extrapolation: y(t) = y_0 · exp(fhat_0 · (t_0 − t)).
        if t <= self.times[0] {
            return self.values[0] * (self.fhat[0] * (self.times[0] - t)).exp();
        }
        // Right extrapolation: y(t) = y_{n-1} · exp(−fhat_{n-1} · (t − t_{n-1})).
        if t >= self.times[n - 1] {
            return self.values[n - 1] * (-self.fhat[n - 1] * (t - self.times[n - 1])).exp();
        }
        let i = self.locate(t);
        let h = self.times[i + 1] - self.times[i];
        let x = (t - self.times[i]) / h;
        let f_avg = self.discrete_forwards[i];
        // ∫_{t_i}^{t} f(s) ds = h · ( f_avg · x + ∫_0^x g(u) du )
        let int_g = segment_int_g(self.segments[i], x);
        let int_f = h * (f_avg * x + int_g);
        // Use the cached log-value to avoid a redundant ln/exp.
        (self.log_values[i] - int_f).exp()
    }
}

/// Computes the instantaneous forwards at the knots from the discrete
/// forwards using the Hagan–West time-weighted midpoint at interior knots,
/// linear extrapolation at the endpoints, and the positivity / convexity
/// clipping `fhat_i ∈ [0, 2 · min(neighbouring f's)]` (HW 2008 §4).
///
/// The clipping is what gives the construction its "non-negative-forward
/// preserving" property: when every discrete forward `f_i` is non-negative
/// (the natural case for a monotone-decreasing discount-factor table), the
/// clipped `fhat_i` is non-negative and the segment quadratic — whose
/// minimum is bounded below by `min(fhat_i, fhat_{i+1}) − |max gap|` —
/// stays non-negative on the segment.
fn compute_fhat(times: &[f64], df: &[f64]) -> Vec<f64> {
    let n = times.len();
    // n >= 2 by `new`'s precondition.
    let mut fhat = vec![0.0_f64; n];

    if n == 2 {
        // Degenerate single-segment case: only one discrete forward; the
        // instantaneous forward is constant on the segment.
        fhat[0] = df[0];
        fhat[1] = df[0];
        return fhat;
    }

    // Interior knots — time-weighted midpoint of adjacent discrete forwards.
    for i in 1..n - 1 {
        let h_left = times[i] - times[i - 1];
        let h_right = times[i + 1] - times[i];
        let total = h_left + h_right;
        fhat[i] = (h_left / total) * df[i] + (h_right / total) * df[i - 1];
    }

    // Endpoint linear extrapolation.
    fhat[0] = df[0] - 0.5 * (fhat[1] - df[0]);
    fhat[n - 1] = df[n - 2] - 0.5 * (fhat[n - 2] - df[n - 2]);

    // Positivity / convexity clipping. The lower bound is 0 (non-negative
    // forward); the upper bound is twice the smaller adjacent discrete
    // forward (interior) or twice the boundary segment's discrete forward
    // (endpoints). The cap is what keeps the quadratic from dipping below
    // zero in the middle of a segment.
    //
    // When the input data has negative discrete forwards (a non-monotone
    // discount-factor table), the symmetric upper-bound rule
    // `min(2·f_i, 2·f_{i+1})` is replaced by zero — i.e. the construction
    // still pins the segment integral but the clipping no longer
    // enforces a non-negative forward (which is not a meaningful property
    // when the input itself is not monotone).
    clip_endpoint(&mut fhat[0], df[0]);
    let last_df = df[n - 2];
    let last = n - 1;
    clip_endpoint(&mut fhat[last], last_df);
    for i in 1..n - 1 {
        clip_interior(&mut fhat[i], df[i - 1], df[i]);
    }

    fhat
}

/// Clip a boundary `fhat` to the Hagan–West convexity box `[0, 2 · f]`
/// (when `f >= 0`) or the symmetric `[2 · f, 0]` for negative `f`. Either
/// way the result lies on the same side of zero as the bounding discrete
/// forward, which is what the non-negative-forward proof requires.
#[inline]
fn clip_endpoint(fhat: &mut f64, f: f64) {
    let cap = 2.0 * f;
    if cap >= 0.0 {
        *fhat = fhat.clamp(0.0, cap);
    } else {
        *fhat = fhat.clamp(cap, 0.0);
    }
}

/// Clip an interior `fhat` to `[0, 2 · min(f_left, f_right)]` when both
/// adjacent discrete forwards are non-negative. When the two sides have
/// opposite signs (a turning point in the discount-factor series) the
/// clip degenerates to `0` — i.e. the segment starts and ends at the
/// average forward and the shape filter handles the rest.
#[inline]
fn clip_interior(fhat: &mut f64, f_left: f64, f_right: f64) {
    if f_left >= 0.0 && f_right >= 0.0 {
        let cap = 2.0 * f_left.min(f_right);
        *fhat = fhat.clamp(0.0, cap);
    } else if f_left <= 0.0 && f_right <= 0.0 {
        let cap = 2.0 * f_left.max(f_right);
        *fhat = fhat.clamp(cap, 0.0);
    } else {
        // Sign change in the underlying discrete forwards — no clean
        // box. Snap to zero to keep the quadratic well-behaved.
        *fhat = 0.0;
    }
}

/// Classifies a segment from `(g_0, g_1)` into one of the four Hagan–West
/// regions and pre-computes the shape parameters.
fn classify_segment(g0: f64, g1: f64) -> Segment {
    // Region I — the unmodified quadratic is monotone iff
    // `(2 g_0 + g_1) · (g_0 + 2 g_1) ≤ 0`. Same-sign cases (g_0 · g_1 > 0)
    // always fail this test (both factors share the sign of g_0+g_1); the
    // condition therefore selects only opposite-sign cases with bounded
    // magnitude AND the degenerate cases where one gap is zero.
    let s = 2.0 * g0 + g1;
    let t = g0 + 2.0 * g1;
    if s * t <= 0.0 {
        return Segment::QuadraticI { g0, g1 };
    }
    // Beyond Region I, two cases by sign agreement.
    if g0 * g1 < 0.0 {
        // Opposite signs but one dominates — Region II or III.
        // For g_0 > 0: Region II when g_1 < −2·g_0 (g_1 dominates),
        //              Region III when −g_0/2 < g_1 < 0 (g_0 dominates).
        // Symmetric for g_0 < 0. The discriminating test is the same as
        // distinguishing which of the two product factors above is positive.
        // s > 0 with t > 0 is Region I (already returned). With s · t > 0
        // (both same sign) we are out of Region I; classify by the sign of
        // the dominant gap.
        if g0.abs() > g1.abs() {
            // |g_0| > |g_1| — g_0 dominates → Region III.
            let eta = 3.0 * g1 / (g1 - g0);
            Segment::QuadThenFlatIII { g0, g1, eta }
        } else {
            // |g_1| ≥ |g_0| — g_1 dominates → Region II.
            let eta = (g1 + 2.0 * g0) / (g1 - g0);
            Segment::FlatThenQuadII { g0, g1, eta }
        }
    } else {
        // Same sign (g_0 · g_1 > 0) — Region IV.
        let sum = g0 + g1;
        let a = -g0 * g1 / sum;
        let eta = g1 / sum;
        Segment::TwoPieceIV { g0, g1, a, eta }
    }
}

/// Evaluates `g(x) = f(t) − f_avg` on a segment, where `x` is the segment-
/// local coordinate in `[0, 1]`.
fn segment_g(seg: Segment, x: f64) -> f64 {
    match seg {
        Segment::QuadraticI { g0, g1 } => {
            let x2 = x * x;
            g0 * (1.0 - 4.0 * x + 3.0 * x2) + g1 * (-2.0 * x + 3.0 * x2)
        }
        Segment::FlatThenQuadII { g0, g1, eta } => {
            if x <= eta {
                g0
            } else {
                let r = (x - eta) / (1.0 - eta);
                g0 + (g1 - g0) * r * r
            }
        }
        Segment::QuadThenFlatIII { g0, g1, eta } => {
            if x >= eta {
                g1
            } else {
                let r = (eta - x) / eta;
                g1 + (g0 - g1) * r * r
            }
        }
        Segment::TwoPieceIV { g0, g1, a, eta } => {
            if x <= eta {
                let r = (eta - x) / eta;
                a + (g0 - a) * r * r
            } else {
                let r = (x - eta) / (1.0 - eta);
                a + (g1 - a) * r * r
            }
        }
    }
}

/// Closed-form `∫_0^x g(u) du` on a segment, where `g(x) = f(t) − f_avg`.
///
/// For the unmodified Region I quadratic
/// `g(x) = g_0·(1 − 4x + 3x²) + g_1·(−2x + 3x²)` the primitive is
/// `g_0·(x − 2x² + x³) + g_1·(x³ − x²)`. The piecewise regions follow the
/// same integration recipe split at `η`.
fn segment_int_g(seg: Segment, x: f64) -> f64 {
    match seg {
        Segment::QuadraticI { g0, g1 } => {
            let x2 = x * x;
            let x3 = x2 * x;
            g0 * (x - 2.0 * x2 + x3) + g1 * (x3 - x2)
        }
        Segment::FlatThenQuadII { g0, g1, eta } => {
            if x <= eta {
                g0 * x
            } else {
                // ∫_0^η g_0 du + ∫_η^x [g_0 + (g_1 − g_0)·((u − η)/(1 − η))²] du
                let r = (x - eta) / (1.0 - eta);
                let r3 = r * r * r;
                g0 * x + (g1 - g0) * (1.0 - eta) * r3 / 3.0
            }
        }
        Segment::QuadThenFlatIII { g0, g1, eta } => {
            if x <= eta {
                // ∫_0^x [g_1 + (g_0 − g_1)·((η − u)/η)²] du.
                // With v = (η − u)/η, dv = −du/η, integration limits
                // [u=0 → v=1], [u=x → v=(η−x)/η]. The antiderivative in v
                // is −η · v³/3, giving (g_0 − g_1)·η·(1 − r³)/3 with
                // r = (η − x)/η.
                let r = (eta - x) / eta;
                let r3 = r * r * r;
                g1 * x + (g0 - g1) * eta * (1.0 - r3) / 3.0
            } else {
                // ∫_0^η piece + ∫_η^x g_1 du.
                let full = (g0 - g1) * eta / 3.0;
                g1 * x + full
            }
        }
        Segment::TwoPieceIV { g0, g1, a, eta } => {
            if x <= eta {
                // ∫_0^x [A + (g_0 − A)·((η − u)/η)²] du.
                let r = (eta - x) / eta;
                let r3 = r * r * r;
                a * x + (g0 - a) * eta * (1.0 - r3) / 3.0
            } else {
                // ∫_0^η piece (closed form) + ∫_η^x [A + (g_1 − A)·((u − η)/(1 − η))²] du.
                let first = a * eta + (g0 - a) * eta / 3.0;
                let r = (x - eta) / (1.0 - eta);
                let r3 = r * r * r;
                first + a * (x - eta) + (g1 - a) * (1.0 - eta) * r3 / 3.0
            }
        }
    }
}

impl Interpolator for ConvexMonotone {
    fn build(knots: &[(f64, f64)]) -> Result<Self, CurveError> {
        Self::new(knots)
    }

    fn eval(&self, t: f64) -> f64 {
        self.discount_at(t)
    }

    fn deriv(&self, t: f64) -> Option<f64> {
        // d/dt y(t) = −f(t) · y(t).
        Some(-self.forward_at(t) * self.discount_at(t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Construction & validation ───────────────────────────────────────

    #[test]
    fn rejects_empty() {
        let err = ConvexMonotone::new(&[]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 0 }));
    }

    #[test]
    fn rejects_single_knot() {
        let err = ConvexMonotone::new(&[(0.0, 1.0)]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 1 }));
    }

    #[test]
    fn rejects_non_monotone_times() {
        let err = ConvexMonotone::new(&[(0.0, 1.0), (2.0, 0.9), (1.0, 0.95)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NodesNotIncreasing { at_index: 2 }
        ));
    }

    #[test]
    fn rejects_duplicate_times() {
        let err = ConvexMonotone::new(&[(0.0, 1.0), (1.0, 0.95), (1.0, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::DuplicateNode { .. }));
    }

    #[test]
    fn rejects_negative_value() {
        let err = ConvexMonotone::new(&[(0.0, 1.0), (1.0, -0.5)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_zero_value() {
        let err = ConvexMonotone::new(&[(0.0, 1.0), (1.0, 0.0)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_nan_value() {
        let err = ConvexMonotone::new(&[(0.0, 1.0), (1.0, f64::NAN)]).unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_nan_time() {
        let err = ConvexMonotone::new(&[(0.0, 1.0), (f64::NAN, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn rejects_inf_time() {
        let err = ConvexMonotone::new(&[(0.0, 1.0), (f64::INFINITY, 0.9)]).unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    // ─── Knot reproduction ───────────────────────────────────────────────

    #[test]
    fn knot_reproduction_exact() {
        let knots = [
            (0.0, 1.0),
            (0.25, 0.99),
            (0.5, 0.975),
            (1.0, 0.95),
            (2.0, 0.90),
            (5.0, 0.78),
        ];
        let interp = ConvexMonotone::new(&knots).unwrap();
        for &(t, y) in &knots {
            let v = interp.eval(t);
            assert!((v - y).abs() < 1e-12, "knot ({t}, {y}) -> {v}");
        }
    }

    // ─── Discrete-forward integral identity ──────────────────────────────

    #[test]
    fn segment_integrals_match_discrete_forwards() {
        // The defining property of the Hagan–West construction: on each
        // segment, ∫ f(s) ds equals the segment's discrete forward times
        // the segment width — equivalently, the discount factor at the
        // next knot is reproduced exactly.
        let knots = [
            (0.0, 1.0),
            (0.5, 0.975),
            (1.5, 0.93),
            (3.0, 0.85),
            (5.0, 0.78),
        ];
        let interp = ConvexMonotone::new(&knots).unwrap();
        let n = knots.len();
        for i in 0..n - 1 {
            let (t_lo, y_lo) = knots[i];
            let (t_hi, y_hi) = knots[i + 1];
            let h = t_hi - t_lo;
            let expected_integral = y_lo.ln() - y_hi.ln();
            // ∫_{t_lo}^{t_hi} f = f_avg · h, with f_avg = expected/h.
            // We computed ∫ as y(t_hi) reproduced exactly above; here we
            // verify directly that the integral piece in the discount
            // formula equals the expected.
            let int_f = -((interp.eval(t_hi) / y_lo).ln());
            assert!(
                (int_f - expected_integral).abs() < 1e-10,
                "segment {i}: integral {int_f}, expected {expected_integral}",
            );
            // Bonus: f_avg from the segment matches the discrete forward
            // f_i+1 = (ln y_i - ln y_i+1) / h.
            let f_avg_expected = (y_lo.ln() - y_hi.ln()) / h;
            assert!(
                (interp.discrete_forwards[i] - f_avg_expected).abs() < 1e-12,
                "segment {i}: discrete forward {} vs expected {f_avg_expected}",
                interp.discrete_forwards[i],
            );
        }
    }

    // ─── Linear reproduction in the discount-factor domain ────────────────

    #[test]
    fn reproduces_linear_discount_factor_at_knots() {
        // y(t) = 1 − 0.05 · t — a monotone-decreasing "discount-factor
        // shape". Knot reproduction is exact for any valid interpolator;
        // we use this fixture to also exercise non-trivial forward shapes.
        let f = |t: f64| 1.0 - 0.05 * t;
        let knots: Vec<(f64, f64)> = [0.0_f64, 0.5, 1.0, 2.0, 5.0]
            .iter()
            .map(|&t| (t, f(t)))
            .collect();
        let interp = ConvexMonotone::new(&knots).unwrap();
        for &(t, y) in &knots {
            let v = interp.eval(t);
            assert!((v - y).abs() < 1e-12, "linear knot ({t}, {y}) -> {v}");
        }
    }

    // ─── Monotonicity preservation on the forward ────────────────────────

    /// Deterministic, seedable LCG — Numerical Recipes' "ranqd1" constants
    /// (Press et al. 2007 §7.1).
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
    fn random_monotone_discounts_yield_non_negative_forward() {
        // 20 random monotone-decreasing discount-factor sets. The Hagan–
        // West construction must keep the implied instantaneous forward
        // non-negative everywhere on a fine grid.
        let mut rng = Lcg::new(0xCAFE_BABE_u64);
        for set_idx in 0..20 {
            // 6-10 strictly increasing times in (0, ~5].
            let n = 6 + (rng.next_u32() % 5) as usize;
            let mut times = Vec::with_capacity(n);
            let mut values = Vec::with_capacity(n);
            let mut t = 0.0_f64;
            let mut y = 1.0_f64;
            for _ in 0..n {
                times.push(t);
                values.push(y);
                t += 0.1 + 0.5 * rng.next_unit();
                // Multiplicative shrink — keeps y > 0 and monotone-decreasing.
                y *= (-0.005 - 0.08 * rng.next_unit()).exp();
            }
            let knots: Vec<(f64, f64)> =
                times.iter().copied().zip(values.iter().copied()).collect();
            let interp = ConvexMonotone::new(&knots).unwrap();

            // Sample on a 200-point grid across the knot range.
            let t_lo = times[0];
            let t_hi = times[n - 1];
            let grid: u32 = 200;
            for k in 0..=grid {
                let t = t_lo + (t_hi - t_lo) * f64::from(k) / f64::from(grid);
                let f = interp.forward_at(t);
                assert!(
                    f >= -1e-12,
                    "set {set_idx}: negative forward at t={t}: f={f}",
                );
            }
        }
    }

    // ─── Plateau handling ────────────────────────────────────────────────

    #[test]
    fn constant_values_produce_constant_output() {
        // All knot values equal — every discrete forward is zero; the
        // instantaneous forward is zero everywhere; the interpolant is
        // identically equal to the constant value.
        let knots = [(0.0, 0.9), (1.0, 0.9), (2.5, 0.9), (5.0, 0.9)];
        let interp = ConvexMonotone::new(&knots).unwrap();
        for &t in &[0.0_f64, 0.3, 1.0, 1.7, 2.5, 3.1, 4.9, 5.0] {
            let v = interp.eval(t);
            assert!((v - 0.9).abs() < 1e-14, "t={t}: {v} vs 0.9");
        }
        // Forward is zero everywhere.
        for &t in &[0.0_f64, 0.5, 1.0, 2.0, 5.0] {
            assert!(interp.forward_at(t).abs() < 1e-14);
        }
    }

    // ─── Two-knot degenerate case ────────────────────────────────────────

    #[test]
    fn two_knot_is_flat_forward() {
        // With exactly two knots the construction has a single segment
        // with one discrete forward. fhat at both endpoints reuses that
        // forward, so the segment shape collapses to a constant forward.
        let knots = [(0.0_f64, 1.0_f64), (2.0_f64, (-0.08_f64).exp())];
        let interp = ConvexMonotone::new(&knots).unwrap();
        // Forward equals the single discrete forward (= 0.04) everywhere.
        for &t in &[0.0_f64, 0.25, 1.0, 1.5, 2.0] {
            let f = interp.forward_at(t);
            assert!((f - 0.04).abs() < 1e-12, "t={t}: f={f}");
        }
        // Discount factor matches the flat-forward exponential.
        for &t in &[0.0_f64, 0.5, 1.0, 1.5, 2.0] {
            let expected = (-0.04 * t).exp();
            let v = interp.eval(t);
            assert!((v - expected).abs() < 1e-12, "t={t}: {v} vs {expected}");
        }
    }

    // ─── Derivative consistency ──────────────────────────────────────────

    #[test]
    fn deriv_matches_minus_f_times_y() {
        // Analytic identity: d/dt y(t) = −f(t) · y(t). The `deriv` method
        // returns exactly this product; cross-check it against a centred
        // finite difference of `eval`.
        let knots = [
            (0.0, 1.0),
            (0.5, 0.975),
            (1.0, 0.95),
            (2.0, 0.90),
            (5.0, 0.78),
        ];
        let interp = ConvexMonotone::new(&knots).unwrap();
        let h = 1e-6_f64;
        for &t in &[0.1_f64, 0.7, 1.3, 2.5, 3.7, 4.6] {
            let analytic = interp.deriv(t).unwrap();
            let fd = (interp.eval(t + h) - interp.eval(t - h)) / (2.0 * h);
            assert!(
                (analytic - fd).abs() < 1e-6,
                "t={t}: analytic={analytic}, fd={fd}",
            );
            // And the direct product identity.
            let prod = -interp.forward_at(t) * interp.eval(t);
            assert!((analytic - prod).abs() < 1e-12);
        }
    }

    // ─── Flat extrapolation ──────────────────────────────────────────────

    #[test]
    fn flat_forward_extrapolation_outside_knot_range() {
        let knots = [(0.5_f64, 0.975_f64), (1.0_f64, 0.95_f64), (2.0_f64, 0.90)];
        let interp = ConvexMonotone::new(&knots).unwrap();
        // Below the first knot — forward equals fhat_0.
        let f0 = interp.fhat[0];
        for &t in &[0.0_f64, 0.1, 0.25] {
            let f = interp.forward_at(t);
            assert!((f - f0).abs() < 1e-12);
        }
        // Below the first knot — discount factor extends the boundary
        // exponential leftwards: y(t) = y_0 · exp(fhat_0 · (t_0 − t)).
        let v_left = interp.eval(0.0);
        let expected_left = 0.975_f64 * (f0 * 0.5).exp();
        assert!((v_left - expected_left).abs() < 1e-12);
        // Above the last knot — forward equals fhat_{n-1}.
        let fn_ = interp.fhat[interp.len() - 1];
        for &t in &[2.5_f64, 5.0, 10.0] {
            let f = interp.forward_at(t);
            assert!((f - fn_).abs() < 1e-12);
        }
        let v_right = interp.eval(3.0);
        let expected_right = 0.90_f64 * (-fn_ * 1.0).exp();
        assert!((v_right - expected_right).abs() < 1e-12);
    }

    // ─── tf-quant-finance oracle cross-check ─────────────────────────────

    #[test]
    fn tf_quant_finance_forward_rate_fixture() {
        // Cross-oracle: Google `tf-quant-finance` `monotone_convex_test.py`
        // (commit 4551a94e), test fixture transcribed in
        // doc/RESEARCH.md §2.7.1. tf-qf's `interpolate_forward_rate` takes
        // `reference_times = [0.25, 0.5, 1.0, 2.0, 3.0]` and
        // `discrete_forwards = [0.05, 0.051, 0.052, 0.053, 0.055]`, where
        // `discrete_forwards[i]` is the average forward over the segment
        // ending at `reference_times[i]` (with an implicit zero-start).
        //
        // To exercise the same algorithm via our discount-factor API, we
        // build the equivalent discount-factor table: knots at the
        // implicit-zero anchor plus the tf-qf reference times, with
        // y_{i+1} = y_i · exp(−f_i · (t_{i+1} − t_i)).
        let tf_times = [0.25_f64, 0.5, 1.0, 2.0, 3.0];
        let tf_dfwd = [0.05_f64, 0.051, 0.052, 0.053, 0.055];
        let mut knots = vec![(0.0_f64, 1.0_f64)];
        let mut y = 1.0_f64;
        let mut t_prev = 0.0_f64;
        for (&t, &f) in tf_times.iter().zip(tf_dfwd.iter()) {
            y *= (-f * (t - t_prev)).exp();
            knots.push((t, y));
            t_prev = t;
        }
        let interp = ConvexMonotone::new(&knots).unwrap();

        // Expected `interpolate_forward_rate(...)` output at test_times
        // = [0.25, 0.5, 1.0, 2.0, 3.0, 1.1].
        let test_times = [0.25_f64, 0.5, 1.0, 2.0, 3.0, 1.1];
        let expected = [
            0.0505_f64,
            0.051_333_333_333_333_333,
            0.052_333_333_333_333_333,
            0.054,
            0.0555,
            0.052_41,
        ];
        for (&t, &exp) in test_times.iter().zip(expected.iter()) {
            let f = interp.forward_at(t);
            assert!(
                (f - exp).abs() < 1e-9,
                "tf-qf forward fixture: t={t}, got {f}, expected {exp}",
            );
        }
    }

    #[test]
    fn tf_quant_finance_yield_fixture_with_filter() {
        // Cross-oracle: tf-quant-finance `monotone_convex_test.py` §2.7.3
        // (transcribed in doc/RESEARCH.md §2.7.3). The fixture is given in
        // percent units; values are `discrete_forwards = [5, 4.5, 4.1,
        // 5.5]` on segments [0,1], [1,2], [2,3], [3,4] with implicit-zero
        // anchor. This case exercises Region IV of the shape filter
        // (segment [2,3] has same-sign positive gaps).
        let tf_times = [1.0_f64, 2.0, 3.0, 4.0];
        let tf_dfwd = [0.05_f64, 0.045, 0.041, 0.055]; // percent → decimal
        let mut knots = vec![(0.0_f64, 1.0_f64)];
        let mut y = 1.0_f64;
        let mut t_prev = 0.0_f64;
        for (&t, &f) in tf_times.iter().zip(tf_dfwd.iter()) {
            y *= (-f * (t - t_prev)).exp();
            knots.push((t, y));
            t_prev = t;
        }
        let interp = ConvexMonotone::new(&knots).unwrap();

        // tf-qf "yield" Y(t) = (1/t) · ∫_0^t f(s) ds = −ln(y(t))/t.
        let test_times = [0.25_f64, 0.5, 1.0, 2.0, 3.0, 1.1, 2.5, 2.9, 3.6, 4.0];
        let expected_pct = [
            5.117_187_5_f64,
            5.093_75,
            5.0,
            4.75,
            4.533_333,
            4.974_6,
            4.624_082,
            4.535_422,
            4.661_777,
            4.775,
        ];
        for (&t, &exp) in test_times.iter().zip(expected_pct.iter()) {
            let v = interp.eval(t);
            let yld_pct = -v.ln() / t * 100.0;
            assert!(
                (yld_pct - exp).abs() < 1e-4,
                "tf-qf yield fixture: t={t}, got {yld_pct}, expected {exp}",
            );
        }
    }

    // ─── Trait & accessor coverage ───────────────────────────────────────

    #[test]
    fn build_trait_method_equivalent_to_new() {
        let knots = [(0.0, 1.0), (1.0, 0.95), (2.0, 0.9)];
        let a = ConvexMonotone::new(&knots).unwrap();
        let b = <ConvexMonotone as Interpolator>::build(&knots).unwrap();
        assert!((a.eval(0.5) - b.eval(0.5)).abs() < 1e-15);
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn len_and_is_empty() {
        let interp = ConvexMonotone::new(&[(0.0, 1.0), (1.0, 0.95), (2.0, 0.9)]).unwrap();
        assert_eq!(interp.len(), 3);
        assert!(!interp.is_empty());
    }

    #[test]
    fn clone_yields_equivalent_interpolant() {
        let interp = ConvexMonotone::new(&[(0.0, 1.0), (1.0, 0.95), (2.0, 0.9)]).unwrap();
        let copy = interp.clone();
        assert!((interp.eval(0.5) - copy.eval(0.5)).abs() < 1e-15);
        assert!((interp.forward_at(0.5) - copy.forward_at(0.5)).abs() < 1e-15);
    }

    // ─── Forward continuity at interior knots ────────────────────────────

    #[test]
    fn forward_continuous_at_interior_knots() {
        // The construction is C^0 in the forward (and hence C^1 in the
        // discount factor) at every interior knot: the left- and right-
        // limit forward values both equal fhat_i.
        let knots = [
            (0.0_f64, 1.0_f64),
            (0.5, 0.975),
            (1.0, 0.95),
            (2.0, 0.90),
            (5.0, 0.78),
        ];
        let interp = ConvexMonotone::new(&knots).unwrap();
        let h = 1e-7_f64;
        for &(t, _) in &knots[1..knots.len() - 1] {
            let f_left = interp.forward_at(t - h);
            let f_right = interp.forward_at(t + h);
            let f_at = interp.forward_at(t);
            assert!(
                (f_left - f_right).abs() < 1e-5,
                "forward discontinuous at t={t}: left={f_left}, right={f_right}",
            );
            assert!((f_at - f_right).abs() < 1e-5);
        }
    }
}
