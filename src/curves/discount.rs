// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Discount curve — the canonical yield-curve view.
//!
//! A [`DiscountCurve`] stores a tabulated discount-factor curve
//! `D : [0, infinity) -> (0, 1]` anchored at a reference date,
//! with `D(0) = 1`, evaluated by interpolating between user-supplied knots.
//!
//! ```text
//! D(t)  -> discount factor at year fraction t from the reference date
//! z(t)  -> -ln D(t) / t                    (zero rate)
//! f(t)  -> -d/dt ln D(t)                   (instantaneous forward rate)
//! L(t1, t2)  -> (D(t1) / D(t2) - 1) / tau  (simply-compounded forward)
//! r_par      -> (D(t_0) - D(t_N)) / sum_i tau_i D(t_i)
//!                                         (par swap rate, single-curve)
//! ```
//!
//! The four canonical derived quantities — zero rate, instantaneous forward,
//! simply-compounded forward, par swap rate — are computed from the stored
//! discount factor on demand. See [`crate::curves`] for the wider design
//! discussion (why `D` is canonical, the role of the three sibling view
//! types).
//!
//! # Knot validation
//!
//! [`DiscountCurve::new`] validates:
//!
//! - at least two nodes;
//! - the first node anchored at `(reference_date, 1.0)`;
//! - dates strictly increasing;
//! - every discount factor strictly positive and finite.
//!
//! The chosen [`Interpolation`] method then validates its own additional
//! invariants (e.g. positivity for [`Interpolation::LogLinear`]).
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §2.
//! - Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling*, Vol. 1,
//!   Atlantic Financial Press (2010), §6.

use crate::errors::CurveError;
use crate::interpolation::{Interpolation, InterpolationImpl};
use crate::types::{Compounding, Date, Daycount, Frequency, Tenor, TenorUnit};

/// A discount-factor curve — the canonical yield-curve view.
///
/// Stores the reference date, the curve's day-count convention, the knot
/// times (year fractions from the reference date) and discount factors, the
/// chosen [`Interpolation`] method, and a single concrete interpolator
/// instance built once at construction.
///
/// # Examples
///
/// ```
/// use regit_curves::curves::DiscountCurve;
/// use regit_curves::interpolation::Interpolation;
/// use regit_curves::types::{Date, Daycount};
///
/// let reference = Date::from_ymd(2024, 1, 2).unwrap();
/// let nodes = [
///     (reference, 1.0),
///     (Date::from_ymd(2025, 1, 2).unwrap(), 0.95),
///     (Date::from_ymd(2026, 1, 2).unwrap(), 0.90),
/// ];
/// let curve = DiscountCurve::new(
///     reference,
///     Daycount::Act365F,
///     &nodes,
///     Interpolation::LogLinear,
/// )
/// .unwrap();
/// // Knot reproduction.
/// assert!((curve.discount_at(nodes[1].0).unwrap() - 0.95).abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct DiscountCurve {
    /// Reference date — `D(reference_date) = 1` by construction.
    reference_date: Date,
    /// Day-count convention used to convert dates to year fractions on the
    /// curve's `t`-axis. Independent of any instrument's own day-count.
    daycount: Daycount,
    /// Knot times (year fractions from `reference_date`), strictly increasing,
    /// `times[0] = 0.0`.
    times: Vec<f64>,
    /// Knot discount factors, all `> 0`, `discounts[0] = 1.0`.
    discounts: Vec<f64>,
    /// Interpolation method.
    interpolation: Interpolation,
    /// Concrete interpolator instance, built once at construction.
    interpolator: InterpolationImpl,
}

impl DiscountCurve {
    /// Constructs a discount curve from `(date, discount-factor)` nodes.
    ///
    /// The first node must be the anchor `(reference_date, 1.0)`. Every
    /// subsequent node must have a strictly later date and a strictly
    /// positive discount factor.
    ///
    /// # Errors
    ///
    /// - [`CurveError::TooFewNodes`] if fewer than two nodes are supplied.
    /// - [`CurveError::AnchorNotUnit`] if the first node is not
    ///   `(reference_date, 1.0)` (within `f64::EPSILON` for the discount).
    /// - [`CurveError::NodesNotIncreasing`] if dates are not strictly
    ///   increasing.
    /// - [`CurveError::DuplicateNode`] if two consecutive dates coincide.
    /// - [`CurveError::NonPositiveDiscount`] if any discount is non-positive
    ///   or non-finite.
    /// - [`CurveError::Type`] if the day-count year-fraction query fails
    ///   (e.g. [`Daycount::Business252`] without a calendar).
    /// - Whatever [`CurveError`] the underlying interpolator's constructor
    ///   may return.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::DiscountCurve;
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Date, Daycount};
    /// use regit_curves::CurveError;
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// // Anchor missing -> rejected.
    /// let err = DiscountCurve::new(
    ///     reference,
    ///     Daycount::Act365F,
    ///     &[(Date::from_ymd(2025, 1, 2).unwrap(), 0.95)],
    ///     Interpolation::LogLinear,
    /// )
    /// .unwrap_err();
    /// assert!(matches!(err, CurveError::TooFewNodes { found: 1 }));
    /// ```
    pub fn new(
        reference_date: Date,
        daycount: Daycount,
        nodes: &[(Date, f64)],
        interpolation: Interpolation,
    ) -> Result<Self, CurveError> {
        if nodes.len() < 2 {
            return Err(CurveError::TooFewNodes { found: nodes.len() });
        }
        // Anchor: first node must be (reference_date, 1.0).
        let (first_date, first_d) = nodes[0];
        if first_date != reference_date || (first_d - 1.0).abs() > f64::EPSILON {
            return Err(CurveError::AnchorNotUnit);
        }
        let n = nodes.len();
        let mut times = Vec::with_capacity(n);
        let mut discounts = Vec::with_capacity(n);
        for (i, &(date, disc)) in nodes.iter().enumerate() {
            if !disc.is_finite() || disc <= 0.0 {
                return Err(CurveError::NonPositiveDiscount {
                    at_index: i,
                    value: disc,
                });
            }
            if i > 0 {
                let prev_date = nodes[i - 1].0;
                if date == prev_date {
                    let t = daycount.year_fraction(reference_date, date)?;
                    return Err(CurveError::DuplicateNode { t });
                }
                if date.serial() < prev_date.serial() {
                    return Err(CurveError::NodesNotIncreasing { at_index: i });
                }
            }
            let t = daycount.year_fraction(reference_date, date)?;
            times.push(t);
            discounts.push(disc);
        }
        let knots: Vec<(f64, f64)> = times
            .iter()
            .copied()
            .zip(discounts.iter().copied())
            .collect();
        let interpolator = interpolation.build(&knots)?;
        Ok(Self {
            reference_date,
            daycount,
            times,
            discounts,
            interpolation,
            interpolator,
        })
    }

    /// Constructs a discount curve directly from `(times, discounts)` slices.
    ///
    /// Useful when the caller already has year-fraction-form curve data
    /// (e.g. when reading from a tabulated source). The first time must be
    /// exactly `0.0`, the first discount factor exactly `1.0`.
    ///
    /// # Errors
    ///
    /// - [`CurveError::TooFewNodes`] if fewer than two knots are supplied.
    /// - [`CurveError::AnchorNotUnit`] if `times[0] != 0.0` or
    ///   `discounts[0] != 1.0`.
    /// - [`CurveError::TooFewNodes`] if `times.len() != discounts.len()`
    ///   (reported as the smaller of the two lengths).
    /// - [`CurveError::NodesNotIncreasing`] if times are not strictly
    ///   increasing.
    /// - [`CurveError::DuplicateNode`] if two consecutive times coincide.
    /// - [`CurveError::NonPositiveDiscount`] if any discount is non-positive.
    /// - [`CurveError::InvalidTime`] if any time is non-finite or negative.
    /// - Whatever [`CurveError`] the underlying interpolator's constructor
    ///   may return.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::DiscountCurve;
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let curve = DiscountCurve::from_times_and_discounts(
    ///     reference,
    ///     Daycount::Act365F,
    ///     &[0.0, 1.0, 2.0],
    ///     &[1.0, 0.95, 0.90],
    ///     Interpolation::LogLinear,
    /// )
    /// .unwrap();
    /// assert!((curve.discount(1.0).unwrap() - 0.95).abs() < 1e-14);
    /// ```
    pub fn from_times_and_discounts(
        reference_date: Date,
        daycount: Daycount,
        times: &[f64],
        discounts: &[f64],
        interpolation: Interpolation,
    ) -> Result<Self, CurveError> {
        if times.len() != discounts.len() {
            return Err(CurveError::TooFewNodes {
                found: times.len().min(discounts.len()),
            });
        }
        if times.len() < 2 {
            return Err(CurveError::TooFewNodes { found: times.len() });
        }
        // Anchor: (0.0, 1.0). Using exact equality on the anchor is the
        // documented contract; suppress `clippy::float_cmp` for this canonical
        // use case.
        #[allow(clippy::float_cmp)]
        let anchor_ok = times[0] == 0.0 && discounts[0] == 1.0;
        if !anchor_ok {
            return Err(CurveError::AnchorNotUnit);
        }
        for (i, (&t, &d)) in times.iter().zip(discounts.iter()).enumerate() {
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
                #[allow(clippy::float_cmp)]
                let duplicate = t == prev;
                if duplicate {
                    return Err(CurveError::DuplicateNode { t });
                }
                if t < prev {
                    return Err(CurveError::NodesNotIncreasing { at_index: i });
                }
            }
        }
        let knots: Vec<(f64, f64)> = times
            .iter()
            .copied()
            .zip(discounts.iter().copied())
            .collect();
        let interpolator = interpolation.build(&knots)?;
        Ok(Self {
            reference_date,
            daycount,
            times: times.to_vec(),
            discounts: discounts.to_vec(),
            interpolation,
            interpolator,
        })
    }

    /// Reference date — `D(reference_date) = 1`.
    #[must_use]
    #[inline]
    pub fn reference_date(&self) -> Date {
        self.reference_date
    }

    /// Day-count convention used to convert dates to year fractions on the
    /// curve's `t`-axis.
    #[must_use]
    #[inline]
    pub fn daycount(&self) -> Daycount {
        self.daycount
    }

    /// Interpolation method used by the curve.
    #[must_use]
    #[inline]
    pub fn interpolation(&self) -> Interpolation {
        self.interpolation
    }

    /// Knot times (year fractions from [`Self::reference_date`]).
    #[must_use]
    #[inline]
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Knot discount factors.
    #[must_use]
    #[inline]
    pub fn discounts(&self) -> &[f64] {
        &self.discounts
    }

    /// Discount factor `D(t)` at year fraction `t >= 0` from the reference
    /// date.
    ///
    /// Dispatches to the stored interpolator. Out-of-range `t` is handled by
    /// the interpolator's extrapolation rule (flat in the value domain for
    /// every variant; see the per-interpolator documentation for the precise
    /// behaviour).
    ///
    /// # Errors
    ///
    /// - [`CurveError::InvalidTime`] if `t` is negative or non-finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::DiscountCurve;
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let curve = DiscountCurve::from_times_and_discounts(
    ///     reference,
    ///     Daycount::Act365F,
    ///     &[0.0, 1.0, 2.0],
    ///     &[1.0, 0.95, 0.90],
    ///     Interpolation::LogLinear,
    /// )
    /// .unwrap();
    /// assert!((curve.discount(0.0).unwrap() - 1.0).abs() < 1e-15);
    /// ```
    pub fn discount(&self, t: f64) -> Result<f64, CurveError> {
        if !t.is_finite() || t < 0.0 {
            return Err(CurveError::InvalidTime { t });
        }
        Ok(self.interpolator.eval(t))
    }

    /// Discount factor at `date`, computed by translating `date` to a year
    /// fraction under the curve's day-count.
    ///
    /// # Errors
    ///
    /// - [`CurveError::Type`] if the day-count query fails (e.g. `date` is
    ///   before [`Self::reference_date`] for a day-count that requires a
    ///   non-negative range).
    /// - [`CurveError::InvalidTime`] if the resulting year fraction is
    ///   non-finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::DiscountCurve;
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let later = Date::from_ymd(2025, 1, 2).unwrap();
    /// let curve = DiscountCurve::new(
    ///     reference,
    ///     Daycount::Act365F,
    ///     &[(reference, 1.0), (later, 0.95)],
    ///     Interpolation::LogLinear,
    /// )
    /// .unwrap();
    /// assert!((curve.discount_at(later).unwrap() - 0.95).abs() < 1e-14);
    /// ```
    pub fn discount_at(&self, date: Date) -> Result<f64, CurveError> {
        let t = self.daycount.year_fraction(self.reference_date, date)?;
        self.discount(t)
    }

    /// Zero rate at year fraction `t` under the supplied `compounding`
    /// convention.
    ///
    /// Computed by inverting `D(t)` via
    /// [`Compounding::rate_from_discount`]. At `t = 0` (the anchor) we return
    /// `0.0` — the canonical "rate at the anchor" value, since `D(0) = 1`
    /// makes any rate consistent.
    ///
    /// # Errors
    ///
    /// - [`CurveError::InvalidTime`] if `t` is negative or non-finite.
    /// - [`CurveError::Type`] if the compounding inverse fails (e.g. invalid
    ///   `periods_per_year` for [`Compounding::Periodic`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::DiscountCurve;
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Compounding, Date, Daycount};
    ///
    /// // Flat continuous curve at r = 0.04.
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let curve = DiscountCurve::from_times_and_discounts(
    ///     reference,
    ///     Daycount::Act365F,
    ///     &[0.0, 1.0, 2.0],
    ///     &[1.0, (-0.04_f64).exp(), (-0.08_f64).exp()],
    ///     Interpolation::LogLinear,
    /// )
    /// .unwrap();
    /// let z = curve.zero_rate(1.0, Compounding::Continuous).unwrap();
    /// assert!((z - 0.04).abs() < 1e-12);
    /// ```
    pub fn zero_rate(&self, t: f64, compounding: Compounding) -> Result<f64, CurveError> {
        if !t.is_finite() || t < 0.0 {
            return Err(CurveError::InvalidTime { t });
        }
        // At the anchor the rate is a 0/0 limit. We return 0.0 by convention
        // — every yield-curve consumer expects "the rate at t = 0" to be a
        // benign value, and 0 is the canonical "no information yet" choice
        // (Hagan & West 2006 §3.1 footnote).
        if t == 0.0 {
            return Ok(0.0);
        }
        let d = self.interpolator.eval(t);
        let r = compounding.rate_from_discount(d, t)?;
        Ok(r)
    }

    /// Instantaneous forward rate `f(t) = -d/dt ln D(t)`.
    ///
    /// Where the interpolant is C¹, the analytic derivative
    /// `-D'(t) / D(t)` is used. Where the interpolant is not C¹ (e.g.
    /// [`Interpolation::PiecewiseConstantForward`] at a knot), a two-sided
    /// finite-difference fallback with step `h = max(1e-7, t * 1e-7)` is
    /// taken from the **right** (forward into the next segment), since
    /// curves are conventionally right-continuous at a pillar. Outside the
    /// knot range the forward is `0` whenever the interpolant
    /// flat-extrapolates.
    ///
    /// # Errors
    ///
    /// - [`CurveError::InvalidTime`] if `t` is negative or non-finite.
    /// - [`CurveError::NonPositiveDiscount`] if the curve evaluates to a
    ///   non-positive discount at `t` (should never happen for a
    ///   well-formed curve).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::DiscountCurve;
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// // Flat continuous curve at r = 0.04. Instantaneous forward at every
    /// // interior t is exactly 0.04.
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let curve = DiscountCurve::from_times_and_discounts(
    ///     reference,
    ///     Daycount::Act365F,
    ///     &[0.0, 1.0, 2.0, 5.0],
    ///     &[
    ///         1.0,
    ///         (-0.04_f64).exp(),
    ///         (-0.08_f64).exp(),
    ///         (-0.20_f64).exp(),
    ///     ],
    ///     Interpolation::LogLinear,
    /// )
    /// .unwrap();
    /// let f = curve.instantaneous_forward(1.5).unwrap();
    /// assert!((f - 0.04).abs() < 1e-10);
    /// ```
    pub fn instantaneous_forward(&self, t: f64) -> Result<f64, CurveError> {
        if !t.is_finite() || t < 0.0 {
            return Err(CurveError::InvalidTime { t });
        }
        let d = self.interpolator.eval(t);
        if d <= 0.0 || !d.is_finite() {
            return Err(CurveError::NonPositiveDiscount {
                at_index: 0,
                value: d,
            });
        }
        // Prefer the analytic derivative when available. `f = -D'(t)/D(t)`.
        if let Some(dprime) = self.interpolator.deriv(t) {
            if dprime.is_finite() {
                return Ok(-dprime / d);
            }
        }
        // Finite-difference fallback (right-sided). Use a relative step that
        // never collapses below 1e-7 to keep round-off small.
        let h = 1e-7_f64.max(t.abs() * 1e-7);
        let d_right = self.interpolator.eval(t + h);
        if d_right <= 0.0 || !d_right.is_finite() {
            return Err(CurveError::NonPositiveDiscount {
                at_index: 0,
                value: d_right,
            });
        }
        // -d/dt ln D = -(ln D(t+h) - ln D(t)) / h.
        Ok(-(d_right.ln() - d.ln()) / h)
    }

    /// Simply-compounded forward rate over `[t1, t2]` with `t2 > t1 >= 0`:
    /// `L = (D(t1) / D(t2) - 1) / tau`.
    ///
    /// **Simplification**: the year fraction `tau` used in the denominator
    /// is taken as `t2 - t1` — i.e. the year fraction implicit in the
    /// curve's own day-count. The `daycount` argument is accepted for
    /// forward compatibility (it will become load-bearing if a future
    /// release adds an explicit `date -> tau` conversion against a different
    /// day-count) but is **not** currently consulted: `t1` and `t2` are
    /// already year fractions in the curve's day-count.
    ///
    /// # Errors
    ///
    /// - [`CurveError::InvalidTime`] if `t1` or `t2` is negative or
    ///   non-finite, or if `t2 <= t1`.
    /// - [`CurveError::NonPositiveDiscount`] if either discount evaluates
    ///   non-positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::DiscountCurve;
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let curve = DiscountCurve::from_times_and_discounts(
    ///     reference,
    ///     Daycount::Act365F,
    ///     &[0.0, 1.0, 2.0],
    ///     &[1.0, (-0.04_f64).exp(), (-0.08_f64).exp()],
    ///     Interpolation::LogLinear,
    /// )
    /// .unwrap();
    /// let l = curve.forward_rate(1.0, 2.0, Daycount::Act365F).unwrap();
    /// // (exp(0.04) - 1) / 1.0 over a flat-r=0.04 curve.
    /// assert!((l - 0.04_f64.exp_m1()).abs() < 1e-12);
    /// ```
    pub fn forward_rate(&self, t1: f64, t2: f64, _daycount: Daycount) -> Result<f64, CurveError> {
        if !t1.is_finite() || t1 < 0.0 {
            return Err(CurveError::InvalidTime { t: t1 });
        }
        if !t2.is_finite() || t2 <= t1 {
            return Err(CurveError::InvalidTime { t: t2 });
        }
        let d1 = self.interpolator.eval(t1);
        let d2 = self.interpolator.eval(t2);
        if d1 <= 0.0 || !d1.is_finite() {
            return Err(CurveError::NonPositiveDiscount {
                at_index: 0,
                value: d1,
            });
        }
        if d2 <= 0.0 || !d2.is_finite() {
            return Err(CurveError::NonPositiveDiscount {
                at_index: 0,
                value: d2,
            });
        }
        let tau = t2 - t1;
        Ok((d1 / d2 - 1.0) / tau)
    }

    /// Par swap rate `r = (D(t_0) - D(t_N)) / sum_i tau_i D(t_i)` for a
    /// regular swap starting at `start`, maturing at `maturity`, paying
    /// with frequency `freq`, and accruing under `daycount`.
    ///
    /// **Single-curve formula**: this curve both discounts and projects,
    /// which collapses the float-leg PV to `D(t_0) - D(t_N)` (a telescoping
    /// sum). See Hagan & West (2006), §2.3.
    ///
    /// The schedule is built **inline** from `(start, maturity, freq)` by
    /// laying down period boundaries at `12 / freq.periods_per_year()`-month
    /// intervals from `start`, identical to [`crate::instruments::SwapSchedule::from_regular`]
    /// but returning [`CurveError`] directly. We chose inlining over a
    /// `SwapSchedule` call to keep [`DiscountCurve`] decoupled from the
    /// instruments layer's [`crate::errors::BootstrapError`] type — the
    /// crate's dependency graph (WORKING.md §2) keeps `curves` at level 6
    /// and the bootstrap engine at level 7.
    ///
    /// # Errors
    ///
    /// - [`CurveError::InvalidTime`] if `start >= maturity`, the schedule
    ///   overflows, or the schedule is not regular at the requested
    ///   frequency.
    /// - [`CurveError::Type`] if the accrual day-count query fails.
    /// - [`CurveError::NonPositiveDiscount`] if the curve discount factor
    ///   at any pillar evaluates non-positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::DiscountCurve;
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Date, Daycount, Frequency};
    ///
    /// // Flat 4% continuous curve, quarterly grid for 5 years.
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let r_c = 0.04_f64;
    /// let mut times = Vec::new();
    /// let mut discs = Vec::new();
    /// for i in 0..=20 {
    ///     let t = f64::from(i) * 0.25;
    ///     times.push(t);
    ///     discs.push((-r_c * t).exp());
    /// }
    /// let curve = DiscountCurve::from_times_and_discounts(
    ///     reference,
    ///     Daycount::Act365F,
    ///     &times,
    ///     &discs,
    ///     Interpolation::LogLinear,
    /// )
    /// .unwrap();
    /// let par = curve
    ///     .par_swap_rate(
    ///         reference,
    ///         Date::from_ymd(2026, 1, 2).unwrap(),
    ///         Frequency::SemiAnnual,
    ///         Daycount::Act365F,
    ///     )
    ///     .unwrap();
    /// // For a flat-r curve the par rate is positive and broadly close to r.
    /// assert!(par > 0.0);
    /// ```
    pub fn par_swap_rate(
        &self,
        start: Date,
        maturity: Date,
        freq: Frequency,
        daycount: Daycount,
    ) -> Result<f64, CurveError> {
        if start.serial() >= maturity.serial() {
            return Err(CurveError::InvalidTime {
                t: f64::from(start.days_between(maturity)),
            });
        }
        // Lay down the regular schedule inline. Mirrors
        // `SwapSchedule::from_regular` but returns `CurveError` directly so
        // we don't pull the bootstrap-layer error type up into `curves`.
        let n = freq.periods_per_year();
        let schedule: Vec<Date> = if matches!(freq, Frequency::OnceAtMaturity) {
            vec![start, maturity]
        } else {
            if n == 0 {
                return Err(CurveError::InvalidTime { t: 0.0 });
            }
            let months_per_period = i32::try_from(12 / n).unwrap_or(1);
            let mut dates = vec![start];
            let mut k: i32 = 1;
            loop {
                let total_months = months_per_period
                    .checked_mul(k)
                    .ok_or(CurveError::InvalidTime { t: f64::from(k) })?;
                let next = Tenor::new(total_months, TenorUnit::Months).add_to(start);
                if next.serial() > maturity.serial() {
                    // Schedule is not regular at the requested frequency.
                    return Err(CurveError::InvalidTime {
                        t: f64::from(next.days_between(maturity)),
                    });
                }
                dates.push(next);
                if next.serial() == maturity.serial() {
                    break;
                }
                k = k
                    .checked_add(1)
                    .ok_or(CurveError::InvalidTime { t: f64::from(k) })?;
                if k > 10_000 {
                    return Err(CurveError::InvalidTime { t: f64::from(k) });
                }
            }
            dates
        };
        // Numerator: D(t_0) - D(t_N).
        let t_start = self.daycount.year_fraction(self.reference_date, start)?;
        let t_end = self.daycount.year_fraction(self.reference_date, maturity)?;
        let d_start = self.discount(t_start)?;
        let d_end = self.discount(t_end)?;
        // Denominator (annuity): sum_i tau_i D(t_i) over the leg payment dates.
        let mut annuity = 0.0_f64;
        for i in 0..(schedule.len() - 1) {
            let period_start = schedule[i];
            let period_end = schedule[i + 1];
            let tau = daycount.year_fraction(period_start, period_end)?;
            let t_pay = self
                .daycount
                .year_fraction(self.reference_date, period_end)?;
            let d_pay = self.discount(t_pay)?;
            if d_pay <= 0.0 {
                return Err(CurveError::NonPositiveDiscount {
                    at_index: i,
                    value: d_pay,
                });
            }
            annuity += tau * d_pay;
        }
        if annuity <= 0.0 || !annuity.is_finite() {
            return Err(CurveError::NonPositiveDiscount {
                at_index: 0,
                value: annuity,
            });
        }
        Ok((d_start - d_end) / annuity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolation::SplineBoundary;

    fn d(y: i32, m: u32, day: u32) -> Date {
        Date::from_ymd(y, m, day).unwrap()
    }

    fn reference_date() -> Date {
        d(2024, 1, 2)
    }

    /// Build a tabulated flat continuous-r curve at quarterly resolution out
    /// to 30 years, expressed in year fractions and discount factors.
    fn flat_curve_tables(reference: Date, daycount: Daycount, r_c: f64) -> (Vec<f64>, Vec<f64>) {
        let mut times = Vec::new();
        let mut discs = Vec::new();
        for i in 0..=120 {
            // Quarterly grid.
            let date = Date::from_serial(reference.serial() + i * 91);
            let t = daycount.year_fraction(reference, date).unwrap();
            times.push(t);
            discs.push((-r_c * t).exp());
        }
        (times, discs)
    }

    // ─── Construction & validation ───────────────────────────────────────

    #[test]
    fn new_accepts_minimal_two_node_curve() {
        let curve = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap();
        assert_eq!(curve.reference_date(), reference_date());
        assert_eq!(curve.times().len(), 2);
        assert_eq!(curve.discounts().len(), 2);
    }

    #[test]
    fn new_rejects_too_few_nodes() {
        let err = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0)],
            Interpolation::LogLinear,
        )
        .unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 1 }));
    }

    #[test]
    fn new_rejects_missing_anchor() {
        // First date != reference_date.
        let err = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(d(2024, 2, 2), 1.0), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap_err();
        assert!(matches!(err, CurveError::AnchorNotUnit));
    }

    #[test]
    fn new_rejects_anchor_not_unit() {
        // First date == reference but D != 1.
        let err = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 0.99), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap_err();
        assert!(matches!(err, CurveError::AnchorNotUnit));
    }

    #[test]
    fn new_rejects_non_increasing_dates() {
        let err = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[
                (reference_date(), 1.0),
                (d(2026, 1, 2), 0.90),
                (d(2025, 1, 2), 0.95),
            ],
            Interpolation::LogLinear,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CurveError::NodesNotIncreasing { at_index: 2 }
        ));
    }

    #[test]
    fn new_rejects_duplicate_dates() {
        let err = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[
                (reference_date(), 1.0),
                (d(2025, 1, 2), 0.95),
                (d(2025, 1, 2), 0.94),
            ],
            Interpolation::LogLinear,
        )
        .unwrap_err();
        assert!(matches!(err, CurveError::DuplicateNode { .. }));
    }

    #[test]
    fn new_rejects_non_positive_discount() {
        let err = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), -0.5)],
            Interpolation::LogLinear,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn new_rejects_nan_discount() {
        let err = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), f64::NAN)],
            Interpolation::LogLinear,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CurveError::NonPositiveDiscount { at_index: 1, .. }
        ));
    }

    #[test]
    fn from_times_and_discounts_rejects_anchor_not_unit() {
        let err = DiscountCurve::from_times_and_discounts(
            reference_date(),
            Daycount::Act365F,
            &[0.5, 1.0],
            &[1.0, 0.95],
            Interpolation::LogLinear,
        )
        .unwrap_err();
        assert!(matches!(err, CurveError::AnchorNotUnit));
        let err = DiscountCurve::from_times_and_discounts(
            reference_date(),
            Daycount::Act365F,
            &[0.0, 1.0],
            &[0.99, 0.95],
            Interpolation::LogLinear,
        )
        .unwrap_err();
        assert!(matches!(err, CurveError::AnchorNotUnit));
    }

    #[test]
    fn from_times_and_discounts_rejects_mismatched_lengths() {
        let err = DiscountCurve::from_times_and_discounts(
            reference_date(),
            Daycount::Act365F,
            &[0.0, 1.0, 2.0],
            &[1.0, 0.95],
            Interpolation::LogLinear,
        )
        .unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { .. }));
    }

    #[test]
    fn from_times_and_discounts_rejects_nan_time() {
        let err = DiscountCurve::from_times_and_discounts(
            reference_date(),
            Daycount::Act365F,
            &[0.0, f64::NAN],
            &[1.0, 0.95],
            Interpolation::LogLinear,
        )
        .unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    // ─── Knot reproduction ───────────────────────────────────────────────

    #[test]
    fn discount_knot_reproduction_log_linear() {
        let (times, discs) = flat_curve_tables(reference_date(), Daycount::Act365F, 0.04);
        let curve = DiscountCurve::from_times_and_discounts(
            reference_date(),
            Daycount::Act365F,
            &times,
            &discs,
            Interpolation::LogLinear,
        )
        .unwrap();
        for (&t, &dd) in times.iter().zip(discs.iter()) {
            let v = curve.discount(t).unwrap();
            assert!((v - dd).abs() < 1e-14, "knot ({t}, {dd}) -> {v}");
        }
    }

    #[test]
    fn discount_knot_reproduction_all_methods() {
        let (times, discs) = flat_curve_tables(reference_date(), Daycount::Act365F, 0.03);
        for method in [
            Interpolation::Linear,
            Interpolation::LogLinear,
            Interpolation::LinearInZero,
            Interpolation::PiecewiseConstantForward,
            Interpolation::CubicSpline(SplineBoundary::NotAKnot),
            Interpolation::CubicSpline(SplineBoundary::Natural),
            Interpolation::HermiteBessel,
            Interpolation::MonotoneCubic,
            Interpolation::MonotoneHyman,
            Interpolation::MonotoneSteffen,
        ] {
            let curve = DiscountCurve::from_times_and_discounts(
                reference_date(),
                Daycount::Act365F,
                &times,
                &discs,
                method,
            )
            .unwrap();
            for (&t, &dd) in times.iter().zip(discs.iter()) {
                let v = curve.discount(t).unwrap();
                assert!(
                    (v - dd).abs() < 1e-12,
                    "method {method:?}: knot ({t}, {dd}) -> {v}"
                );
            }
        }
    }

    // ─── Anchor ──────────────────────────────────────────────────────────

    #[test]
    fn discount_anchor_is_unit_to_round_off() {
        let curve = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap();
        assert!((curve.discount(0.0).unwrap() - 1.0).abs() < 1e-15);
        assert!((curve.discount_at(reference_date()).unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn discount_rejects_negative_time() {
        let curve = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap();
        assert!(matches!(
            curve.discount(-1.0).unwrap_err(),
            CurveError::InvalidTime { .. }
        ));
    }

    #[test]
    fn discount_rejects_non_finite_time() {
        let curve = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap();
        assert!(matches!(
            curve.discount(f64::NAN).unwrap_err(),
            CurveError::InvalidTime { .. }
        ));
    }

    // ─── Zero rate ───────────────────────────────────────────────────────

    #[test]
    fn zero_rate_flat_curve_continuous() {
        let r_c = 0.04;
        let (times, discs) = flat_curve_tables(reference_date(), Daycount::Act365F, r_c);
        let curve = DiscountCurve::from_times_and_discounts(
            reference_date(),
            Daycount::Act365F,
            &times,
            &discs,
            Interpolation::LogLinear,
        )
        .unwrap();
        for t in [0.25, 1.0, 5.0, 10.0, 25.0] {
            let z = curve.zero_rate(t, Compounding::Continuous).unwrap();
            assert!(
                (z - r_c).abs() < 1e-12,
                "zero rate at t={t}: got {z}, expected {r_c}"
            );
        }
    }

    #[test]
    fn zero_rate_at_anchor_is_zero() {
        let curve = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap();
        let z = curve.zero_rate(0.0, Compounding::Continuous).unwrap();
        assert!((z - 0.0).abs() < 1e-15);
    }

    #[test]
    fn zero_rate_each_compounding_variant() {
        // Flat continuous curve at r_c = 0.05. At t = 1y the discount is
        // exp(-0.05) regardless of compounding choice — only the implied
        // rate under that compounding changes.
        let r_c = 0.05_f64;
        let curve = DiscountCurve::from_times_and_discounts(
            reference_date(),
            Daycount::Act365F,
            &[0.0, 1.0, 2.0],
            &[1.0, (-r_c).exp(), (-2.0 * r_c).exp()],
            Interpolation::LogLinear,
        )
        .unwrap();
        let d_1y = (-r_c).exp();
        // Continuous: r = -ln(D)/t = r_c.
        let z_c = curve.zero_rate(1.0, Compounding::Continuous).unwrap();
        assert!((z_c - r_c).abs() < 1e-12);
        // Simple: r = (1/D - 1) / t.
        let z_s = curve.zero_rate(1.0, Compounding::Simple).unwrap();
        assert!((z_s - (1.0 / d_1y - 1.0)).abs() < 1e-12);
        // Periodic n=2: r = n * (D^(-1/(n*t)) - 1).
        let z_p = curve
            .zero_rate(
                1.0,
                Compounding::Periodic {
                    periods_per_year: 2,
                },
            )
            .unwrap();
        let expected = 2.0 * (d_1y.powf(-0.5) - 1.0);
        assert!((z_p - expected).abs() < 1e-12);
    }

    #[test]
    fn zero_rate_rejects_negative_time() {
        let curve = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap();
        assert!(matches!(
            curve.zero_rate(-1.0, Compounding::Continuous).unwrap_err(),
            CurveError::InvalidTime { .. }
        ));
    }

    // ─── Instantaneous forward ───────────────────────────────────────────

    #[test]
    fn instantaneous_forward_flat_curve_loglinear() {
        let r_c = 0.04_f64;
        let (times, discs) = flat_curve_tables(reference_date(), Daycount::Act365F, r_c);
        let curve = DiscountCurve::from_times_and_discounts(
            reference_date(),
            Daycount::Act365F,
            &times,
            &discs,
            Interpolation::LogLinear,
        )
        .unwrap();
        // LogLinear has analytic right-derivative everywhere; flat curve ->
        // every segment forward equals r_c.
        for t in [0.1, 0.5, 1.0, 2.0, 5.0] {
            let f = curve.instantaneous_forward(t).unwrap();
            assert!(
                (f - r_c).abs() < 1e-10,
                "instantaneous fwd at t={t}: got {f}, expected {r_c}"
            );
        }
    }

    #[test]
    fn instantaneous_forward_flat_curve_smooth_methods() {
        // Smooth interpolators (cubic spline, monotone) should also recover
        // the constant forward to high precision on a flat curve.
        let r_c = 0.04_f64;
        let (times, discs) = flat_curve_tables(reference_date(), Daycount::Act365F, r_c);
        for method in [
            Interpolation::CubicSpline(SplineBoundary::NotAKnot),
            Interpolation::MonotoneCubic,
            Interpolation::HermiteBessel,
        ] {
            let curve = DiscountCurve::from_times_and_discounts(
                reference_date(),
                Daycount::Act365F,
                &times,
                &discs,
                method,
            )
            .unwrap();
            // Pick t at a quarter-grid mid-segment.
            let f = curve.instantaneous_forward(1.5).unwrap();
            assert!(
                (f - r_c).abs() < 5e-3,
                "method {method:?}: fwd at 1.5y -> {f}"
            );
        }
    }

    #[test]
    fn instantaneous_forward_rejects_negative_time() {
        let curve = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap();
        assert!(matches!(
            curve.instantaneous_forward(-0.5).unwrap_err(),
            CurveError::InvalidTime { .. }
        ));
    }

    // ─── Simply-compounded forward ───────────────────────────────────────

    #[test]
    fn forward_rate_flat_curve_closed_form() {
        let r_c = 0.04_f64;
        let (times, discs) = flat_curve_tables(reference_date(), Daycount::Act365F, r_c);
        let curve = DiscountCurve::from_times_and_discounts(
            reference_date(),
            Daycount::Act365F,
            &times,
            &discs,
            Interpolation::LogLinear,
        )
        .unwrap();
        // On a flat continuous curve, the simply-compounded forward over
        // [t1, t2] is (exp(r_c * (t2 - t1)) - 1) / (t2 - t1).
        let cases = [(1.0_f64, 2.0_f64), (0.5, 3.0), (2.0, 5.0)];
        for (t1, t2) in cases {
            let l = curve.forward_rate(t1, t2, Daycount::Act365F).unwrap();
            let expected = (r_c * (t2 - t1)).exp_m1() / (t2 - t1);
            assert!(
                (l - expected).abs() < 1e-12,
                "fwd[{t1},{t2}] -> {l}, expected {expected}"
            );
        }
    }

    #[test]
    fn forward_rate_rejects_t2_le_t1() {
        let curve = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap();
        assert!(matches!(
            curve.forward_rate(2.0, 1.0, Daycount::Act365F).unwrap_err(),
            CurveError::InvalidTime { .. }
        ));
        assert!(matches!(
            curve.forward_rate(2.0, 2.0, Daycount::Act365F).unwrap_err(),
            CurveError::InvalidTime { .. }
        ));
    }

    // ─── Par swap rate ───────────────────────────────────────────────────

    #[test]
    fn par_swap_rate_flat_curve_2y_semi_annual() {
        // Flat continuous curve at r_c = 0.04 -> the implied 2y semi-annual
        // par rate has a closed form. With Act/365F and 2y exact (730 days),
        // the four pillars at 0.5, 1.0, 1.5, 2.0 of D = exp(-r_c * t) give:
        //   numerator   = D(0) - D(2)
        //   annuity_sa  = SUM_{i=1..4} tau_i exp(-r_c * (i / 2))
        // where each tau_i is the 6m Act/365F year fraction (computed from
        // the actual day count of the period).
        let r_c = 0.04_f64;
        let (times, discs) = flat_curve_tables(reference_date(), Daycount::Act365F, r_c);
        let curve = DiscountCurve::from_times_and_discounts(
            reference_date(),
            Daycount::Act365F,
            &times,
            &discs,
            Interpolation::LogLinear,
        )
        .unwrap();
        let par = curve
            .par_swap_rate(
                reference_date(),
                d(2026, 1, 2),
                Frequency::SemiAnnual,
                Daycount::Act365F,
            )
            .unwrap();
        // Verify via direct schedule reconstruction.
        let mut annuity = 0.0_f64;
        let pillars = [d(2024, 7, 2), d(2025, 1, 2), d(2025, 7, 2), d(2026, 1, 2)];
        let mut prev = reference_date();
        for date in pillars {
            let tau = Daycount::Act365F.year_fraction(prev, date).unwrap();
            let t = Daycount::Act365F
                .year_fraction(reference_date(), date)
                .unwrap();
            annuity += tau * (-r_c * t).exp();
            prev = date;
        }
        let t_end = Daycount::Act365F
            .year_fraction(reference_date(), d(2026, 1, 2))
            .unwrap();
        let expected = (1.0 - (-r_c * t_end).exp()) / annuity;
        assert!(
            (par - expected).abs() < 1e-12,
            "par={par}, expected={expected}"
        );
    }

    #[test]
    fn par_swap_rate_rejects_start_ge_maturity() {
        let curve = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap();
        let err = curve
            .par_swap_rate(
                d(2025, 1, 2),
                reference_date(),
                Frequency::SemiAnnual,
                Daycount::Act365F,
            )
            .unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn par_swap_rate_rejects_irregular_schedule() {
        let curve = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap();
        // 13 months at semi-annual cadence is not regular.
        let err = curve
            .par_swap_rate(
                d(2024, 1, 2),
                d(2025, 2, 2),
                Frequency::SemiAnnual,
                Daycount::Act365F,
            )
            .unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    // ─── Cross-check against CurveSnapshot ───────────────────────────────

    #[test]
    fn discount_agrees_with_curve_snapshot() {
        // The bootstrap-internal CurveSnapshot uses log-linear-on-D
        // interpolation through `(times, discounts)`. A DiscountCurve built
        // with `Interpolation::LogLinear` on the same data must produce
        // identical discount factors at every knot AND at non-knot points.
        use crate::instruments::CurveSnapshot;
        let reference = reference_date();
        let dc = Daycount::Act365F;
        let r_c = 0.04_f64;
        let (times, discs) = flat_curve_tables(reference, dc, r_c);
        let curve = DiscountCurve::from_times_and_discounts(
            reference,
            dc,
            &times,
            &discs,
            Interpolation::LogLinear,
        )
        .unwrap();
        let snap = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discs,
        };
        // Every knot.
        for &t in &times {
            let a = curve.discount(t).unwrap();
            let b = snap.discount_at(t).unwrap();
            assert!((a - b).abs() < 1e-14, "knot t={t}: {a} vs {b}");
        }
        // Ten interior probe points (deterministic, evenly spread).
        for i in 1..=10 {
            let t = (f64::from(i) / 11.0) * 25.0;
            let a = curve.discount(t).unwrap();
            let b = snap.discount_at(t).unwrap();
            assert!((a - b).abs() < 1e-14, "probe t={t}: {a} vs {b}");
        }
    }

    // ─── Accessors ───────────────────────────────────────────────────────

    #[test]
    fn accessors_round_trip_constructor_inputs() {
        let curve = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap();
        assert_eq!(curve.reference_date(), reference_date());
        assert_eq!(curve.daycount(), Daycount::Act365F);
        assert_eq!(curve.interpolation(), Interpolation::LogLinear);
        assert_eq!(curve.times().len(), 2);
        assert_eq!(curve.discounts().len(), 2);
    }

    #[test]
    fn clone_yields_equivalent_curve() {
        let curve = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), 0.95)],
            Interpolation::LogLinear,
        )
        .unwrap();
        let copy = curve.clone();
        assert!((curve.discount(0.5).unwrap() - copy.discount(0.5).unwrap()).abs() < 1e-15);
    }
}
