// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Par-rate view of a discount curve.
//!
//! A [`ParCurve`] is a lightweight borrowing view over a [`DiscountCurve`]
//! that exposes the curve as a function of par swap rates. For a swap
//! starting at `t_0`, maturing at `t_N`, with fixed-leg payment frequency
//! `freq` and accrual day-count `dc`,
//!
//! ```text
//! r_par  =  (D(t_0) - D(t_N)) / sum_i tau_i D(t_i),
//! ```
//!
//! where `t_i` are the period end dates and `tau_i = dc.year_fraction(t_{i-1},
//! t_i)`. This is the **single-curve** par-rate formula: the same curve both
//! discounts and projects, which collapses the float-leg PV to
//! `D(t_0) - D(t_N)` (Hagan & West 2006, §2.3).
//!
//! Multi-curve par rates — under an OIS discount curve with a separately
//! projected floating leg — live in `multi_curve.rs`.
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §2.3.
//! - Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling*, Vol. 1,
//!   Atlantic Financial Press (2010), §6.

use crate::curves::DiscountCurve;
use crate::errors::CurveError;
use crate::types::{Date, Daycount, Frequency};

/// Par-rate view of a [`DiscountCurve`].
///
/// Holds a borrow of the parent curve. All par-rate queries delegate to
/// [`DiscountCurve::par_swap_rate`].
///
/// # Examples
///
/// ```
/// use regit_curves::curves::{DiscountCurve, ParCurve};
/// use regit_curves::interpolation::Interpolation;
/// use regit_curves::types::{Date, Daycount, Frequency};
///
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
/// let p = ParCurve::from(&curve);
/// let par = p
///     .par_rate(
///         reference,
///         Date::from_ymd(2026, 1, 2).unwrap(),
///         Frequency::SemiAnnual,
///         Daycount::Act365F,
///     )
///     .unwrap();
/// assert!(par > 0.0);
/// ```
#[derive(Debug)]
pub struct ParCurve<'a> {
    curve: &'a DiscountCurve,
}

impl<'a> ParCurve<'a> {
    /// Constructs a par-rate view over the supplied discount curve.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::{DiscountCurve, ParCurve};
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let curve = DiscountCurve::from_times_and_discounts(
    ///     reference,
    ///     Daycount::Act365F,
    ///     &[0.0, 1.0],
    ///     &[1.0, 0.95],
    ///     Interpolation::LogLinear,
    /// )
    /// .unwrap();
    /// let _p = ParCurve::from(&curve);
    /// ```
    #[must_use]
    #[inline]
    pub fn from(curve: &'a DiscountCurve) -> Self {
        Self { curve }
    }

    /// Par swap rate for a regular swap from `start` to `maturity`, paying
    /// at `freq`, accruing under `daycount`. Delegates to
    /// [`DiscountCurve::par_swap_rate`].
    ///
    /// # Errors
    ///
    /// - [`CurveError::InvalidTime`] if `start >= maturity` or the schedule
    ///   is irregular at the requested frequency.
    /// - [`CurveError::Type`] if a day-count year-fraction query fails.
    /// - [`CurveError::NonPositiveDiscount`] if any discount on the schedule
    ///   evaluates non-positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::{DiscountCurve, ParCurve};
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Date, Daycount, Frequency};
    ///
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
    /// let p = ParCurve::from(&curve);
    /// let par = p
    ///     .par_rate(
    ///         reference,
    ///         Date::from_ymd(2026, 1, 2).unwrap(),
    ///         Frequency::Annual,
    ///         Daycount::Act365F,
    ///     )
    ///     .unwrap();
    /// assert!(par > 0.0);
    /// ```
    pub fn par_rate(
        &self,
        start: Date,
        maturity: Date,
        freq: Frequency,
        daycount: Daycount,
    ) -> Result<f64, CurveError> {
        self.curve.par_swap_rate(start, maturity, freq, daycount)
    }

    /// Convenience: par rate of a swap starting at the parent curve's
    /// reference date and maturing at `maturity`.
    ///
    /// Equivalent to `self.par_rate(curve.reference_date(), maturity, freq,
    /// daycount)`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::par_rate`].
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::{DiscountCurve, ParCurve};
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Date, Daycount, Frequency};
    ///
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
    /// let p = ParCurve::from(&curve);
    /// let a = p
    ///     .par_rate(
    ///         reference,
    ///         Date::from_ymd(2026, 1, 2).unwrap(),
    ///         Frequency::Annual,
    ///         Daycount::Act365F,
    ///     )
    ///     .unwrap();
    /// let b = p
    ///     .par_rate_from_anchor(
    ///         Date::from_ymd(2026, 1, 2).unwrap(),
    ///         Frequency::Annual,
    ///         Daycount::Act365F,
    ///     )
    ///     .unwrap();
    /// assert!((a - b).abs() < 1e-15);
    /// ```
    pub fn par_rate_from_anchor(
        &self,
        maturity: Date,
        freq: Frequency,
        daycount: Daycount,
    ) -> Result<f64, CurveError> {
        self.curve
            .par_swap_rate(self.curve.reference_date(), maturity, freq, daycount)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolation::Interpolation;

    fn d(y: i32, m: u32, day: u32) -> Date {
        Date::from_ymd(y, m, day).unwrap()
    }

    fn reference_date() -> Date {
        d(2024, 1, 2)
    }

    /// Build a tabulated flat continuous-r curve, quarterly resolution to
    /// 30 years.
    fn flat_curve(r_c: f64) -> DiscountCurve {
        let mut times = Vec::new();
        let mut discs = Vec::new();
        for i in 0..=120 {
            let date = Date::from_serial(reference_date().serial() + i * 91);
            let t = Daycount::Act365F
                .year_fraction(reference_date(), date)
                .unwrap();
            times.push(t);
            discs.push((-r_c * t).exp());
        }
        DiscountCurve::from_times_and_discounts(
            reference_date(),
            Daycount::Act365F,
            &times,
            &discs,
            Interpolation::LogLinear,
        )
        .unwrap()
    }

    /// Compute the closed-form par swap rate on a flat continuous curve.
    fn closed_form_par(
        reference: Date,
        maturity: Date,
        freq: Frequency,
        accrual_dc: Daycount,
        r_c: f64,
    ) -> f64 {
        // Reconstruct the schedule by laying down `12 / n`-month boundaries.
        let periods_per_year = freq.periods_per_year();
        let months_per_period = i32::try_from(12 / periods_per_year).unwrap_or(1);
        let mut dates = vec![reference];
        let mut step: i32 = 1;
        loop {
            let nxt =
                crate::types::Tenor::new(months_per_period * step, crate::types::TenorUnit::Months)
                    .add_to(reference);
            dates.push(nxt);
            if nxt.serial() == maturity.serial() {
                break;
            }
            step += 1;
        }
        let mut annuity = 0.0_f64;
        for i in 0..(dates.len() - 1) {
            let start = dates[i];
            let end = dates[i + 1];
            let tau = accrual_dc.year_fraction(start, end).unwrap();
            let t = Daycount::Act365F.year_fraction(reference, end).unwrap();
            annuity += tau * (-r_c * t).exp();
        }
        let t_end = Daycount::Act365F
            .year_fraction(reference, maturity)
            .unwrap();
        (1.0 - (-r_c * t_end).exp()) / annuity
    }

    #[test]
    fn from_constructs_view() {
        let curve = flat_curve(0.04);
        let _p = ParCurve::from(&curve);
    }

    #[test]
    fn par_rate_flat_curve_2y_semi_annual_matches_closed_form() {
        let r_c = 0.04_f64;
        let curve = flat_curve(r_c);
        let p = ParCurve::from(&curve);
        let par = p
            .par_rate(
                reference_date(),
                d(2026, 1, 2),
                Frequency::SemiAnnual,
                Daycount::Act365F,
            )
            .unwrap();
        let expected = closed_form_par(
            reference_date(),
            d(2026, 1, 2),
            Frequency::SemiAnnual,
            Daycount::Act365F,
            r_c,
        );
        assert!(
            (par - expected).abs() < 1e-12,
            "par={par}, expected={expected}"
        );
    }

    #[test]
    fn par_rate_from_anchor_matches_par_rate() {
        let curve = flat_curve(0.04);
        let p = ParCurve::from(&curve);
        let a = p
            .par_rate(
                reference_date(),
                d(2026, 1, 2),
                Frequency::Annual,
                Daycount::Act365F,
            )
            .unwrap();
        let b = p
            .par_rate_from_anchor(d(2026, 1, 2), Frequency::Annual, Daycount::Act365F)
            .unwrap();
        assert!((a - b).abs() < 1e-15);
    }

    #[test]
    fn par_rate_increases_with_frequency_on_upward_curve() {
        // On a flat continuous curve r_c, the par rate is r_c plus a small
        // compounding-gap that vanishes as the payment frequency increases.
        // Concretely on a 2y flat-0.04 curve: annual gap ≈ +81bp, semi-
        // annual ≈ +40bp, quarterly ≈ +20bp.
        let r_c = 0.04_f64;
        let curve = flat_curve(r_c);
        let p = ParCurve::from(&curve);
        let p_a = p
            .par_rate_from_anchor(d(2026, 1, 2), Frequency::Annual, Daycount::Act365F)
            .unwrap();
        let p_s = p
            .par_rate_from_anchor(d(2026, 1, 2), Frequency::SemiAnnual, Daycount::Act365F)
            .unwrap();
        let p_q = p
            .par_rate_from_anchor(d(2026, 1, 2), Frequency::Quarterly, Daycount::Act365F)
            .unwrap();
        // Loose sanity: all three are within 100bp of the continuous base.
        assert!((p_a - r_c).abs() < 1e-2);
        assert!((p_s - r_c).abs() < 1e-2);
        assert!((p_q - r_c).abs() < 1e-2);
        // All three exceed r_c (simple-compounded rate > continuous-
        // compounded rate for positive r).
        assert!(p_a > r_c);
        assert!(p_s > r_c);
        assert!(p_q > r_c);
        // More frequent payments -> par rate closer to r_c from above.
        assert!(p_q < p_s);
        assert!(p_s < p_a);
    }

    #[test]
    fn par_rate_rejects_inverted_dates() {
        let curve = flat_curve(0.04);
        let p = ParCurve::from(&curve);
        let err = p
            .par_rate(
                d(2025, 1, 2),
                d(2024, 1, 2),
                Frequency::Annual,
                Daycount::Act365F,
            )
            .unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn par_rate_rejects_irregular_schedule() {
        let curve = flat_curve(0.04);
        let p = ParCurve::from(&curve);
        let err = p
            .par_rate(
                d(2024, 1, 2),
                d(2025, 2, 2),
                Frequency::SemiAnnual,
                Daycount::Act365F,
            )
            .unwrap_err();
        assert!(matches!(err, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn par_rate_once_at_maturity_single_period() {
        let r_c = 0.04_f64;
        let curve = flat_curve(r_c);
        let p = ParCurve::from(&curve);
        let par = p
            .par_rate_from_anchor(d(2025, 1, 2), Frequency::OnceAtMaturity, Daycount::Act365F)
            .unwrap();
        // Single-period swap: par rate = (1 - D(T)) / (tau * D(T)) (zero-
        // coupon par rate). On a flat curve this equals the simply-compounded
        // 1y rate.
        let t = Daycount::Act365F
            .year_fraction(reference_date(), d(2025, 1, 2))
            .unwrap();
        let d_t = (-r_c * t).exp();
        let expected = (1.0 - d_t) / (t * d_t);
        assert!(
            (par - expected).abs() < 1e-12,
            "par={par}, expected={expected}"
        );
    }
}
