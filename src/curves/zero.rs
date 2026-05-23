// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Zero-rate view of a discount curve.
//!
//! A [`ZeroCurve`] is a lightweight borrowing view over a [`DiscountCurve`]
//! that exposes the curve as a function of zero rate `z(t)` under a chosen
//! [`Compounding`]. For continuous compounding:
//!
//! ```text
//! z(t) = -ln D(t) / t,    t > 0.
//! D(t) = exp(-z(t) * t).
//! ```
//!
//! For other compounding choices the conversion runs through
//! [`Compounding::rate_from_discount`].
//!
//! The view does not own the curve nodes — it stores `&'a DiscountCurve` and
//! the chosen compounding. This keeps the view zero-cost to construct but
//! ties its lifetime to that of the parent curve.
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §2.

use crate::curves::DiscountCurve;
use crate::errors::CurveError;
use crate::types::{Compounding, Date};

/// Zero-rate view of a [`DiscountCurve`].
///
/// Stores a borrow of the parent curve and the compounding convention under
/// which zero rates are reported. All queries delegate to the parent curve
/// and then convert via [`Compounding::rate_from_discount`].
///
/// # Examples
///
/// ```
/// use regit_curves::curves::{DiscountCurve, ZeroCurve};
/// use regit_curves::interpolation::Interpolation;
/// use regit_curves::types::{Compounding, Date, Daycount};
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
/// let z = ZeroCurve::from(&curve, Compounding::Continuous);
/// assert!((z.rate(1.0).unwrap() - 0.04).abs() < 1e-12);
/// ```
#[derive(Debug)]
pub struct ZeroCurve<'a> {
    curve: &'a DiscountCurve,
    compounding: Compounding,
}

impl<'a> ZeroCurve<'a> {
    /// Constructs a zero-rate view over the supplied discount curve under
    /// the chosen `compounding`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::{DiscountCurve, ZeroCurve};
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Compounding, Date, Daycount};
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
    /// let z = ZeroCurve::from(&curve, Compounding::Simple);
    /// assert_eq!(z.compounding(), Compounding::Simple);
    /// ```
    #[must_use]
    #[inline]
    pub fn from(curve: &'a DiscountCurve, compounding: Compounding) -> Self {
        Self { curve, compounding }
    }

    /// Compounding convention under which zero rates are reported.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::{DiscountCurve, ZeroCurve};
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Compounding, Date, Daycount};
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
    /// let z = ZeroCurve::from(&curve, Compounding::Continuous);
    /// assert_eq!(z.compounding(), Compounding::Continuous);
    /// ```
    #[must_use]
    #[inline]
    pub fn compounding(&self) -> Compounding {
        self.compounding
    }

    /// Zero rate at year fraction `t` from the parent curve's reference date.
    ///
    /// Delegates to [`DiscountCurve::zero_rate`].
    ///
    /// # Errors
    ///
    /// - [`CurveError::InvalidTime`] if `t` is negative or non-finite.
    /// - [`CurveError::Type`] if the compounding inverse fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::{DiscountCurve, ZeroCurve};
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Compounding, Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let curve = DiscountCurve::from_times_and_discounts(
    ///     reference,
    ///     Daycount::Act365F,
    ///     &[0.0, 1.0],
    ///     &[1.0, (-0.04_f64).exp()],
    ///     Interpolation::LogLinear,
    /// )
    /// .unwrap();
    /// let z = ZeroCurve::from(&curve, Compounding::Continuous);
    /// assert!((z.rate(1.0).unwrap() - 0.04).abs() < 1e-12);
    /// ```
    pub fn rate(&self, t: f64) -> Result<f64, CurveError> {
        self.curve.zero_rate(t, self.compounding)
    }

    /// Zero rate at `date`, computed by translating to a year fraction under
    /// the curve's day-count.
    ///
    /// # Errors
    ///
    /// - [`CurveError::Type`] if the day-count query fails.
    /// - [`CurveError::InvalidTime`] if the resulting year fraction is
    ///   non-finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::{DiscountCurve, ZeroCurve};
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Compounding, Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let later = Date::from_ymd(2025, 1, 2).unwrap();
    /// let curve = DiscountCurve::new(
    ///     reference,
    ///     Daycount::Act365F,
    ///     &[(reference, 1.0), (later, (-0.04_f64).exp())],
    ///     Interpolation::LogLinear,
    /// )
    /// .unwrap();
    /// let z = ZeroCurve::from(&curve, Compounding::Continuous);
    /// let r = z.rate_at(later).unwrap();
    /// // Act/365F over 366 days (2024 leap) -> t ~ 1.00274; r ~ 0.04 / 1.00274.
    /// assert!((r - 0.04 / (366.0_f64 / 365.0)).abs() < 1e-12);
    /// ```
    pub fn rate_at(&self, date: Date) -> Result<f64, CurveError> {
        let t = self
            .curve
            .daycount()
            .year_fraction(self.curve.reference_date(), date)?;
        self.rate(t)
    }

    /// Discount factor at year fraction `t` — delegates to the parent
    /// [`DiscountCurve::discount`].
    ///
    /// # Errors
    ///
    /// - [`CurveError::InvalidTime`] if `t` is negative or non-finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::{DiscountCurve, ZeroCurve};
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Compounding, Date, Daycount};
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
    /// let z = ZeroCurve::from(&curve, Compounding::Continuous);
    /// assert!((z.discount(1.0).unwrap() - 0.95).abs() < 1e-14);
    /// ```
    pub fn discount(&self, t: f64) -> Result<f64, CurveError> {
        self.curve.discount(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolation::Interpolation;
    use crate::types::Daycount;

    fn d(y: i32, m: u32, day: u32) -> Date {
        Date::from_ymd(y, m, day).unwrap()
    }

    fn reference_date() -> Date {
        d(2024, 1, 2)
    }

    /// Build a tabulated flat continuous-r curve at quarterly resolution.
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

    #[test]
    fn from_constructs_view() {
        let curve = flat_curve(0.04);
        let z = ZeroCurve::from(&curve, Compounding::Continuous);
        assert_eq!(z.compounding(), Compounding::Continuous);
    }

    #[test]
    fn rate_flat_curve_continuous() {
        let curve = flat_curve(0.04);
        let z = ZeroCurve::from(&curve, Compounding::Continuous);
        for t in [0.5, 1.0, 5.0, 10.0] {
            let r = z.rate(t).unwrap();
            assert!((r - 0.04).abs() < 1e-12, "t={t}: {r}");
        }
    }

    #[test]
    fn rate_flat_curve_simple() {
        let curve = flat_curve(0.04);
        let z = ZeroCurve::from(&curve, Compounding::Simple);
        // r_simple at t = (1/D - 1)/t, with D = exp(-r_c * t).
        for t in [0.5_f64, 1.0, 5.0] {
            let r = z.rate(t).unwrap();
            let expected = ((-0.04_f64 * t).exp().recip() - 1.0) / t;
            assert!((r - expected).abs() < 1e-12, "t={t}: {r} vs {expected}");
        }
    }

    #[test]
    fn rate_round_trip_through_discount() {
        // For any compounding c and any t, ZeroCurve::rate(t) inverts to
        // DiscountCurve::discount(t) via Compounding::discount_from_rate.
        let curve = flat_curve(0.04);
        for compounding in [
            Compounding::Continuous,
            Compounding::Simple,
            Compounding::Periodic {
                periods_per_year: 2,
            },
        ] {
            let z = ZeroCurve::from(&curve, compounding);
            for t in [0.5_f64, 1.0, 3.0, 7.0] {
                let r = z.rate(t).unwrap();
                let d = compounding.discount_from_rate(r, t).unwrap();
                let d_curve = z.discount(t).unwrap();
                assert!(
                    (d - d_curve).abs() < 1e-12,
                    "round-trip {compounding:?} at t={t}: {d} vs {d_curve}"
                );
            }
        }
    }

    #[test]
    fn discount_delegates_to_parent() {
        let curve = flat_curve(0.04);
        let z = ZeroCurve::from(&curve, Compounding::Continuous);
        // Pull a few knot points from the curve and verify identity.
        for t in [0.0_f64, 1.0, 5.0] {
            let a = z.discount(t).unwrap();
            let b = curve.discount(t).unwrap();
            assert!((a - b).abs() < 1e-15, "t={t}: {a} vs {b}");
        }
    }

    #[test]
    fn rate_at_uses_curve_daycount() {
        let curve = DiscountCurve::new(
            reference_date(),
            Daycount::Act365F,
            &[(reference_date(), 1.0), (d(2025, 1, 2), (-0.04_f64).exp())],
            Interpolation::LogLinear,
        )
        .unwrap();
        let z = ZeroCurve::from(&curve, Compounding::Continuous);
        let r = z.rate_at(d(2025, 1, 2)).unwrap();
        // Year fraction is 366 days / 365 (2024 is a leap year), so the
        // implied continuous zero rate is 0.04 / (366/365).
        let expected = 0.04 / (366.0_f64 / 365.0);
        assert!((r - expected).abs() < 1e-12);
    }

    #[test]
    fn rate_rejects_negative_t() {
        let curve = flat_curve(0.04);
        let z = ZeroCurve::from(&curve, Compounding::Continuous);
        assert!(matches!(
            z.rate(-1.0).unwrap_err(),
            CurveError::InvalidTime { .. }
        ));
    }
}
