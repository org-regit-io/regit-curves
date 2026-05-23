// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Forward-rate view of a discount curve.
//!
//! A [`ForwardCurve`] is a lightweight borrowing view over a
//! [`DiscountCurve`] that exposes both:
//!
//! - **Instantaneous forward** `f(t) = -d/dt ln D(t)`, and
//! - **Simply-compounded forward** over `[t_1, t_2]`:
//!   `L(t_1, t_2) = (D(t_1) / D(t_2) - 1) / (t_2 - t_1)`.
//!
//! Both delegate to the parent [`DiscountCurve`].
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §2.

use crate::curves::DiscountCurve;
use crate::errors::CurveError;
use crate::types::Daycount;

/// Forward-rate view of a [`DiscountCurve`].
///
/// Holds a borrow of the parent curve. All forward-rate queries delegate to
/// the parent curve's [`DiscountCurve::instantaneous_forward`] and
/// [`DiscountCurve::forward_rate`].
///
/// # Examples
///
/// ```
/// use regit_curves::curves::{DiscountCurve, ForwardCurve};
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
/// let f = ForwardCurve::from(&curve);
/// // Instantaneous forward at t = 0.5y on a flat-r=0.04 curve = 0.04.
/// assert!((f.instantaneous(0.5).unwrap() - 0.04).abs() < 1e-10);
/// ```
#[derive(Debug)]
pub struct ForwardCurve<'a> {
    curve: &'a DiscountCurve,
}

impl<'a> ForwardCurve<'a> {
    /// Constructs a forward-rate view over the supplied discount curve.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::{DiscountCurve, ForwardCurve};
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
    /// let _f = ForwardCurve::from(&curve);
    /// ```
    #[must_use]
    #[inline]
    pub fn from(curve: &'a DiscountCurve) -> Self {
        Self { curve }
    }

    /// Instantaneous forward `f(t) = -d/dt ln D(t)`.
    ///
    /// Delegates to [`DiscountCurve::instantaneous_forward`].
    ///
    /// # Errors
    ///
    /// - [`CurveError::InvalidTime`] if `t` is negative or non-finite.
    /// - [`CurveError::NonPositiveDiscount`] if the curve evaluates to a
    ///   non-positive discount at `t`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::{DiscountCurve, ForwardCurve};
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
    /// let f = ForwardCurve::from(&curve);
    /// assert!((f.instantaneous(0.5).unwrap() - 0.04).abs() < 1e-10);
    /// ```
    pub fn instantaneous(&self, t: f64) -> Result<f64, CurveError> {
        self.curve.instantaneous_forward(t)
    }

    /// Simply-compounded forward rate over `[t_1, t_2]` with `t_2 > t_1 >= 0`:
    ///
    /// ```text
    /// L(t_1, t_2) = (D(t_1) / D(t_2) - 1) / (t_2 - t_1).
    /// ```
    ///
    /// Delegates to [`DiscountCurve::forward_rate`]; see that method's
    /// documentation for the simplification of the `daycount` argument.
    ///
    /// # Errors
    ///
    /// - [`CurveError::InvalidTime`] if either endpoint is invalid or
    ///   `t_2 <= t_1`.
    /// - [`CurveError::NonPositiveDiscount`] if either endpoint's discount
    ///   evaluates non-positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::{DiscountCurve, ForwardCurve};
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
    /// let f = ForwardCurve::from(&curve);
    /// let l = f.forward(1.0, 2.0, Daycount::Act365F).unwrap();
    /// assert!((l - 0.04_f64.exp_m1()).abs() < 1e-12);
    /// ```
    pub fn forward(&self, t1: f64, t2: f64, daycount: Daycount) -> Result<f64, CurveError> {
        self.curve.forward_rate(t1, t2, daycount)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolation::Interpolation;
    use crate::types::Date;

    fn d(y: i32, m: u32, day: u32) -> Date {
        Date::from_ymd(y, m, day).unwrap()
    }

    fn reference_date() -> Date {
        d(2024, 1, 2)
    }

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
        let _f = ForwardCurve::from(&curve);
    }

    #[test]
    fn instantaneous_flat_curve_loglinear() {
        let r_c = 0.04;
        let curve = flat_curve(r_c);
        let f = ForwardCurve::from(&curve);
        for t in [0.5_f64, 1.0, 5.0, 10.0] {
            let v = f.instantaneous(t).unwrap();
            assert!((v - r_c).abs() < 1e-10, "t={t}: {v}");
        }
    }

    #[test]
    fn forward_flat_curve_closed_form() {
        let r_c = 0.04_f64;
        let curve = flat_curve(r_c);
        let f = ForwardCurve::from(&curve);
        let cases = [(1.0_f64, 2.0_f64), (0.5, 3.0), (2.0, 5.0)];
        for (t1, t2) in cases {
            let l = f.forward(t1, t2, Daycount::Act365F).unwrap();
            let expected = (r_c * (t2 - t1)).exp_m1() / (t2 - t1);
            assert!((l - expected).abs() < 1e-12, "fwd[{t1},{t2}]: {l}");
        }
    }

    #[test]
    fn instantaneous_via_forward_finite_difference_limit() {
        // On a smooth curve, `forward(t, t+h, _)` converges to
        // `instantaneous(t)` as h -> 0. Verify to 1e-6 with h = 1e-4.
        let r_c = 0.04_f64;
        let curve = flat_curve(r_c);
        let f = ForwardCurve::from(&curve);
        let t = 1.5;
        let h = 1e-4_f64;
        // Use a finite-difference window that doesn't straddle a knot.
        let l = f.forward(t, t + h, Daycount::Act365F).unwrap();
        let inst = f.instantaneous(t).unwrap();
        assert!(
            (l - inst).abs() < 1e-6,
            "fwd FD = {l}, instantaneous = {inst}"
        );
    }

    #[test]
    fn forward_rejects_t2_le_t1() {
        let curve = flat_curve(0.04);
        let f = ForwardCurve::from(&curve);
        assert!(matches!(
            f.forward(2.0, 1.0, Daycount::Act365F).unwrap_err(),
            CurveError::InvalidTime { .. }
        ));
    }

    #[test]
    fn instantaneous_rejects_negative_time() {
        let curve = flat_curve(0.04);
        let f = ForwardCurve::from(&curve);
        assert!(matches!(
            f.instantaneous(-0.5).unwrap_err(),
            CurveError::InvalidTime { .. }
        ));
    }

    #[test]
    fn instantaneous_at_anchor() {
        // At t = 0 on a flat-r curve, the instantaneous fwd should still
        // recover r_c (right-derivative of log D at the anchor).
        let r_c = 0.04_f64;
        let curve = flat_curve(r_c);
        let f = ForwardCurve::from(&curve);
        let v = f.instantaneous(0.0).unwrap();
        // LogLinear's deriv at t = 0 returns the right-segment slope; on a
        // flat curve that equals -r_c * D(0) = -r_c. So fwd = r_c.
        assert!(
            (v - r_c).abs() < 1e-10,
            "instantaneous at anchor: {v}, expected {r_c}"
        );
    }
}
