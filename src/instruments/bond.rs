// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Coupon-bearing bond instrument.
//!
//! A coupon-bearing bond pays a regular fixed coupon equal to
//! `coupon * tau_i * notional` on each scheduled coupon date `t_i`, and
//! repays the notional principal on the maturity date `t_N`. By
//! no-arbitrage the bond's dirty price equals the present value of the
//! coupon and principal cash flows:
//!
//! ```text
//! dirty_price = clean_price + accrued
//!             = coupon * SUM_i tau_i * D(t_i) * notional
//!               + notional * D(t_N),
//! ```
//!
//! where `tau_i` is the period accrual under [`Bond::daycount`] and `D(t)`
//! is the curve's discount factor evaluated under the curve's own
//! day-count. The residual returned by the instrument's `residual` method
//! is
//!
//! ```text
//! residual = coupon_pv + principal_pv - dirty_price
//!          = coupon * SUM_i tau_i * D(t_i) * notional
//!            + notional * D(t_N)
//!            - (clean_price + accrued).
//! ```
//!
//! Zero at the bootstrap solution. For a **par bond at issue** the quoted
//! clean price equals the notional and the accrued interest is zero, so
//! the par-bond identity reduces to
//!
//! ```text
//! coupon * SUM_i tau_i * D(t_i) + D(t_N) = 1,
//! ```
//!
//! which is the same shape as the OIS-swap par equation.
//!
//! # References
//!
//! - Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling*,
//!   Volume I: Foundations and Vanilla Models, Atlantic Financial Press
//!   (2010), §5.1-5.2. Bond pricing and bootstrap identity.
//! - ISDA, *2006 ISDA Definitions*, §6 ("Fixed Amounts and Floating
//!   Amounts"). Coupon calculation conventions.

use crate::errors::BootstrapError;
use crate::types::{Date, Daycount, Frequency};

use super::schedule::SwapSchedule;
use super::{CurveSnapshot, InstrumentLike};

/// A coupon-bearing bond, quoted by clean price (plus accrued interest).
///
/// Fields:
///
/// - `issue` — issue / settlement date when the curve begins discounting
///   (`t_0`).
/// - `maturity` — maturity date when the principal is repaid (`t_N`).
/// - `coupon` — annualised coupon rate, decimal (e.g. `0.05` for 5%).
/// - `freq` — coupon frequency (typically [`Frequency::SemiAnnual`] for
///   USD / GBP government bonds, [`Frequency::Annual`] for many Eurozone
///   sovereign issues).
/// - `daycount` — day-count convention for the coupon accruals `tau_i`.
/// - `notional` — principal repaid at maturity. Typically `1.0` for a
///   par-quoted bond or `100.0` for percent-of-par quoting.
/// - `clean_price` — quoted clean price, in the same units as `notional`.
///   A par bond quotes at `notional`; a discount bond below; a premium
///   bond above.
/// - `accrued` — accrued interest at settlement, in the same units as
///   `notional`. Zero at issue; non-zero when the bond is quoted between
///   coupon dates. The dirty price is `clean_price + accrued`.
/// - `schedule` — coupon schedule generated from `(issue, maturity, freq)`.
///
/// Constructed via [`Bond::new`] (which builds the regular schedule
/// internally) or [`Bond::with_schedule`] (which accepts a caller-built
/// schedule for irregular cases such as stub periods).
///
/// # Examples
///
/// ```
/// use regit_curves::instruments::Bond;
/// use regit_curves::types::{Date, Daycount, Frequency};
///
/// let issue    = Date::from_ymd(2024, 1, 2).unwrap();
/// let maturity = Date::from_ymd(2029, 1, 2).unwrap();
/// // 5y annual 5% bond on Act/365F, quoted at par with no accrued.
/// let bond = Bond::new(
///     issue,
///     maturity,
///     0.05,
///     Frequency::Annual,
///     Daycount::Act365F,
///     1.0,
///     1.0,
///     0.0,
/// )
/// .unwrap();
/// assert_eq!(bond.schedule.len(), 5);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Bond {
    /// Issue / settlement date — when the curve begins discounting.
    pub issue: Date,
    /// Maturity date — when principal is repaid.
    pub maturity: Date,
    /// Annualised coupon rate (decimal, e.g. `0.05` for 5%).
    pub coupon: f64,
    /// Coupon frequency.
    pub freq: Frequency,
    /// Day-count for coupon accruals.
    pub daycount: Daycount,
    /// Notional principal repaid at maturity (typically `1.0` for a
    /// par-quoted bond or `100.0` for percent-of-par quoting).
    pub notional: f64,
    /// Quoted clean price (in the same units as `notional`). A par bond
    /// quotes at `notional` (clean price = 1.0 for `notional = 1.0`, or
    /// 100.0 for `notional = 100.0`); a discount bond quotes below par; a
    /// premium bond quotes above.
    pub clean_price: f64,
    /// Accrued interest at settlement (in the same units as `notional`).
    /// Conventionally zero at issue; non-zero if the bond is quoted
    /// between coupon dates. The dirty price is `clean_price + accrued`.
    pub accrued: f64,
    /// Coupon schedule generated from `(issue, maturity, freq)`.
    pub schedule: SwapSchedule,
}

impl Bond {
    /// Constructs a bond with a regular coupon schedule generated from
    /// `(issue, maturity, freq)`.
    ///
    /// Validation:
    ///
    /// - `coupon` must be finite (negative coupons are accepted for
    ///   completeness; negative-yielding bonds with coupon-stripped
    ///   structures exist in EUR / CHF markets).
    /// - `notional > 0`.
    /// - `clean_price > 0`.
    /// - `accrued` finite and `>= 0`.
    /// - `issue < maturity`.
    /// - The regular schedule generator must succeed — see
    ///   [`SwapSchedule::from_regular`] for the regularity constraint.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if any validation step fails,
    ///   or if [`SwapSchedule::from_regular`] returns an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::Bond;
    /// use regit_curves::types::{Date, Daycount, Frequency};
    /// use regit_curves::BootstrapError;
    ///
    /// let issue    = Date::from_ymd(2024, 1, 2).unwrap();
    /// let maturity = Date::from_ymd(2029, 1, 2).unwrap();
    /// assert!(
    ///     Bond::new(
    ///         issue, maturity, 0.05, Frequency::Annual, Daycount::Act365F,
    ///         1.0, 1.0, 0.0,
    ///     )
    ///     .is_ok()
    /// );
    /// // Non-positive notional rejected:
    /// assert!(matches!(
    ///     Bond::new(
    ///         issue, maturity, 0.05, Frequency::Annual, Daycount::Act365F,
    ///         0.0, 1.0, 0.0,
    ///     )
    ///     .unwrap_err(),
    ///     BootstrapError::InvalidInstrument { .. },
    /// ));
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        issue: Date,
        maturity: Date,
        coupon: f64,
        freq: Frequency,
        daycount: Daycount,
        notional: f64,
        clean_price: f64,
        accrued: f64,
    ) -> Result<Self, BootstrapError> {
        Self::validate(coupon, notional, clean_price, accrued, issue, maturity)?;
        let schedule = SwapSchedule::from_regular(issue, maturity, freq)?;
        Ok(Self {
            issue,
            maturity,
            coupon,
            freq,
            daycount,
            notional,
            clean_price,
            accrued,
            schedule,
        })
    }

    /// Constructs a bond from an already-built schedule.
    ///
    /// Use this entry point when the coupon schedule is irregular (e.g. a
    /// stub first or last period). The supplied schedule must agree with
    /// `(issue, maturity)`: its first boundary date must equal `issue` and
    /// its last must equal `maturity`.
    ///
    /// Validation matches [`Bond::new`]; additionally the schedule's
    /// endpoints are checked against `(issue, maturity)`.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if any validation step
    ///   fails, or if the schedule's endpoints do not match
    ///   `(issue, maturity)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::{Bond, SwapSchedule};
    /// use regit_curves::types::{Date, Daycount, Frequency};
    ///
    /// let issue    = Date::from_ymd(2024, 1, 2).unwrap();
    /// let maturity = Date::from_ymd(2026, 1, 2).unwrap();
    /// let sch = SwapSchedule::from_regular(issue, maturity, Frequency::Annual).unwrap();
    /// let bond = Bond::with_schedule(
    ///     issue, maturity, 0.05, Frequency::Annual, Daycount::Act365F,
    ///     1.0, 1.0, 0.0, sch,
    /// )
    /// .unwrap();
    /// assert_eq!(bond.schedule.len(), 2);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn with_schedule(
        issue: Date,
        maturity: Date,
        coupon: f64,
        freq: Frequency,
        daycount: Daycount,
        notional: f64,
        clean_price: f64,
        accrued: f64,
        schedule: SwapSchedule,
    ) -> Result<Self, BootstrapError> {
        Self::validate(coupon, notional, clean_price, accrued, issue, maturity)?;
        if schedule.start() != issue || schedule.maturity() != maturity {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "schedule endpoints must match (issue, maturity)",
            });
        }
        Ok(Self {
            issue,
            maturity,
            coupon,
            freq,
            daycount,
            notional,
            clean_price,
            accrued,
            schedule,
        })
    }

    /// Shared validation for [`Bond::new`] and [`Bond::with_schedule`].
    fn validate(
        coupon: f64,
        notional: f64,
        clean_price: f64,
        accrued: f64,
        issue: Date,
        maturity: Date,
    ) -> Result<(), BootstrapError> {
        if !coupon.is_finite() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "bond coupon must be finite",
            });
        }
        if !notional.is_finite() || notional <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "bond notional must be strictly positive",
            });
        }
        if !clean_price.is_finite() || clean_price <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "bond clean price must be strictly positive",
            });
        }
        if !accrued.is_finite() || accrued < 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "bond accrued interest must be finite and non-negative",
            });
        }
        if issue.days_between(maturity) <= 0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "bond issue must precede maturity",
            });
        }
        Ok(())
    }

    /// Present value of the coupon-leg cashflows at `reference_date`:
    ///
    /// ```text
    /// coupon_pv = coupon * SUM_i tau_i * D(t_i) * notional,
    /// ```
    ///
    /// with `tau_i` the period accruals under [`Bond::daycount`] and
    /// `D(t_i)` the curve's discount factor at each coupon's payment date.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::Type`] if the day-count convention cannot
    ///   produce a year fraction (e.g. [`Daycount::Business252`] without a
    ///   calendar).
    /// - [`BootstrapError::InvalidInstrument`] if the curve snapshot is
    ///   empty or returns a non-positive discount factor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use regit_curves::instruments::Bond;
    /// # use regit_curves::types::{Date, Daycount, Frequency};
    /// let issue    = Date::from_ymd(2024, 1, 2).unwrap();
    /// let maturity = Date::from_ymd(2025, 1, 2).unwrap();
    /// let bond = Bond::new(
    ///     issue, maturity, 0.05, Frequency::Annual, Daycount::Act365F,
    ///     1.0, 1.0, 0.0,
    /// ).unwrap();
    /// assert!((bond.coupon - 0.05).abs() < 1e-15);
    /// ```
    pub(crate) fn coupon_pv(
        &self,
        _reference_date: Date,
        curve: &CurveSnapshot<'_>,
    ) -> Result<f64, BootstrapError> {
        let mut annuity = 0.0_f64;
        for i in 0..self.schedule.len() {
            let period_start = self.schedule.period_start(i);
            let period_end = self.schedule.period_end(i);
            let tau = self.daycount.year_fraction(period_start, period_end)?;
            let t_pay = curve
                .daycount
                .year_fraction(curve.reference_date, period_end)?;
            let d_pay = curve
                .discount_at(t_pay)
                .ok_or(BootstrapError::InvalidInstrument {
                    at_index: 0,
                    reason: "curve snapshot is empty",
                })?;
            if !d_pay.is_finite() || d_pay <= 0.0 {
                return Err(BootstrapError::InvalidInstrument {
                    at_index: 0,
                    reason: "non-positive discount factor in curve snapshot",
                });
            }
            annuity += tau * d_pay;
        }
        Ok(self.coupon * annuity * self.notional)
    }

    /// Present value of the principal repayment at `reference_date`:
    ///
    /// ```text
    /// principal_pv = notional * D(t_N).
    /// ```
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::Type`] if the curve's day-count cannot produce a
    ///   year fraction for `maturity`.
    /// - [`BootstrapError::InvalidInstrument`] if the curve snapshot is
    ///   empty or returns a non-positive discount factor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use regit_curves::instruments::Bond;
    /// # use regit_curves::types::{Date, Daycount, Frequency};
    /// let issue    = Date::from_ymd(2024, 1, 2).unwrap();
    /// let maturity = Date::from_ymd(2025, 1, 2).unwrap();
    /// let bond = Bond::new(
    ///     issue, maturity, 0.05, Frequency::Annual, Daycount::Act365F,
    ///     1.0, 1.0, 0.0,
    /// ).unwrap();
    /// assert_eq!(bond.maturity, maturity);
    /// ```
    pub(crate) fn principal_pv(
        &self,
        _reference_date: Date,
        curve: &CurveSnapshot<'_>,
    ) -> Result<f64, BootstrapError> {
        let t_maturity = curve
            .daycount
            .year_fraction(curve.reference_date, self.maturity)?;
        let d_maturity =
            curve
                .discount_at(t_maturity)
                .ok_or(BootstrapError::InvalidInstrument {
                    at_index: 0,
                    reason: "curve snapshot is empty",
                })?;
        if !d_maturity.is_finite() || d_maturity <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "non-positive discount factor in curve snapshot",
            });
        }
        Ok(self.notional * d_maturity)
    }
}

impl InstrumentLike for Bond {
    #[inline]
    fn pillar(&self) -> Date {
        self.maturity
    }

    fn residual(
        &self,
        reference_date: Date,
        curve: &CurveSnapshot<'_>,
    ) -> Result<f64, BootstrapError> {
        // residual = coupon_pv + principal_pv - dirty_price
        //          = coupon_pv + principal_pv - (clean_price + accrued).
        // Zero at the bootstrap solution.
        let coupon_pv = self.coupon_pv(reference_date, curve)?;
        let principal_pv = self.principal_pv(reference_date, curve)?;
        Ok(coupon_pv + principal_pv - (self.clean_price + self.accrued))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruments::CurveSnapshot;

    fn d(y: i32, m: u32, day: u32) -> Date {
        Date::from_ymd(y, m, day).unwrap()
    }

    // ─── Construction & validation ───────────────────────────────────────

    #[test]
    fn new_accepts_valid_5y_annual_bond() {
        let issue = d(2024, 1, 2);
        let maturity = d(2029, 1, 2);
        let bond = Bond::new(
            issue,
            maturity,
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            0.0,
        )
        .unwrap();
        assert_eq!(bond.issue, issue);
        assert_eq!(bond.maturity, maturity);
        assert!((bond.coupon - 0.05).abs() < 1e-15);
        assert_eq!(bond.freq, Frequency::Annual);
        assert_eq!(bond.daycount, Daycount::Act365F);
        assert!((bond.notional - 1.0).abs() < 1e-15);
        assert!((bond.clean_price - 1.0).abs() < 1e-15);
        assert!((bond.accrued - 0.0).abs() < 1e-15);
        assert_eq!(bond.schedule.len(), 5);
        assert_eq!(bond.pillar(), maturity);
    }

    #[test]
    fn new_rejects_nan_coupon() {
        let err = Bond::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            f64::NAN,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_inf_coupon() {
        let err = Bond::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            f64::INFINITY,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_zero_notional() {
        let err = Bond::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            0.0,
            1.0,
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_negative_notional() {
        let err = Bond::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            -1.0,
            1.0,
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_zero_clean_price() {
        let err = Bond::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            0.0,
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_negative_clean_price() {
        let err = Bond::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            -0.5,
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_negative_accrued() {
        let err = Bond::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            -0.01,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_nan_accrued() {
        let err = Bond::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            f64::NAN,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_inverted_dates() {
        let err = Bond::new(
            d(2029, 1, 2),
            d(2024, 1, 2),
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_equal_issue_and_maturity() {
        let s = d(2024, 1, 2);
        let err = Bond::new(
            s,
            s,
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_propagates_irregular_schedule_error() {
        let err = Bond::new(
            d(2024, 1, 2),
            d(2025, 2, 2),
            0.05,
            Frequency::SemiAnnual,
            Daycount::Act365F,
            1.0,
            1.0,
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    // ─── with_schedule entry point ───────────────────────────────────────

    #[test]
    fn with_schedule_accepts_matching_schedule() {
        let issue = d(2024, 1, 2);
        let maturity = d(2026, 1, 2);
        let sch = SwapSchedule::from_regular(issue, maturity, Frequency::Annual).unwrap();
        let bond = Bond::with_schedule(
            issue,
            maturity,
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            0.0,
            sch,
        )
        .unwrap();
        assert_eq!(bond.schedule.len(), 2);
    }

    #[test]
    fn with_schedule_rejects_mismatched_endpoints() {
        let issue = d(2024, 1, 2);
        let maturity = d(2026, 1, 2);
        let other = d(2027, 1, 2);
        let sch = SwapSchedule::from_regular(issue, other, Frequency::Annual).unwrap();
        let err = Bond::with_schedule(
            issue,
            maturity,
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            0.0,
            sch,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn with_schedule_rejects_invalid_inputs() {
        let issue = d(2024, 1, 2);
        let maturity = d(2026, 1, 2);
        let sch = SwapSchedule::from_regular(issue, maturity, Frequency::Annual).unwrap();
        let err = Bond::with_schedule(
            issue,
            maturity,
            f64::NAN,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            0.0,
            sch,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    // ─── Pricing identity on a flat continuously-compounded curve ────────

    /// Builds a hand-rolled flat continuously-compounded discount curve
    /// `D(t) = exp(-r * t)` over a quarterly grid spanning 30 years.
    fn flat_curve(reference_date: Date, daycount: Daycount, r: f64) -> (Vec<f64>, Vec<f64>) {
        let mut times = Vec::new();
        let mut discounts = Vec::new();
        for i in 0..=120 {
            let date = Date::from_serial(reference_date.serial() + i * 91);
            let t = daycount.year_fraction(reference_date, date).unwrap();
            times.push(t);
            discounts.push((-r * t).exp());
        }
        (times, discounts)
    }

    /// Closed-form par-bond coupon consistent with a flat continuously-
    /// compounded curve `D(t) = exp(-r_c * t)`. Solves
    /// `coupon * SUM_i tau_i * D(t_i) + D(t_N) = 1` (par-bond identity,
    /// unit notional, zero accrued).
    fn par_coupon_flat(bond: &Bond, reference: Date, r_c: f64) -> f64 {
        let dc_curve = bond.daycount; // same axis here
        let mut annuity = 0.0_f64;
        for i in 0..bond.schedule.len() {
            let s = bond.schedule.period_start(i);
            let e = bond.schedule.period_end(i);
            let tau = bond.daycount.year_fraction(s, e).unwrap();
            let t_pay = dc_curve.year_fraction(reference, e).unwrap();
            annuity += tau * (-r_c * t_pay).exp();
        }
        let t_n = dc_curve.year_fraction(reference, bond.maturity).unwrap();
        (1.0 - (-r_c * t_n).exp()) / annuity
    }

    #[test]
    fn par_bond_residual_is_zero_5y_annual_flat_5pct() {
        // Numerical target from the spec: 5y annual 5% bond on a flat 5%
        // continuously-compounded curve. Pick the par coupon analytically
        // and verify the residual is zero to 1e-12.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act365F;
        let r_c = 0.05_f64;
        let issue = reference;
        let maturity = d(2029, 1, 2);
        let (times, discounts) = flat_curve(reference, dc, r_c);

        // Derive par coupon analytically.
        let placeholder = Bond::new(
            issue,
            maturity,
            0.0, // dummy coupon for schedule
            Frequency::Annual,
            dc,
            1.0,
            1.0,
            0.0,
        )
        .unwrap();
        let par_coupon = par_coupon_flat(&placeholder, reference, r_c);

        let bond = Bond::new(
            issue,
            maturity,
            par_coupon,
            Frequency::Annual,
            dc,
            1.0,
            1.0,
            0.0,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discounts,
        };
        let residual = bond.residual(reference, &snapshot).unwrap();
        assert!(
            residual.abs() < 1e-12,
            "par-bond residual on flat curve must be zero to 1e-12, got {residual}",
        );
    }

    #[test]
    fn off_par_bond_residual_matches_clean_price_shift() {
        // Same bond at par coupon, but quoted at clean_price = 0.95 instead
        // of 1.0 -> residual should be PV - 0.95 = 1.0 - 0.95 = +0.05.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act365F;
        let r_c = 0.05_f64;
        let issue = reference;
        let maturity = d(2029, 1, 2);
        let (times, discounts) = flat_curve(reference, dc, r_c);

        let placeholder =
            Bond::new(issue, maturity, 0.0, Frequency::Annual, dc, 1.0, 1.0, 0.0).unwrap();
        let par_coupon = par_coupon_flat(&placeholder, reference, r_c);

        let bond = Bond::new(
            issue,
            maturity,
            par_coupon,
            Frequency::Annual,
            dc,
            1.0,
            0.95, // discount quote
            0.0,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discounts,
        };
        let residual = bond.residual(reference, &snapshot).unwrap();
        // PV = 1.0 (par bond), clean_price = 0.95, accrued = 0 -> residual
        // = 1.0 - 0.95 = 0.05.
        assert!(
            (residual - 0.05).abs() < 1e-12,
            "off-par residual should be 0.05, got {residual}",
        );
    }

    #[test]
    fn coupon_pv_matches_manual_sum() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act365F;
        let r_c = 0.04_f64;
        let issue = reference;
        let maturity = d(2027, 1, 2);
        let (times, discounts) = flat_curve(reference, dc, r_c);

        let bond = Bond::new(issue, maturity, 0.06, Frequency::Annual, dc, 1.0, 1.0, 0.0).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discounts,
        };

        let mut expected = 0.0_f64;
        for i in 0..bond.schedule.len() {
            let p_start = bond.schedule.period_start(i);
            let p_end = bond.schedule.period_end(i);
            let tau = bond.daycount.year_fraction(p_start, p_end).unwrap();
            let t = dc.year_fraction(reference, p_end).unwrap();
            expected += tau * (-r_c * t).exp();
        }
        expected *= 0.06; // coupon * annuity * notional, notional = 1.0
        let got = bond.coupon_pv(reference, &snapshot).unwrap();
        assert!(
            (got - expected).abs() < 1e-12,
            "coupon_pv mismatch: got {got}, expected {expected}",
        );
    }

    #[test]
    fn principal_pv_equals_notional_times_discount() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act365F;
        let r_c = 0.04_f64;
        let issue = reference;
        let maturity = d(2029, 1, 2);
        let (times, discounts) = flat_curve(reference, dc, r_c);

        let bond = Bond::new(
            issue,
            maturity,
            0.05,
            Frequency::Annual,
            dc,
            100.0,
            100.0,
            0.0,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discounts,
        };
        let t_n = dc.year_fraction(reference, maturity).unwrap();
        let expected = 100.0 * (-r_c * t_n).exp();
        let got = bond.principal_pv(reference, &snapshot).unwrap();
        assert!(
            (got - expected).abs() < 1e-10,
            "principal_pv mismatch: got {got}, expected {expected}",
        );
    }

    #[test]
    fn residual_errors_on_empty_snapshot() {
        let reference = d(2024, 1, 2);
        let bond = Bond::new(
            reference,
            d(2029, 1, 2),
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            0.0,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: Daycount::Act365F,
            times: &[],
            discounts: &[],
        };
        let err = bond.residual(reference, &snapshot).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn bond_residual_propagates_business252_error() {
        // Coupon day-count is Business252; the accrual query inside
        // `coupon_pv` surfaces the day-count error rather than producing a
        // silent NaN.
        let reference = d(2024, 1, 2);
        let dc_curve = Daycount::Act365F;
        let (times, discounts) = flat_curve(reference, dc_curve, 0.04);
        let bond = Bond::new(
            reference,
            d(2029, 1, 2),
            0.05,
            Frequency::Annual,
            Daycount::Business252,
            1.0,
            1.0,
            0.0,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc_curve,
            times: &times,
            discounts: &discounts,
        };
        let err = bond.residual(reference, &snapshot).unwrap_err();
        assert!(matches!(err, BootstrapError::Type(_)));
    }

    #[test]
    fn ten_year_semi_annual_par_bond_residual_zero() {
        // 10y semi-annual bond on a flat 4% curve.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act365F;
        let r_c = 0.04_f64;
        let issue = reference;
        let maturity = d(2034, 1, 2);
        let (times, discounts) = flat_curve(reference, dc, r_c);

        let placeholder = Bond::new(
            issue,
            maturity,
            0.0,
            Frequency::SemiAnnual,
            dc,
            1.0,
            1.0,
            0.0,
        )
        .unwrap();
        let par_coupon = par_coupon_flat(&placeholder, reference, r_c);

        let bond = Bond::new(
            issue,
            maturity,
            par_coupon,
            Frequency::SemiAnnual,
            dc,
            1.0,
            1.0,
            0.0,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discounts,
        };
        let residual = bond.residual(reference, &snapshot).unwrap();
        assert!(
            residual.abs() < 1e-10,
            "10y semi-annual residual at par should be < 1e-10, got {residual}",
        );
    }

    #[test]
    fn par_bond_invariant_under_mixed_daycounts() {
        // Coupons Thirty360BondBasis against Act/365F curve — the par-bond
        // identity still drives residual to zero with the right accruals.
        let reference = d(2024, 1, 2);
        let issue = reference;
        let maturity = d(2029, 1, 2);
        let dc_curve = Daycount::Act365F;
        let dc_coupon = Daycount::Thirty360BondBasis;
        let r_c = 0.04_f64;
        let (times, discounts) = flat_curve(reference, dc_curve, r_c);

        // Manual par-coupon under mixed day-counts.
        let schedule = SwapSchedule::from_regular(issue, maturity, Frequency::Annual).unwrap();
        let mut annuity = 0.0_f64;
        for i in 0..schedule.len() {
            let s = schedule.period_start(i);
            let e = schedule.period_end(i);
            let tau = dc_coupon.year_fraction(s, e).unwrap();
            let t_pay = dc_curve.year_fraction(reference, e).unwrap();
            annuity += tau * (-r_c * t_pay).exp();
        }
        let t_n = dc_curve.year_fraction(reference, maturity).unwrap();
        let par_coupon = (1.0 - (-r_c * t_n).exp()) / annuity;

        let bond = Bond::new(
            issue,
            maturity,
            par_coupon,
            Frequency::Annual,
            dc_coupon,
            1.0,
            1.0,
            0.0,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc_curve,
            times: &times,
            discounts: &discounts,
        };
        let residual = bond.residual(reference, &snapshot).unwrap();
        assert!(residual.abs() < 1e-10, "mixed-DC residual: {residual}");
    }

    #[test]
    fn debug_format_contains_struct_name() {
        let bond = Bond::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            0.0,
        )
        .unwrap();
        let s = format!("{bond:?}");
        assert!(s.contains("Bond"));
    }

    #[test]
    fn clone_and_eq_round_trip() {
        let bond = Bond::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            0.05,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            0.0,
        )
        .unwrap();
        let cloned = bond.clone();
        assert_eq!(bond, cloned);
    }

    #[test]
    fn residual_with_accrued_subtracts_dirty_price() {
        // PV = 1.0 (par), clean = 0.98, accrued = 0.02 -> residual = 0.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act365F;
        let r_c = 0.05_f64;
        let issue = reference;
        let maturity = d(2029, 1, 2);
        let (times, discounts) = flat_curve(reference, dc, r_c);

        let placeholder =
            Bond::new(issue, maturity, 0.0, Frequency::Annual, dc, 1.0, 1.0, 0.0).unwrap();
        let par_coupon = par_coupon_flat(&placeholder, reference, r_c);

        let bond = Bond::new(
            issue,
            maturity,
            par_coupon,
            Frequency::Annual,
            dc,
            1.0,
            0.98,
            0.02,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discounts,
        };
        let residual = bond.residual(reference, &snapshot).unwrap();
        assert!(
            residual.abs() < 1e-12,
            "residual with accrued must be zero when PV = clean + accrued, got {residual}",
        );
    }
}
