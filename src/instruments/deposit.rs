// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Money-market deposit instrument.
//!
//! A deposit is a single cash flow: cash leaves at the **fixing** (value)
//! date and is repaid at the **payment** (maturity) date, accruing at a
//! quoted **simply-compounded** money-market rate. By no-arbitrage in the
//! single-curve world, the repayment cash flow is fair iff
//!
//! ```text
//! D(fixing) / D(payment) = 1 + rate * tau(fixing, payment),
//! ```
//!
//! where `D` is the discount-factor curve and `tau(d1, d2)` is the year
//! fraction under the deposit's day-count convention. Equivalently, the
//! discount factor at maturity is determined by the discount factor at
//! fixing via
//!
//! ```text
//! D(payment) = D(fixing) / (1 + rate * tau).
//! ```
//!
//! This is the canonical deposit pricing identity used in the short-end of
//! every yield-curve bootstrap.
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §2. Deposit / FRA
//!   pricing identities in the bootstrap context.
//! - ISDA, *2006 ISDA Definitions*, §4.6 ("Calculation Period") and §7.1
//!   ("Single-Period Floating Rate Notes"). Market-rate conventions for
//!   money-market deposits.

use crate::errors::{BootstrapError, TypeError};
use crate::types::{Date, Daycount};

use super::{CurveSnapshot, InstrumentLike};

/// A money-market deposit with a simply-compounded quoted rate.
///
/// Fields:
///
/// - `fixing` — value date: the day cash leaves.
/// - `payment` — maturity / repayment date.
/// - `rate` — the simply-compounded money-market rate (decimal, e.g. `0.05`
///   for 5%).
/// - `daycount` — the day-count convention used to compute the accrual `tau`.
///
/// Constructed via [`Deposit::new`], which validates the invariants:
/// `fixing <= payment` and `rate.is_finite()`. (Negative rates are
/// permitted — they are routinely quoted on EUR / CHF money markets.)
///
/// # Examples
///
/// ```
/// use regit_curves::instruments::Deposit;
/// use regit_curves::types::{Date, Daycount};
///
/// let fixing  = Date::from_ymd(2024, 1, 2).unwrap();
/// let payment = Date::from_ymd(2024, 4, 2).unwrap();
/// let d = Deposit::new(fixing, payment, 0.05, Daycount::Act360).unwrap();
/// // Implied D(payment) given D(fixing) = 1.0 and tau = 91/360:
/// let tau = d.accrual().unwrap();
/// let d_payment = d.implied_discount(1.0).unwrap();
/// assert!((d_payment - 1.0 / (1.0 + 0.05 * tau)).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deposit {
    /// Value date — the day cash leaves.
    pub fixing: Date,
    /// Maturity / repayment date.
    pub payment: Date,
    /// Simply-compounded money-market rate (decimal).
    pub rate: f64,
    /// Day-count convention used to compute the accrual `tau`.
    pub daycount: Daycount,
}

impl Deposit {
    /// Constructs a deposit after validating its invariants.
    ///
    /// Validation:
    ///
    /// - `rate` must be finite.
    /// - `fixing` must not be strictly after `payment`.
    ///
    /// Negative rates are permitted; the formula `1 + rate * tau` remains
    /// strictly positive for any practical EUR / CHF / JPY money-market
    /// quote.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `rate` is not finite or
    ///   if `fixing > payment`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::Deposit;
    /// use regit_curves::types::{Date, Daycount};
    /// use regit_curves::BootstrapError;
    ///
    /// let fixing  = Date::from_ymd(2024, 1, 2).unwrap();
    /// let payment = Date::from_ymd(2024, 4, 2).unwrap();
    /// assert!(Deposit::new(fixing, payment, 0.05, Daycount::Act360).is_ok());
    /// // Inverted dates rejected:
    /// assert!(matches!(
    ///     Deposit::new(payment, fixing, 0.05, Daycount::Act360).unwrap_err(),
    ///     BootstrapError::InvalidInstrument { .. },
    /// ));
    /// ```
    pub fn new(
        fixing: Date,
        payment: Date,
        rate: f64,
        daycount: Daycount,
    ) -> Result<Self, BootstrapError> {
        if !rate.is_finite() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "deposit rate must be finite",
            });
        }
        if fixing.days_between(payment) < 0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "deposit fixing must be on or before payment",
            });
        }
        Ok(Self {
            fixing,
            payment,
            rate,
            daycount,
        })
    }

    /// Year fraction from `reference_date` to [`Deposit::payment`] under the
    /// deposit's day-count convention.
    ///
    /// # Errors
    ///
    /// - [`TypeError::NonPositiveRange`] if `reference_date > payment`.
    /// - [`TypeError::InvalidTenor`] if the day-count convention requires a
    ///   calendar it has not been supplied (e.g. [`Daycount::Business252`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::Deposit;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let fixing  = Date::from_ymd(2024, 1, 2).unwrap();
    /// let payment = Date::from_ymd(2024, 4, 2).unwrap();
    /// let d = Deposit::new(fixing, payment, 0.05, Daycount::Act360).unwrap();
    /// let tau = d.year_fraction_to_maturity(fixing).unwrap();
    /// assert!((tau - 91.0 / 360.0).abs() < 1e-15);
    /// ```
    pub fn year_fraction_to_maturity(&self, reference_date: Date) -> Result<f64, TypeError> {
        self.daycount.year_fraction(reference_date, self.payment)
    }

    /// Year fraction across the deposit's accrual period
    /// `[fixing, payment]`.
    ///
    /// # Errors
    ///
    /// - [`TypeError::InvalidTenor`] if the day-count convention requires a
    ///   calendar it has not been supplied (e.g. [`Daycount::Business252`]).
    /// - [`TypeError::NonPositiveRange`] only if the constructor was bypassed
    ///   (the constructor validates `fixing <= payment`).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::Deposit;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let fixing  = Date::from_ymd(2024, 1, 2).unwrap();
    /// let payment = Date::from_ymd(2024, 4, 2).unwrap();
    /// let d = Deposit::new(fixing, payment, 0.05, Daycount::Act360).unwrap();
    /// assert!((d.accrual().unwrap() - 91.0 / 360.0).abs() < 1e-15);
    /// ```
    pub fn accrual(&self) -> Result<f64, TypeError> {
        self.daycount.year_fraction(self.fixing, self.payment)
    }

    /// Returns the discount factor implied at the payment date given the
    /// discount factor at the fixing date:
    ///
    /// ```text
    /// D(payment) = D(fixing) / (1 + rate * tau).
    /// ```
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `discount_at_fixing` is
    ///   non-positive or non-finite, or if `1 + rate * tau` is non-positive
    ///   (pathological deeply-negative-rate input).
    /// - [`BootstrapError::Type`] if the day-count query fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::Deposit;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let fixing  = Date::from_ymd(2024, 1, 2).unwrap();
    /// let payment = Date::from_ymd(2024, 4, 2).unwrap();
    /// let d = Deposit::new(fixing, payment, 0.05, Daycount::Act360).unwrap();
    /// let d_pay = d.implied_discount(1.0).unwrap();
    /// // 91/360 days at 5% simply-compounded:
    /// let expected = 1.0 / (1.0 + 0.05 * 91.0 / 360.0);
    /// assert!((d_pay - expected).abs() < 1e-15);
    /// ```
    pub fn implied_discount(&self, discount_at_fixing: f64) -> Result<f64, BootstrapError> {
        if !discount_at_fixing.is_finite() || discount_at_fixing <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "discount factor at fixing must be finite and positive",
            });
        }
        let tau = self.accrual()?;
        let growth = 1.0 + self.rate * tau;
        if !growth.is_finite() || growth <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "non-positive accrual factor (1 + rate * tau)",
            });
        }
        Ok(discount_at_fixing / growth)
    }
}

impl InstrumentLike for Deposit {
    #[inline]
    fn pillar(&self) -> Date {
        self.payment
    }

    fn residual(
        &self,
        _reference_date: Date,
        curve: &CurveSnapshot<'_>,
    ) -> Result<f64, BootstrapError> {
        let tau = self.accrual()?;
        // Curve-side year fractions are taken under the curve's own day-count
        // (which is independent of the instrument's `daycount`).
        let t_fixing = curve
            .daycount
            .year_fraction(curve.reference_date, self.fixing)?;
        let t_payment = curve
            .daycount
            .year_fraction(curve.reference_date, self.payment)?;
        let d_fixing = curve
            .discount_at(t_fixing)
            .ok_or(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "curve snapshot is empty",
            })?;
        let d_payment = curve
            .discount_at(t_payment)
            .ok_or(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "curve snapshot is empty",
            })?;
        if d_payment <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "non-positive discount factor in curve snapshot",
            });
        }
        // Residual: D(fixing) / D(payment) - (1 + rate * tau).
        // Zero at the bootstrap solution.
        Ok(d_fixing / d_payment - (1.0 + self.rate * tau))
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
    fn new_accepts_valid_deposit() {
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), 0.05, Daycount::Act360).unwrap();
        assert_eq!(dep.fixing, d(2024, 1, 2));
        assert_eq!(dep.payment, d(2024, 4, 2));
        assert!((dep.rate - 0.05).abs() < 1e-15);
    }

    #[test]
    fn new_accepts_negative_rate() {
        // EUR / CHF / JPY money markets quote negative rates routinely.
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), -0.005, Daycount::Act360).unwrap();
        assert!(dep.rate < 0.0);
    }

    #[test]
    fn new_accepts_zero_accrual_at_fixing_equals_payment() {
        // Degenerate but technically valid: same-day deposit (zero accrual).
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 1, 2), 0.05, Daycount::Act360).unwrap();
        assert!((dep.accrual().unwrap() - 0.0).abs() < 1e-15);
    }

    #[test]
    fn new_rejects_nan_rate() {
        let err =
            Deposit::new(d(2024, 1, 2), d(2024, 4, 2), f64::NAN, Daycount::Act360).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_inf_rate() {
        let err = Deposit::new(
            d(2024, 1, 2),
            d(2024, 4, 2),
            f64::INFINITY,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_inverted_dates() {
        let err = Deposit::new(d(2024, 4, 2), d(2024, 1, 2), 0.05, Daycount::Act360).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    // ─── Year-fraction helpers ───────────────────────────────────────────

    #[test]
    fn accrual_matches_isda_worked_example() {
        // ISDA §4.16(e) worked example: 2003-11-01 -> 2004-05-01 Act/360.
        let dep = Deposit::new(d(2003, 11, 1), d(2004, 5, 1), 0.04, Daycount::Act360).unwrap();
        let tau = dep.accrual().unwrap();
        assert!((tau - 182.0 / 360.0).abs() < 1e-15);
    }

    #[test]
    fn year_fraction_to_maturity_matches_daycount() {
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), 0.05, Daycount::Act360).unwrap();
        let tau = dep.year_fraction_to_maturity(d(2024, 1, 2)).unwrap();
        assert!((tau - 91.0 / 360.0).abs() < 1e-15);
    }

    #[test]
    fn year_fraction_to_maturity_propagates_business252_error() {
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), 0.05, Daycount::Business252).unwrap();
        let err = dep.year_fraction_to_maturity(d(2024, 1, 2)).unwrap_err();
        assert!(matches!(err, TypeError::InvalidTenor { .. }));
    }

    // ─── Pricing identity: implied_discount ──────────────────────────────

    #[test]
    fn implied_discount_basic_formula() {
        // Three-month deposit at 5% on Act/360 between Jan 2 and Apr 2.
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), 0.05, Daycount::Act360).unwrap();
        let d_pay = dep.implied_discount(1.0).unwrap();
        let tau = 91.0_f64 / 360.0;
        let expected = 1.0 / (1.0 + 0.05 * tau);
        assert!((d_pay - expected).abs() < 1e-15);
    }

    #[test]
    fn implied_discount_matches_flat_curve_value() {
        // Build a fake flat continuously-compounded curve at r = 0.05; the
        // deposit pricing identity does NOT use continuous compounding —
        // it uses the simply-compounded `1 + r * tau`. So the implied
        // discount factor is exactly that, not exp(-r * tau).
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), 0.05, Daycount::Act360).unwrap();
        let tau = dep.accrual().unwrap();
        let d_pay = dep.implied_discount(1.0).unwrap();
        let cont_continuous = (-0.05_f64 * tau).exp();
        // These should NOT agree exactly — the difference is the convexity
        // gap between simple and continuous compounding. Sanity check:
        assert!((d_pay - cont_continuous).abs() > 1e-9);
        assert!((d_pay - 1.0 / (1.0 + 0.05 * tau)).abs() < 1e-15);
    }

    #[test]
    fn implied_discount_rejects_non_finite_d_fixing() {
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), 0.05, Daycount::Act360).unwrap();
        assert!(matches!(
            dep.implied_discount(f64::NAN).unwrap_err(),
            BootstrapError::InvalidInstrument { .. },
        ));
        assert!(matches!(
            dep.implied_discount(f64::INFINITY).unwrap_err(),
            BootstrapError::InvalidInstrument { .. },
        ));
    }

    #[test]
    fn implied_discount_rejects_non_positive_d_fixing() {
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), 0.05, Daycount::Act360).unwrap();
        assert!(matches!(
            dep.implied_discount(0.0).unwrap_err(),
            BootstrapError::InvalidInstrument { .. },
        ));
        assert!(matches!(
            dep.implied_discount(-0.5).unwrap_err(),
            BootstrapError::InvalidInstrument { .. },
        ));
    }

    #[test]
    fn implied_discount_rejects_non_positive_growth() {
        // Pathological deeply-negative rate where (1 + r*tau) <= 0.
        // For tau = 91/360 ≈ 0.2528, r = -10 gives 1 + (-10)(0.2528) ≈ -1.528.
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), -10.0, Daycount::Act360).unwrap();
        let err = dep.implied_discount(1.0).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    // ─── Residual against a flat curve ───────────────────────────────────

    /// Builds a hand-rolled flat continuously-compounded discount curve
    /// `D(t) = exp(-r * t)` evaluated on a regular t-grid. The curve is
    /// expressed in year fractions from `reference_date` under the supplied
    /// `daycount`.
    fn flat_curve(reference_date: Date, daycount: Daycount, r: f64) -> (Vec<f64>, Vec<f64>) {
        // Cover up to 30 years at quarterly resolution. The pillar dates
        // used in the residual test fall inside this grid so the
        // `discount_at` interpolation is interior, not extrapolated.
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

    #[test]
    fn deposit_residual_is_zero_on_flat_curve_with_implied_rate() {
        // The flat continuously-compounded curve at r_c implies a deposit
        // par rate of r_simple = (exp(r_c * tau) - 1) / tau over any period
        // of length tau. Feed that rate back to the deposit; the residual
        // must be zero (to round-off).
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.05_f64;
        let (times, discounts) = flat_curve(reference, daycount, r_c);

        let fixing = d(2024, 1, 2); // == reference
        let payment = d(2024, 4, 2); // 91 days later under Act/360
        let tau = daycount.year_fraction(fixing, payment).unwrap();
        let r_simple = (r_c * tau).exp_m1() / tau;

        let dep = Deposit::new(fixing, payment, r_simple, daycount).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };
        let residual = dep.residual(reference, &snapshot).unwrap();
        // The curve table is exact at the grid points and log-linear-on-D
        // is exact between them when the underlying curve is flat-z (i.e.
        // exp(-r*t)), since log of exp(-r*t) is linear in t. So the
        // residual is bounded by round-off only.
        assert!(
            residual.abs() < 1e-12,
            "residual on flat curve must be zero to 1e-12, got {residual}",
        );
    }

    #[test]
    fn deposit_residual_sign_responds_to_rate_perturbation() {
        // A rate slightly above par implies the curve over-discounts -> the
        // residual `D(fixing)/D(payment) - (1 + r*tau)` becomes negative.
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.05_f64;
        let (times, discounts) = flat_curve(reference, daycount, r_c);

        let fixing = d(2024, 1, 2);
        let payment = d(2024, 4, 2);
        let tau = daycount.year_fraction(fixing, payment).unwrap();
        let r_simple = (r_c * tau).exp_m1() / tau;

        // Over-quote the deposit by 50bp -> residual goes negative.
        let dep = Deposit::new(fixing, payment, r_simple + 0.005, daycount).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };
        let residual = dep.residual(reference, &snapshot).unwrap();
        assert!(residual < -1e-6);
    }

    #[test]
    fn deposit_residual_errors_on_empty_curve_snapshot() {
        let reference = d(2024, 1, 2);
        let dep = Deposit::new(reference, d(2024, 4, 2), 0.05, Daycount::Act360).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: Daycount::Act360,
            times: &[],
            discounts: &[],
        };
        let err = dep.residual(reference, &snapshot).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn deposit_pillar_is_payment_date() {
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), 0.05, Daycount::Act360).unwrap();
        assert_eq!(dep.pillar(), d(2024, 4, 2));
    }

    // ─── Round-trip identity check ───────────────────────────────────────

    #[test]
    fn deposit_discount_roundtrip_through_growth_factor() {
        // If we know D(fixing), implied_discount produces D(payment) such
        // that D(fixing)/D(payment) == 1 + r*tau exactly.
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), 0.05, Daycount::Act360).unwrap();
        let d_fix = 0.9876;
        let d_pay = dep.implied_discount(d_fix).unwrap();
        let tau = dep.accrual().unwrap();
        assert!((d_fix / d_pay - (1.0 + 0.05 * tau)).abs() < 1e-15);
    }
}
