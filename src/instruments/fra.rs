// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Forward-rate agreement (FRA) instrument.
//!
//! A FRA fixes today, for a forward accrual period `[start, end]`, a
//! simply-compounded **forward rate**. By no-arbitrage in the single-curve
//! world, the quote is fair iff the discount-curve growth between `start`
//! and `end` matches the simple-compounding factor implied by the rate:
//!
//! ```text
//! D(start) / D(end) = 1 + rate * tau(start, end),
//! ```
//!
//! where `D` is the discount-factor curve and `tau(d1, d2)` is the year
//! fraction under the FRA's day-count convention. Equivalently, the
//! discount factor at the end of the forward period is determined by the
//! discount factor at the start via
//!
//! ```text
//! D(end) = D(start) / (1 + rate * tau).
//! ```
//!
//! This is the canonical FRA pricing identity used in the short-end of every
//! yield-curve bootstrap, and it is mathematically the same identity as the
//! money-market deposit's — only the period sits in the future rather than
//! beginning at the value date.
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §2. FRA pricing
//!   identity in the bootstrap context.
//! - Mercurio, F., "Interest Rates and The Credit Crunch: New Formulas and
//!   Market Models", Bloomberg Portfolio Research Paper No. 2010-01-FRONTIERS
//!   (2009), §2. Single-curve forward-rate identity reused under
//!   multi-curve as the projection-leg definition.
//! - ISDA, *2006 ISDA Definitions*, §4.6 ("Calculation Period") and §7.1
//!   ("Single-Period Floating Rate Notes"). Market-rate conventions for
//!   forward-rate agreements.

use crate::errors::{BootstrapError, TypeError};
use crate::types::{Date, Daycount};

use super::{CurveSnapshot, InstrumentLike};

/// A forward-rate agreement with a simply-compounded quoted forward rate.
///
/// Fields:
///
/// - `start` — forward start date: the beginning of the accrual period.
/// - `end` — forward end date: the period maturity.
/// - `rate` — the simply-compounded forward rate (decimal, e.g. `0.04` for
///   4%) over `[start, end]`.
/// - `daycount` — the day-count convention used to compute the accrual
///   `tau(start, end)`.
///
/// Constructed via [`Fra::new`], which validates the invariants:
/// `start < end` and `rate.is_finite()`. (Negative forward rates are
/// permitted — they have been quoted on EUR / CHF curves since the
/// post-2008 low-rate regime.)
///
/// # Examples
///
/// ```
/// use regit_curves::instruments::fra::Fra;
/// use regit_curves::types::{Date, Daycount};
///
/// let start = Date::from_ymd(2024, 7, 2).unwrap();
/// let end   = Date::from_ymd(2024, 10, 1).unwrap();
/// let f = Fra::new(start, end, 0.04, Daycount::Act360).unwrap();
/// // Implied D(end) given D(start) = 0.98 and tau = 91/360:
/// let tau = f.accrual().unwrap();
/// let d_end = f.implied_discount(0.98).unwrap();
/// assert!((d_end - 0.98 / (1.0 + 0.04 * tau)).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fra {
    /// Forward start date — the beginning of the accrual period.
    pub start: Date,
    /// Forward end date — the period maturity.
    pub end: Date,
    /// Simply-compounded forward rate (decimal) over `[start, end]`.
    pub rate: f64,
    /// Day-count convention used to compute the accrual `tau(start, end)`.
    pub daycount: Daycount,
}

impl Fra {
    /// Constructs a FRA after validating its invariants.
    ///
    /// Validation:
    ///
    /// - `rate` must be finite.
    /// - `start` must be strictly before `end` (the forward period must have
    ///   positive length).
    ///
    /// Negative forward rates are permitted; the formula `1 + rate * tau`
    /// remains strictly positive for any practical EUR / CHF / JPY quote.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `rate` is not finite or
    ///   if `start >= end`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::fra::Fra;
    /// use regit_curves::types::{Date, Daycount};
    /// use regit_curves::BootstrapError;
    ///
    /// let start = Date::from_ymd(2024, 7, 2).unwrap();
    /// let end   = Date::from_ymd(2024, 10, 1).unwrap();
    /// assert!(Fra::new(start, end, 0.04, Daycount::Act360).is_ok());
    /// // Inverted dates rejected:
    /// assert!(matches!(
    ///     Fra::new(end, start, 0.04, Daycount::Act360).unwrap_err(),
    ///     BootstrapError::InvalidInstrument { .. },
    /// ));
    /// // Zero-length forward period rejected:
    /// assert!(matches!(
    ///     Fra::new(start, start, 0.04, Daycount::Act360).unwrap_err(),
    ///     BootstrapError::InvalidInstrument { .. },
    /// ));
    /// ```
    pub fn new(
        start: Date,
        end: Date,
        rate: f64,
        daycount: Daycount,
    ) -> Result<Self, BootstrapError> {
        if !rate.is_finite() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "fra rate must be finite",
            });
        }
        if start.days_between(end) <= 0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "fra start must be strictly before end",
            });
        }
        Ok(Self {
            start,
            end,
            rate,
            daycount,
        })
    }

    /// Year fraction across the FRA's forward accrual period
    /// `[start, end]` under the FRA's day-count convention.
    ///
    /// # Errors
    ///
    /// - [`TypeError::InvalidTenor`] if the day-count convention requires a
    ///   calendar it has not been supplied (e.g. [`Daycount::Business252`]).
    /// - [`TypeError::NonPositiveRange`] only if the constructor was
    ///   bypassed (the constructor validates `start < end`).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::fra::Fra;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let start = Date::from_ymd(2024, 7, 2).unwrap();
    /// let end   = Date::from_ymd(2024, 10, 1).unwrap();
    /// let f = Fra::new(start, end, 0.04, Daycount::Act360).unwrap();
    /// assert!((f.accrual().unwrap() - 91.0 / 360.0).abs() < 1e-15);
    /// ```
    pub fn accrual(&self) -> Result<f64, TypeError> {
        self.daycount.year_fraction(self.start, self.end)
    }

    /// Returns the discount factor implied at the end of the forward period
    /// given the discount factor at the start:
    ///
    /// ```text
    /// D(end) = D(start) / (1 + rate * tau).
    /// ```
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `discount_at_start` is
    ///   non-positive or non-finite, or if `1 + rate * tau` is non-positive
    ///   (pathological deeply-negative-rate input).
    /// - [`BootstrapError::Type`] if the day-count query fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::fra::Fra;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let start = Date::from_ymd(2024, 7, 2).unwrap();
    /// let end   = Date::from_ymd(2024, 10, 1).unwrap();
    /// let f = Fra::new(start, end, 0.04, Daycount::Act360).unwrap();
    /// let d_end = f.implied_discount(0.98).unwrap();
    /// // 91/360 days at 4% simply-compounded:
    /// let expected = 0.98 / (1.0 + 0.04 * 91.0 / 360.0);
    /// assert!((d_end - expected).abs() < 1e-15);
    /// ```
    pub fn implied_discount(&self, discount_at_start: f64) -> Result<f64, BootstrapError> {
        if !discount_at_start.is_finite() || discount_at_start <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "discount factor at start must be finite and positive",
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
        Ok(discount_at_start / growth)
    }
}

impl InstrumentLike for Fra {
    #[inline]
    fn pillar(&self) -> Date {
        self.end
    }

    fn residual(
        &self,
        _reference_date: Date,
        curve: &CurveSnapshot<'_>,
    ) -> Result<f64, BootstrapError> {
        let tau = self.accrual()?;
        // Curve-side year fractions are taken under the curve's own day-count
        // (which is independent of the instrument's `daycount`).
        let t_start = curve
            .daycount
            .year_fraction(curve.reference_date, self.start)?;
        let t_end = curve
            .daycount
            .year_fraction(curve.reference_date, self.end)?;
        let d_start = curve
            .discount_at(t_start)
            .ok_or(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "curve snapshot is empty",
            })?;
        let d_end = curve
            .discount_at(t_end)
            .ok_or(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "curve snapshot is empty",
            })?;
        if d_end <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "non-positive discount factor in curve snapshot",
            });
        }
        // Residual: D(start) / D(end) - (1 + rate * tau).
        // Zero at the bootstrap solution.
        Ok(d_start / d_end - (1.0 + self.rate * tau))
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
    fn new_accepts_valid_fra() {
        let fra = Fra::new(d(2024, 7, 2), d(2024, 10, 1), 0.04, Daycount::Act360).unwrap();
        assert_eq!(fra.start, d(2024, 7, 2));
        assert_eq!(fra.end, d(2024, 10, 1));
        assert!((fra.rate - 0.04).abs() < 1e-15);
    }

    #[test]
    fn new_accepts_negative_rate() {
        // EUR / CHF / JPY money markets quote negative forward rates.
        let fra = Fra::new(d(2024, 7, 2), d(2024, 10, 1), -0.005, Daycount::Act360).unwrap();
        assert!(fra.rate < 0.0);
    }

    #[test]
    fn new_rejects_nan_rate() {
        let err = Fra::new(d(2024, 7, 2), d(2024, 10, 1), f64::NAN, Daycount::Act360).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_inf_rate() {
        let err = Fra::new(
            d(2024, 7, 2),
            d(2024, 10, 1),
            f64::INFINITY,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_neg_inf_rate() {
        let err = Fra::new(
            d(2024, 7, 2),
            d(2024, 10, 1),
            f64::NEG_INFINITY,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_inverted_dates() {
        let err = Fra::new(d(2024, 10, 1), d(2024, 7, 2), 0.04, Daycount::Act360).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_zero_length_period() {
        // Unlike a deposit (which can be same-day), a FRA must have a
        // strictly positive forward accrual period.
        let err = Fra::new(d(2024, 7, 2), d(2024, 7, 2), 0.04, Daycount::Act360).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    // ─── Year-fraction helper ────────────────────────────────────────────

    #[test]
    fn accrual_act360_matches_91_days() {
        // 2024-07-02 -> 2024-10-01 is exactly 91 days under Act/360.
        // This mirrors the ISDA §4.16(e) Act/360 worked-example arithmetic
        // (RESEARCH.md §2.3 records the canonical Act/360 case as
        // 182/360 over a six-month period; here we use a three-month
        // forward period for the FRA equivalent).
        let fra = Fra::new(d(2024, 7, 2), d(2024, 10, 1), 0.04, Daycount::Act360).unwrap();
        let tau = fra.accrual().unwrap();
        assert!((tau - 91.0 / 360.0).abs() < 1e-15);
    }

    #[test]
    fn accrual_propagates_business252_error() {
        let fra = Fra::new(d(2024, 7, 2), d(2024, 10, 1), 0.04, Daycount::Business252).unwrap();
        let err = fra.accrual().unwrap_err();
        assert!(matches!(err, TypeError::InvalidTenor { .. }));
    }

    // ─── Pricing identity: implied_discount ──────────────────────────────

    #[test]
    fn implied_discount_basic_formula() {
        // Three-month FRA at 4% on Act/360 over [Jul 2, Oct 1].
        let fra = Fra::new(d(2024, 7, 2), d(2024, 10, 1), 0.04, Daycount::Act360).unwrap();
        let d_end = fra.implied_discount(0.98).unwrap();
        let tau = 91.0_f64 / 360.0;
        let expected = 0.98 / (1.0 + 0.04 * tau);
        assert!((d_end - expected).abs() < 1e-15);
    }

    #[test]
    fn implied_discount_with_unit_d_start_equals_simple_discount() {
        // If D(start) = 1.0 the FRA identity collapses to the deposit's.
        let fra = Fra::new(d(2024, 7, 2), d(2024, 10, 1), 0.04, Daycount::Act360).unwrap();
        let tau = fra.accrual().unwrap();
        let d_end = fra.implied_discount(1.0).unwrap();
        // Sanity check: NOT equal to exp(-r * tau) (continuous compounding).
        let cont = (-0.04_f64 * tau).exp();
        assert!((d_end - cont).abs() > 1e-9);
        assert!((d_end - 1.0 / (1.0 + 0.04 * tau)).abs() < 1e-15);
    }

    #[test]
    fn implied_discount_rejects_non_finite_d_start() {
        let fra = Fra::new(d(2024, 7, 2), d(2024, 10, 1), 0.04, Daycount::Act360).unwrap();
        assert!(matches!(
            fra.implied_discount(f64::NAN).unwrap_err(),
            BootstrapError::InvalidInstrument { .. },
        ));
        assert!(matches!(
            fra.implied_discount(f64::INFINITY).unwrap_err(),
            BootstrapError::InvalidInstrument { .. },
        ));
    }

    #[test]
    fn implied_discount_rejects_non_positive_d_start() {
        let fra = Fra::new(d(2024, 7, 2), d(2024, 10, 1), 0.04, Daycount::Act360).unwrap();
        assert!(matches!(
            fra.implied_discount(0.0).unwrap_err(),
            BootstrapError::InvalidInstrument { .. },
        ));
        assert!(matches!(
            fra.implied_discount(-0.5).unwrap_err(),
            BootstrapError::InvalidInstrument { .. },
        ));
    }

    #[test]
    fn implied_discount_rejects_non_positive_growth() {
        // Pathological deeply-negative rate where (1 + r*tau) <= 0.
        // tau = 91/360 ≈ 0.2528; r = -10 gives 1 + (-10)(0.2528) ≈ -1.528.
        let fra = Fra::new(d(2024, 7, 2), d(2024, 10, 1), -10.0, Daycount::Act360).unwrap();
        let err = fra.implied_discount(1.0).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    // ─── Residual against a flat curve ───────────────────────────────────

    /// Builds a hand-rolled flat continuously-compounded discount curve
    /// `D(t) = exp(-r * t)` evaluated on a regular t-grid. The curve is
    /// expressed in year fractions from `reference_date` under the supplied
    /// `daycount`.
    fn flat_curve(reference_date: Date, daycount: Daycount, r: f64) -> (Vec<f64>, Vec<f64>) {
        // Cover up to 30 years at quarterly resolution. The pillar dates
        // used in the residual test fall on the grid (multiples of 91 days
        // from `reference_date`) so the `discount_at` lookup hits exact
        // knots — log-linear-on-D is exact between flat-z knots anyway.
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
    fn fra_residual_is_zero_on_flat_curve_with_implied_rate() {
        // The flat continuously-compounded curve at r_c implies a FRA par
        // rate of r_simple = (exp(r_c * tau) - 1) / tau over any forward
        // period of length tau (the same identity as the deposit, taken
        // over [start, end] rather than [reference, payment]). Feed that
        // rate back to the FRA; the residual must be zero (to round-off).
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.04_f64;
        let (times, discounts) = flat_curve(reference, daycount, r_c);

        // Six-month forward (start = reference + 182d) to nine-month
        // forward (end = reference + 273d), both falling on the
        // quarterly grid (182 = 2*91, 273 = 3*91).
        let start = Date::from_serial(reference.serial() + 2 * 91);
        let end = Date::from_serial(reference.serial() + 3 * 91);
        let tau = daycount.year_fraction(start, end).unwrap();
        let r_simple = (r_c * tau).exp_m1() / tau;

        let fra = Fra::new(start, end, r_simple, daycount).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };
        let residual = fra.residual(reference, &snapshot).unwrap();
        // The curve table is exact at the grid points and log-linear-on-D
        // is exact between flat-z knots, so the residual is bounded only
        // by floating-point round-off.
        assert!(
            residual.abs() < 1e-12,
            "residual on flat curve must be zero to 1e-12, got {residual}",
        );
    }

    #[test]
    fn fra_residual_sign_responds_to_rate_perturbation() {
        // A rate slightly above par implies the curve under-discounts the
        // forward leg -> residual `D(start)/D(end) - (1 + r*tau)` is
        // negative (the curve growth factor is below the FRA's quoted
        // growth factor).
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.04_f64;
        let (times, discounts) = flat_curve(reference, daycount, r_c);

        let start = Date::from_serial(reference.serial() + 2 * 91);
        let end = Date::from_serial(reference.serial() + 3 * 91);
        let tau = daycount.year_fraction(start, end).unwrap();
        let r_simple = (r_c * tau).exp_m1() / tau;

        // Over-quote the FRA by 50bp -> residual goes negative.
        let fra = Fra::new(start, end, r_simple + 0.005, daycount).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };
        let residual = fra.residual(reference, &snapshot).unwrap();
        assert!(residual < -1e-6);
    }

    #[test]
    fn fra_residual_errors_on_empty_curve_snapshot() {
        let reference = d(2024, 1, 2);
        let fra = Fra::new(d(2024, 7, 2), d(2024, 10, 1), 0.04, Daycount::Act360).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: Daycount::Act360,
            times: &[],
            discounts: &[],
        };
        let err = fra.residual(reference, &snapshot).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn fra_residual_propagates_business252_error() {
        // Curve day-count is Act/360 (fine), but the FRA's own day-count
        // is Business252 — the accrual query inside `residual` must
        // surface that error rather than producing a silent NaN.
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let (times, discounts) = flat_curve(reference, daycount, 0.04);
        let fra = Fra::new(d(2024, 7, 2), d(2024, 10, 1), 0.04, Daycount::Business252).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };
        let err = fra.residual(reference, &snapshot).unwrap_err();
        assert!(matches!(err, BootstrapError::Type(_)));
    }

    #[test]
    fn fra_pillar_is_end_date() {
        let fra = Fra::new(d(2024, 7, 2), d(2024, 10, 1), 0.04, Daycount::Act360).unwrap();
        assert_eq!(fra.pillar(), d(2024, 10, 1));
    }

    // ─── Round-trip identity check ───────────────────────────────────────

    #[test]
    fn fra_discount_roundtrip_through_growth_factor() {
        // If we know D(start), implied_discount produces D(end) such that
        // D(start)/D(end) == 1 + r*tau exactly.
        let fra = Fra::new(d(2024, 7, 2), d(2024, 10, 1), 0.04, Daycount::Act360).unwrap();
        let d_start = 0.9876;
        let d_end = fra.implied_discount(d_start).unwrap();
        let tau = fra.accrual().unwrap();
        assert!((d_start / d_end - (1.0 + 0.04 * tau)).abs() < 1e-15);
    }

    // ─── Sanity check on the numerical-target FRA from WORKING.md ────────

    #[test]
    fn fra_par_rate_target_value_just_above_4_pct() {
        // The pass spec asks for the implied simple par rate of a 91/360
        // forward period against a flat continuously-compounded r_c = 4%
        // curve. r_simple = (exp(0.04 * 91/360) - 1) / (91/360) and the
        // convexity gap pushes r_simple a hair above 0.04 (since
        // (e^x - 1) / x > 1 for x > 0).
        let tau = 91.0_f64 / 360.0;
        let r_simple = (0.04_f64 * tau).exp_m1() / tau;
        // Identity: r_simple * tau == exp_m1(r_c * tau).
        assert!((r_simple * tau - (0.04_f64 * tau).exp_m1()).abs() < 1e-18);
        // Sanity: r_simple sits strictly above r_c by the convexity gap,
        // which is O(r_c^2 * tau / 2) for small r_c.
        assert!(r_simple > 0.04);
        assert!(r_simple < 0.041);
    }
}
