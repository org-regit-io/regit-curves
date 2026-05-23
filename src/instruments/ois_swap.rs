// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Overnight-indexed (OIS) swap instrument.
//!
//! An OIS exchanges, on each payment period, a fixed coupon against the
//! daily-compounded realisation of an overnight rate over the same period. By
//! no-arbitrage in the OIS-discounted world the expected realised compounded
//! overnight rate over `[t_{i-1}, t_i]` is the simply-compounded OIS forward
//! rate over the same interval (Mercurio 2009, §3.2). Because OIS is a
//! *single-curve* instrument — the OIS curve both projects the float leg's
//! forwards and discounts every cash flow — the float-leg present value
//! telescopes exactly:
//!
//! ```text
//! PV_float = sum_i D(t_i) * tau_i * F_i
//!          = sum_i D(t_i) * (D(t_{i-1}) / D(t_i) - 1)
//!          = sum_i (D(t_{i-1}) - D(t_i))
//!          = D(t_0) - D(t_N),
//! ```
//!
//! where `F_i = (D(t_{i-1}) / D(t_i) - 1) / tau_i` is the OIS forward rate
//! over the period under the swap's day-count and `t_0` is the swap start,
//! `t_N` the swap maturity. The fixed-leg present value is
//!
//! ```text
//! PV_fixed = rate * sum_i tau_i * D(t_i).
//! ```
//!
//! Equating the two legs yields the par-OIS-rate identity used by the
//! bootstrap engine:
//!
//! ```text
//! rate * sum_i tau_i * D(t_i) = D(t_0) - D(t_N).
//! ```
//!
//! This is the same single-curve form as a vanilla fixed-float swap. The
//! identity is what pins the OIS discount curve to its quoted par OIS rates.
//!
//! # References
//!
//! - Mercurio, F., *Interest Rates and The Credit Crunch: New Formulas and
//!   Market Models*, Bloomberg Portfolio Research Paper No. 2010-01-FRONTIERS
//!   (Feb 2009), §3.2. Float-leg telescoping argument in the OIS-discounted
//!   world.
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §2. Single-curve
//!   par-swap-rate equation.

use crate::errors::BootstrapError;
use crate::types::{Date, Daycount, Frequency};

use super::schedule::SwapSchedule;
use super::{CurveSnapshot, InstrumentLike};

/// A par-quoted overnight-indexed swap.
///
/// Fields:
///
/// - `start` — swap effective date (`t_0`).
/// - `maturity` — swap maturity (`t_N` = the last payment date).
/// - `rate` — par OIS rate (decimal, e.g. `0.03` for 3%). Negative rates
///   are permitted — they have been quoted on EUR / CHF OIS markets.
/// - `freq` — payment frequency (typically [`Frequency::Annual`] for tenors
///   above one year; [`Frequency::OnceAtMaturity`] is the market convention
///   for tenors of one year or less, where there is a single bullet payment
///   at maturity).
/// - `daycount` — day-count convention used to compute the period accruals
///   `tau_i` on both the fixed leg and (notionally) the float leg.
/// - `schedule` — the precomputed [`SwapSchedule`] of period boundary dates.
///
/// Constructed via [`OisSwap::new`] (which builds the regular schedule
/// internally) or [`OisSwap::with_schedule`] (which accepts a caller-built
/// schedule for irregular cases such as stub periods).
///
/// # Examples
///
/// ```
/// use regit_curves::instruments::OisSwap;
/// use regit_curves::types::{Date, Daycount, Frequency};
///
/// let start    = Date::from_ymd(2024, 1, 2).unwrap();
/// let maturity = Date::from_ymd(2029, 1, 2).unwrap();
/// // 5y annual OIS quoted at 3% on Act/360.
/// let swap = OisSwap::new(start, maturity, 0.03, Frequency::Annual, Daycount::Act360).unwrap();
/// assert_eq!(swap.schedule.len(), 5);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OisSwap {
    /// Swap effective date — `t_0`, the start of the first accrual period.
    pub start: Date,
    /// Swap maturity — `t_N`, the last payment date.
    pub maturity: Date,
    /// Par OIS rate (decimal).
    pub rate: f64,
    /// Payment frequency (used to build the regular schedule).
    pub freq: Frequency,
    /// Day-count convention used to compute period accruals on both legs.
    pub daycount: Daycount,
    /// Precomputed schedule of period boundary dates.
    pub schedule: SwapSchedule,
}

impl OisSwap {
    /// Constructs an OIS swap with a regular schedule generated from
    /// `(start, maturity, freq)`.
    ///
    /// Validation:
    ///
    /// - `rate` must be finite.
    /// - `start` must be strictly before `maturity`.
    /// - The regular schedule generator must succeed — see
    ///   [`SwapSchedule::from_regular`] for the regularity constraint.
    ///
    /// Negative rates are accepted (the par-OIS-rate identity is linear in
    /// `rate` and remains well-posed under any finite quote).
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `rate` is not finite, if
    ///   `start >= maturity`, or if [`SwapSchedule::from_regular`] returns an
    ///   error.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::OisSwap;
    /// use regit_curves::types::{Date, Daycount, Frequency};
    /// use regit_curves::BootstrapError;
    ///
    /// let start    = Date::from_ymd(2024, 1, 2).unwrap();
    /// let maturity = Date::from_ymd(2029, 1, 2).unwrap();
    /// assert!(
    ///     OisSwap::new(start, maturity, 0.03, Frequency::Annual, Daycount::Act360).is_ok()
    /// );
    /// // Inverted dates rejected:
    /// assert!(matches!(
    ///     OisSwap::new(maturity, start, 0.03, Frequency::Annual, Daycount::Act360).unwrap_err(),
    ///     BootstrapError::InvalidInstrument { .. },
    /// ));
    /// ```
    pub fn new(
        start: Date,
        maturity: Date,
        rate: f64,
        freq: Frequency,
        daycount: Daycount,
    ) -> Result<Self, BootstrapError> {
        if !rate.is_finite() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "OIS swap rate must be finite",
            });
        }
        if start.days_between(maturity) <= 0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "OIS swap start must precede maturity",
            });
        }
        let schedule = SwapSchedule::from_regular(start, maturity, freq)?;
        Ok(Self {
            start,
            maturity,
            rate,
            freq,
            daycount,
            schedule,
        })
    }

    /// Constructs an OIS swap from an already-built schedule.
    ///
    /// Use this entry point when the schedule is irregular (e.g. stub first
    /// or last period, or business-day-adjusted dates). The supplied schedule
    /// must agree with `(start, maturity)`: its first boundary date must
    /// equal `start` and its last must equal `maturity`.
    ///
    /// Validation matches [`OisSwap::new`]; additionally the schedule's
    /// endpoints are checked against `(start, maturity)`.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `rate` is not finite, if
    ///   `start >= maturity`, or if the schedule's start / maturity do not
    ///   match the supplied dates.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::{OisSwap, SwapSchedule};
    /// use regit_curves::types::{Date, Daycount, Frequency};
    ///
    /// let start    = Date::from_ymd(2024, 1, 2).unwrap();
    /// let maturity = Date::from_ymd(2026, 1, 2).unwrap();
    /// let sch = SwapSchedule::from_regular(start, maturity, Frequency::Annual).unwrap();
    /// let swap = OisSwap::with_schedule(
    ///     start, maturity, 0.03, Frequency::Annual, Daycount::Act360, sch,
    /// ).unwrap();
    /// assert_eq!(swap.schedule.len(), 2);
    /// ```
    pub fn with_schedule(
        start: Date,
        maturity: Date,
        rate: f64,
        freq: Frequency,
        daycount: Daycount,
        schedule: SwapSchedule,
    ) -> Result<Self, BootstrapError> {
        if !rate.is_finite() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "OIS swap rate must be finite",
            });
        }
        if start.days_between(maturity) <= 0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "OIS swap start must precede maturity",
            });
        }
        if schedule.start() != start || schedule.maturity() != maturity {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "schedule endpoints must match (start, maturity)",
            });
        }
        Ok(Self {
            start,
            maturity,
            rate,
            freq,
            daycount,
            schedule,
        })
    }

    /// Present value of the fixed leg:
    ///
    /// ```text
    /// PV_fixed = rate * sum_i tau_i * D(t_i),
    /// ```
    ///
    /// with `tau_i` the period accruals under [`OisSwap::daycount`] and
    /// `D(t_i)` the curve's discount factor at each period's payment date.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::Type`] if the day-count convention cannot produce
    ///   a year fraction (e.g. [`Daycount::Business252`] without a calendar).
    /// - [`BootstrapError::InvalidInstrument`] if the curve snapshot is empty
    ///   or returns a non-positive discount factor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use regit_curves::instruments::OisSwap;
    /// # use regit_curves::types::{Date, Daycount, Frequency};
    /// let start = Date::from_ymd(2024, 1, 2).unwrap();
    /// let maturity = Date::from_ymd(2025, 1, 2).unwrap();
    /// let swap = OisSwap::new(
    ///     start, maturity, 0.03, Frequency::Annual, Daycount::Act360,
    /// ).unwrap();
    /// assert_eq!(swap.rate, 0.03);
    /// ```
    pub(crate) fn fixed_leg_pv(
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
        Ok(self.rate * annuity)
    }

    /// Present value of the float leg under OIS discounting:
    ///
    /// ```text
    /// PV_float = D(t_0) - D(t_N),
    /// ```
    ///
    /// the telescoped sum of period forwards (see the module-level derivation
    /// citing Mercurio 2009 §3.2).
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::Type`] if the curve's day-count cannot produce a
    ///   year fraction for `start` or `maturity`.
    /// - [`BootstrapError::InvalidInstrument`] if the curve snapshot is empty
    ///   or returns a non-positive discount factor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use regit_curves::instruments::OisSwap;
    /// # use regit_curves::types::{Date, Daycount, Frequency};
    /// let start = Date::from_ymd(2024, 1, 2).unwrap();
    /// let maturity = Date::from_ymd(2025, 1, 2).unwrap();
    /// let swap = OisSwap::new(
    ///     start, maturity, 0.03, Frequency::Annual, Daycount::Act360,
    /// ).unwrap();
    /// assert_eq!(swap.start, start);
    /// assert_eq!(swap.maturity, maturity);
    /// ```
    pub(crate) fn float_leg_pv(
        &self,
        _reference_date: Date,
        curve: &CurveSnapshot<'_>,
    ) -> Result<f64, BootstrapError> {
        let t_start = curve
            .daycount
            .year_fraction(curve.reference_date, self.start)?;
        let t_maturity = curve
            .daycount
            .year_fraction(curve.reference_date, self.maturity)?;
        let d_start = curve
            .discount_at(t_start)
            .ok_or(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "curve snapshot is empty",
            })?;
        let d_maturity =
            curve
                .discount_at(t_maturity)
                .ok_or(BootstrapError::InvalidInstrument {
                    at_index: 0,
                    reason: "curve snapshot is empty",
                })?;
        if !d_start.is_finite() || d_start <= 0.0 || !d_maturity.is_finite() || d_maturity <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "non-positive discount factor in curve snapshot",
            });
        }
        Ok(d_start - d_maturity)
    }
}

impl InstrumentLike for OisSwap {
    #[inline]
    fn pillar(&self) -> Date {
        self.maturity
    }

    fn residual(
        &self,
        reference_date: Date,
        curve: &CurveSnapshot<'_>,
    ) -> Result<f64, BootstrapError> {
        // Residual: PV_fixed - PV_float. Zero at the bootstrap solution.
        // Equivalent to rate * sum_i tau_i * D(t_i) - (D(t_0) - D(t_N)).
        let fixed = self.fixed_leg_pv(reference_date, curve)?;
        let float = self.float_leg_pv(reference_date, curve)?;
        Ok(fixed - float)
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
    fn new_accepts_5y_annual_swap() {
        let start = d(2024, 1, 2);
        let maturity = d(2029, 1, 2);
        let swap =
            OisSwap::new(start, maturity, 0.03, Frequency::Annual, Daycount::Act360).unwrap();
        assert_eq!(swap.start, start);
        assert_eq!(swap.maturity, maturity);
        assert!((swap.rate - 0.03).abs() < 1e-15);
        assert_eq!(swap.freq, Frequency::Annual);
        assert_eq!(swap.daycount, Daycount::Act360);
        assert_eq!(swap.schedule.len(), 5);
    }

    #[test]
    fn new_accepts_6m_once_at_maturity() {
        // Short OIS: a single bullet payment at maturity.
        let start = d(2024, 1, 2);
        let maturity = d(2024, 7, 2);
        let swap = OisSwap::new(
            start,
            maturity,
            0.025,
            Frequency::OnceAtMaturity,
            Daycount::Act360,
        )
        .unwrap();
        assert_eq!(swap.schedule.len(), 1);
        assert_eq!(swap.schedule.period_start(0), start);
        assert_eq!(swap.schedule.period_end(0), maturity);
    }

    #[test]
    fn new_accepts_negative_rate() {
        // EUR / CHF OIS quoted negative is routine.
        let start = d(2024, 1, 2);
        let maturity = d(2026, 1, 2);
        let swap =
            OisSwap::new(start, maturity, -0.002, Frequency::Annual, Daycount::Act360).unwrap();
        assert!(swap.rate < 0.0);
    }

    #[test]
    fn new_rejects_nan_rate() {
        let err = OisSwap::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            f64::NAN,
            Frequency::Annual,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_inf_rate() {
        let err = OisSwap::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            f64::INFINITY,
            Frequency::Annual,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_inverted_dates() {
        let err = OisSwap::new(
            d(2029, 1, 2),
            d(2024, 1, 2),
            0.03,
            Frequency::Annual,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_equal_start_and_maturity() {
        let s = d(2024, 1, 2);
        let err = OisSwap::new(s, s, 0.03, Frequency::Annual, Daycount::Act360).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_propagates_irregular_schedule_error() {
        // 13 months at semi-annual cadence is not regular.
        let err = OisSwap::new(
            d(2024, 1, 2),
            d(2025, 2, 2),
            0.03,
            Frequency::SemiAnnual,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    // ─── with_schedule entry point ───────────────────────────────────────

    #[test]
    fn with_schedule_accepts_matching_schedule() {
        let start = d(2024, 1, 2);
        let maturity = d(2026, 1, 2);
        let sch = SwapSchedule::from_regular(start, maturity, Frequency::Annual).unwrap();
        let swap = OisSwap::with_schedule(
            start,
            maturity,
            0.03,
            Frequency::Annual,
            Daycount::Act360,
            sch,
        )
        .unwrap();
        assert_eq!(swap.schedule.len(), 2);
    }

    #[test]
    fn with_schedule_rejects_mismatched_endpoints() {
        let start = d(2024, 1, 2);
        let maturity = d(2026, 1, 2);
        let other = d(2027, 1, 2);
        let sch = SwapSchedule::from_regular(start, other, Frequency::Annual).unwrap();
        let err = OisSwap::with_schedule(
            start,
            maturity,
            0.03,
            Frequency::Annual,
            Daycount::Act360,
            sch,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn with_schedule_rejects_nan_rate() {
        let start = d(2024, 1, 2);
        let maturity = d(2026, 1, 2);
        let sch = SwapSchedule::from_regular(start, maturity, Frequency::Annual).unwrap();
        let err = OisSwap::with_schedule(
            start,
            maturity,
            f64::NAN,
            Frequency::Annual,
            Daycount::Act360,
            sch,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    // ─── Pillar accessor ─────────────────────────────────────────────────

    #[test]
    fn pillar_is_maturity_date() {
        let start = d(2024, 1, 2);
        let maturity = d(2029, 1, 2);
        let swap =
            OisSwap::new(start, maturity, 0.03, Frequency::Annual, Daycount::Act360).unwrap();
        assert_eq!(swap.pillar(), maturity);
    }

    // ─── Residual against a flat curve ───────────────────────────────────

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

    /// Computes the par OIS rate consistent with a flat continuously-
    /// compounded curve `D(t) = exp(-r_c * t)`:
    ///
    /// ```text
    /// r_par = (D(t_0) - D(t_N)) / sum_i tau_i * D(t_i).
    /// ```
    fn par_ois_rate_flat(swap: &OisSwap, reference: Date, r_c: f64) -> f64 {
        let dc = swap.daycount;
        let t0 = dc.year_fraction(reference, swap.start).unwrap();
        let tn = dc.year_fraction(reference, swap.maturity).unwrap();
        let numerator = (-r_c * t0).exp() - (-r_c * tn).exp();
        let mut annuity = 0.0_f64;
        for i in 0..swap.schedule.len() {
            let s = swap.schedule.period_start(i);
            let e = swap.schedule.period_end(i);
            let tau = dc.year_fraction(s, e).unwrap();
            let t_pay = dc.year_fraction(reference, e).unwrap();
            annuity += tau * (-r_c * t_pay).exp();
        }
        numerator / annuity
    }

    #[test]
    fn par_ois_rate_zeroes_residual_5y_annual() {
        // Numerical test target from the spec: flat r_c = 3%, 5y annual on
        // Act/360. Build the par rate analytically, feed it to the swap,
        // assert residual == 0 to 1e-10.
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.03_f64;
        let start = reference;
        let maturity = d(2029, 1, 2);
        let (times, discounts) = flat_curve(reference, daycount, r_c);

        // Build a placeholder swap to derive the par rate from the schedule.
        let placeholder = OisSwap::new(start, maturity, 0.0, Frequency::Annual, daycount).unwrap();
        let r_par = par_ois_rate_flat(&placeholder, reference, r_c);

        let swap = OisSwap::new(start, maturity, r_par, Frequency::Annual, daycount).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };
        let residual = swap.residual(reference, &snapshot).unwrap();
        assert!(
            residual.abs() < 1e-10,
            "OIS residual on flat curve must be zero to 1e-10, got {residual}",
        );
    }

    #[test]
    fn par_ois_rate_zeroes_residual_6m_once_at_maturity() {
        // Short OIS: single bullet payment.
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.025_f64;
        let start = reference;
        let maturity = d(2024, 7, 2);
        let (times, discounts) = flat_curve(reference, daycount, r_c);

        let placeholder =
            OisSwap::new(start, maturity, 0.0, Frequency::OnceAtMaturity, daycount).unwrap();
        let r_par = par_ois_rate_flat(&placeholder, reference, r_c);

        let swap =
            OisSwap::new(start, maturity, r_par, Frequency::OnceAtMaturity, daycount).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };
        let residual = swap.residual(reference, &snapshot).unwrap();
        assert!(
            residual.abs() < 1e-10,
            "short-OIS residual on flat curve must be zero to 1e-10, got {residual}",
        );
    }

    #[test]
    fn residual_sign_responds_to_rate_perturbation() {
        // Over-quoting the par rate makes the fixed leg too rich, so the
        // residual (fixed - float) becomes positive.
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.03_f64;
        let start = reference;
        let maturity = d(2029, 1, 2);
        let (times, discounts) = flat_curve(reference, daycount, r_c);

        let placeholder = OisSwap::new(start, maturity, 0.0, Frequency::Annual, daycount).unwrap();
        let r_par = par_ois_rate_flat(&placeholder, reference, r_c);

        let swap =
            OisSwap::new(start, maturity, r_par + 0.005, Frequency::Annual, daycount).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };
        let residual = swap.residual(reference, &snapshot).unwrap();
        assert!(residual > 1e-6);
    }

    #[test]
    fn fixed_and_float_legs_match_at_par() {
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.03_f64;
        let start = reference;
        let maturity = d(2029, 1, 2);
        let (times, discounts) = flat_curve(reference, daycount, r_c);

        let placeholder = OisSwap::new(start, maturity, 0.0, Frequency::Annual, daycount).unwrap();
        let r_par = par_ois_rate_flat(&placeholder, reference, r_c);
        let swap = OisSwap::new(start, maturity, r_par, Frequency::Annual, daycount).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };
        let fixed = swap.fixed_leg_pv(reference, &snapshot).unwrap();
        let float = swap.float_leg_pv(reference, &snapshot).unwrap();
        assert!((fixed - float).abs() < 1e-10);
    }

    #[test]
    fn float_leg_telescopes_to_d_start_minus_d_maturity() {
        // PV_float = D(t_0) - D(t_N). Verify against the flat curve directly.
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.03_f64;
        let start = reference;
        let maturity = d(2029, 1, 2);
        let (times, discounts) = flat_curve(reference, daycount, r_c);

        let swap = OisSwap::new(start, maturity, 0.03, Frequency::Annual, daycount).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };
        let float = swap.float_leg_pv(reference, &snapshot).unwrap();
        let t0 = daycount.year_fraction(reference, start).unwrap();
        let tn = daycount.year_fraction(reference, maturity).unwrap();
        let expected = (-r_c * t0).exp() - (-r_c * tn).exp();
        assert!((float - expected).abs() < 1e-12);
    }

    #[test]
    fn residual_errors_on_empty_curve_snapshot() {
        let reference = d(2024, 1, 2);
        let swap = OisSwap::new(
            reference,
            d(2029, 1, 2),
            0.03,
            Frequency::Annual,
            Daycount::Act360,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: Daycount::Act360,
            times: &[],
            discounts: &[],
        };
        let err = swap.residual(reference, &snapshot).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn residual_linear_in_rate() {
        // residual(rate) = rate * annuity - (D(t_0) - D(t_N)) is affine in
        // rate. Perturb by dr and verify the residual shifts by dr * annuity.
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.03_f64;
        let start = reference;
        let maturity = d(2029, 1, 2);
        let (times, discounts) = flat_curve(reference, daycount, r_c);
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };

        let placeholder = OisSwap::new(start, maturity, 0.0, Frequency::Annual, daycount).unwrap();
        let r_par = par_ois_rate_flat(&placeholder, reference, r_c);

        let swap_a = OisSwap::new(start, maturity, r_par, Frequency::Annual, daycount).unwrap();
        let swap_b =
            OisSwap::new(start, maturity, r_par + 0.01, Frequency::Annual, daycount).unwrap();
        let r_a = swap_a.residual(reference, &snapshot).unwrap();
        let r_b = swap_b.residual(reference, &snapshot).unwrap();
        // Expected shift: 0.01 * annuity, where annuity is computed from the
        // same flat curve.
        let mut annuity = 0.0_f64;
        for i in 0..swap_a.schedule.len() {
            let s = swap_a.schedule.period_start(i);
            let e = swap_a.schedule.period_end(i);
            let tau = daycount.year_fraction(s, e).unwrap();
            let t_pay = daycount.year_fraction(reference, e).unwrap();
            annuity += tau * (-r_c * t_pay).exp();
        }
        assert!(((r_b - r_a) - 0.01 * annuity).abs() < 1e-12);
    }
}
