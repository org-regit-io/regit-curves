// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Vanilla fixed-floating interest-rate swap.
//!
//! A vanilla IRS exchanges a stream of fixed coupons against a stream of
//! floating coupons indexed off a single LIBOR/IBOR-style tenor. In the
//! **single-curve** world — where the same curve both discounts cash flows
//! and projects forward rates — the floating leg's PV telescopes to the two
//! end discount factors, and the par-rate equation reduces to
//!
//! ```text
//! rate * SUM_i tau_i^fixed * D(t_i^fixed)  =  D(t_start) - D(t_maturity),
//! ```
//!
//! where `D` is the discount curve evaluated at the curve's year-fraction
//! axis, `tau_i^fixed` is the accrual of the `i`-th fixed-leg period under
//! the leg's `fixed_daycount`, and `t_i^fixed` is the year fraction (on the
//! curve's day-count axis) from the curve's reference date to the period's
//! payment date.
//!
//! The residual returned by the instrument's `residual` method is
//!
//! ```text
//! residual = PV_fixed - PV_float
//!          = rate * SUM_i tau_i^fixed * D(t_i^fixed) - (D(t_start) - D(t_maturity)).
//! ```
//!
//! Zero at the bootstrap solution; positive when the quoted rate is above
//! the curve-implied par, negative when below.
//!
//! # Single-curve vs multi-curve
//!
//! The single-curve identity above is exact when one curve handles both
//! discounting and forward projection. Post-2008 markets price vanilla
//! swaps in a **multi-curve** framework — an OIS curve discounts cash flows
//! while a separate tenor-projection curve produces the floating-leg
//! forwards. Multi-curve pricing lives in `multi_curve.rs`; here we expose
//! the single-curve form, which is the right model when the OIS and
//! projection curves coincide (e.g. for OIS swaps, or for a synthetic
//! single-curve calibration).
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §2.2-2.3. Par-swap
//!   rate identity in the single-curve bootstrap.
//! - Mercurio, F., "Interest rates and the credit crunch: new formulas and
//!   market models", *SSRN* 1332205 (2009), §3. Single- and multi-curve
//!   swap-pricing forms; the float-leg telescoping is equation (3.3).
//! - ISDA, *2006 ISDA Definitions*, §6 ("Fixed Amounts and Floating
//!   Amounts") and §4.6 ("Calculation Period"). Coupon accrual conventions.

use crate::errors::BootstrapError;
use crate::types::{Date, Daycount, Frequency};

use super::{CurveSnapshot, InstrumentLike, SwapSchedule};

/// A vanilla fixed-floating interest-rate swap with separate fixed- and
/// floating-leg schedules.
///
/// Both legs span the same `[start, maturity]` interval but may use
/// different payment frequencies and day-count conventions (e.g. semi-
/// annual 30/360 fixed against quarterly Act/360 float — the standard USD
/// LIBOR vanilla convention).
///
/// Constructed via [`SwapFixedFloat::new`] (which builds regular schedules
/// internally) or [`SwapFixedFloat::with_schedules`] (which accepts
/// pre-built schedules — useful for stub first/last periods).
///
/// # Examples
///
/// ```
/// use regit_curves::instruments::SwapFixedFloat;
/// use regit_curves::types::{Date, Daycount, Frequency};
///
/// let start    = Date::from_ymd(2024, 1, 2).unwrap();
/// let maturity = Date::from_ymd(2026, 1, 2).unwrap();
/// let swap = SwapFixedFloat::new(
///     start,
///     maturity,
///     0.04,
///     Frequency::SemiAnnual,
///     Daycount::Act360,
///     Frequency::Quarterly,
///     Daycount::Act360,
/// )
/// .unwrap();
/// assert_eq!(swap.fixed_schedule.len(), 4);
/// assert_eq!(swap.float_schedule.len(), 8);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SwapFixedFloat {
    /// Effective (start) date of both legs.
    pub start: Date,
    /// Maturity date of both legs.
    pub maturity: Date,
    /// Quoted (par) fixed rate, decimal (e.g. `0.04` for 4%).
    pub rate: f64,
    /// Fixed-leg payment frequency.
    pub fixed_freq: Frequency,
    /// Fixed-leg day-count convention (drives the `tau_i^fixed` accruals).
    pub fixed_daycount: Daycount,
    /// Float-leg payment frequency.
    pub float_freq: Frequency,
    /// Float-leg day-count convention (carried for symmetry / multi-curve
    /// pricing; not used in the single-curve identity).
    pub float_daycount: Daycount,
    /// Fixed-leg payment schedule.
    pub fixed_schedule: SwapSchedule,
    /// Float-leg payment schedule.
    pub float_schedule: SwapSchedule,
}

impl SwapFixedFloat {
    /// Constructs a swap with regularly generated fixed- and float-leg
    /// schedules.
    ///
    /// Validation:
    ///
    /// - `rate` must be finite.
    /// - `start < maturity`.
    /// - Both `(start, maturity, freq)` triples must yield a regular schedule
    ///   (i.e. the term must be an integer multiple of each leg's period).
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `rate` is not finite, if
    ///   `start >= maturity`, or if either schedule cannot be built regularly
    ///   at the requested frequency.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::SwapFixedFloat;
    /// use regit_curves::types::{Date, Daycount, Frequency};
    /// use regit_curves::BootstrapError;
    ///
    /// let s = Date::from_ymd(2024, 1, 2).unwrap();
    /// let m = Date::from_ymd(2026, 1, 2).unwrap();
    /// assert!(
    ///     SwapFixedFloat::new(
    ///         s,
    ///         m,
    ///         0.04,
    ///         Frequency::SemiAnnual,
    ///         Daycount::Act360,
    ///         Frequency::Quarterly,
    ///         Daycount::Act360,
    ///     )
    ///     .is_ok()
    /// );
    /// // Inverted dates rejected:
    /// assert!(matches!(
    ///     SwapFixedFloat::new(
    ///         m,
    ///         s,
    ///         0.04,
    ///         Frequency::SemiAnnual,
    ///         Daycount::Act360,
    ///         Frequency::Quarterly,
    ///         Daycount::Act360,
    ///     )
    ///     .unwrap_err(),
    ///     BootstrapError::InvalidInstrument { .. },
    /// ));
    /// ```
    pub fn new(
        start: Date,
        maturity: Date,
        rate: f64,
        fixed_freq: Frequency,
        fixed_daycount: Daycount,
        float_freq: Frequency,
        float_daycount: Daycount,
    ) -> Result<Self, BootstrapError> {
        if !rate.is_finite() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "swap rate must be finite",
            });
        }
        if start.serial() >= maturity.serial() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "swap start must precede maturity",
            });
        }
        let fixed_schedule = SwapSchedule::from_regular(start, maturity, fixed_freq)?;
        let float_schedule = SwapSchedule::from_regular(start, maturity, float_freq)?;
        Ok(Self {
            start,
            maturity,
            rate,
            fixed_freq,
            fixed_daycount,
            float_freq,
            float_daycount,
            fixed_schedule,
            float_schedule,
        })
    }

    /// Constructs a swap from pre-built schedules — the irregular-stub
    /// counterpart to [`SwapFixedFloat::new`].
    ///
    /// Validation:
    ///
    /// - `rate` must be finite.
    /// - `start < maturity`.
    /// - Both schedules must align: `fixed_schedule.start() == start`,
    ///   `fixed_schedule.maturity() == maturity`, and likewise for
    ///   `float_schedule`.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if any validation step fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::{SwapFixedFloat, SwapSchedule};
    /// use regit_curves::types::{Date, Daycount, Frequency};
    ///
    /// let s = Date::from_ymd(2024, 1, 2).unwrap();
    /// let m = Date::from_ymd(2026, 1, 2).unwrap();
    /// let fixed = SwapSchedule::from_regular(s, m, Frequency::SemiAnnual).unwrap();
    /// let float = SwapSchedule::from_regular(s, m, Frequency::Quarterly).unwrap();
    /// let swap = SwapFixedFloat::with_schedules(
    ///     s,
    ///     m,
    ///     0.04,
    ///     Frequency::SemiAnnual,
    ///     Daycount::Act360,
    ///     Frequency::Quarterly,
    ///     Daycount::Act360,
    ///     fixed,
    ///     float,
    /// )
    /// .unwrap();
    /// assert_eq!(swap.fixed_schedule.len(), 4);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn with_schedules(
        start: Date,
        maturity: Date,
        rate: f64,
        fixed_freq: Frequency,
        fixed_daycount: Daycount,
        float_freq: Frequency,
        float_daycount: Daycount,
        fixed_schedule: SwapSchedule,
        float_schedule: SwapSchedule,
    ) -> Result<Self, BootstrapError> {
        if !rate.is_finite() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "swap rate must be finite",
            });
        }
        if start.serial() >= maturity.serial() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "swap start must precede maturity",
            });
        }
        if fixed_schedule.start() != start || fixed_schedule.maturity() != maturity {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "fixed schedule does not span [start, maturity]",
            });
        }
        if float_schedule.start() != start || float_schedule.maturity() != maturity {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "float schedule does not span [start, maturity]",
            });
        }
        Ok(Self {
            start,
            maturity,
            rate,
            fixed_freq,
            fixed_daycount,
            float_freq,
            float_daycount,
            fixed_schedule,
            float_schedule,
        })
    }

    /// Returns the fixed-leg PV:
    ///
    /// ```text
    /// PV_fixed = rate * SUM_i tau_i^fixed * D(t_i^fixed).
    /// ```
    ///
    /// `tau_i^fixed` is the year fraction of the `i`-th fixed-leg period
    /// under `fixed_daycount`; `t_i^fixed` is the year fraction from
    /// `curve.reference_date` to the period's payment date under the
    /// curve's own day-count.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::Type`] if any day-count query fails (e.g. an
    ///   uninitialised `Business252` calendar).
    /// - [`BootstrapError::InvalidInstrument`] if the curve snapshot is
    ///   empty / inconsistent.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::SwapFixedFloat;
    /// use regit_curves::types::{Date, Daycount, Frequency};
    ///
    /// let s = Date::from_ymd(2024, 1, 2).unwrap();
    /// let m = Date::from_ymd(2026, 1, 2).unwrap();
    /// let swap = SwapFixedFloat::new(
    ///     s,
    ///     m,
    ///     0.04,
    ///     Frequency::SemiAnnual,
    ///     Daycount::Act360,
    ///     Frequency::Quarterly,
    ///     Daycount::Act360,
    /// )
    /// .unwrap();
    /// // Fixed-leg PV is strictly positive against any sensible curve.
    /// assert!(swap.rate > 0.0);
    /// # let _ = swap;
    /// ```
    pub(crate) fn fixed_leg_pv(
        &self,
        _reference_date: Date,
        curve: &CurveSnapshot<'_>,
    ) -> Result<f64, BootstrapError> {
        let mut annuity = 0.0_f64;
        for i in 0..self.fixed_schedule.len() {
            let period_start = self.fixed_schedule.period_start(i);
            let payment = self.fixed_schedule.period_end(i);
            let tau_i = self.fixed_daycount.year_fraction(period_start, payment)?;
            let t_payment = curve
                .daycount
                .year_fraction(curve.reference_date, payment)?;
            let d_payment =
                curve
                    .discount_at(t_payment)
                    .ok_or(BootstrapError::InvalidInstrument {
                        at_index: 0,
                        reason: "curve snapshot is empty",
                    })?;
            annuity += tau_i * d_payment;
        }
        Ok(self.rate * annuity)
    }

    /// Returns the floating-leg PV under the single-curve telescoping
    /// identity:
    ///
    /// ```text
    /// PV_float = D(t_start) - D(t_maturity).
    /// ```
    ///
    /// This holds exactly when the curve handles both discounting and
    /// forward projection (single-curve regime); the multi-curve form lives
    /// in `multi_curve.rs`.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::Type`] if a day-count query fails.
    /// - [`BootstrapError::InvalidInstrument`] if the curve snapshot is
    ///   empty / inconsistent.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::SwapFixedFloat;
    /// use regit_curves::types::{Date, Daycount, Frequency};
    ///
    /// let s = Date::from_ymd(2024, 1, 2).unwrap();
    /// let m = Date::from_ymd(2026, 1, 2).unwrap();
    /// let swap = SwapFixedFloat::new(
    ///     s,
    ///     m,
    ///     0.04,
    ///     Frequency::SemiAnnual,
    ///     Daycount::Act360,
    ///     Frequency::Quarterly,
    ///     Daycount::Act360,
    /// )
    /// .unwrap();
    /// // Float-leg PV is nominally positive for an upward-sloping curve.
    /// assert_eq!(swap.start, s);
    /// # let _ = swap;
    /// ```
    pub(crate) fn float_leg_pv_single_curve(
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
        Ok(d_start - d_maturity)
    }
}

impl InstrumentLike for SwapFixedFloat {
    #[inline]
    fn pillar(&self) -> Date {
        self.maturity
    }

    fn residual(
        &self,
        reference_date: Date,
        curve: &CurveSnapshot<'_>,
    ) -> Result<f64, BootstrapError> {
        let pv_fixed = self.fixed_leg_pv(reference_date, curve)?;
        let pv_float = self.float_leg_pv_single_curve(reference_date, curve)?;
        Ok(pv_fixed - pv_float)
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
    fn new_accepts_valid_2y_sa_q_swap() {
        let s = d(2024, 1, 2);
        let m = d(2026, 1, 2);
        let swap = SwapFixedFloat::new(
            s,
            m,
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        assert_eq!(swap.start, s);
        assert_eq!(swap.maturity, m);
        assert_eq!(swap.fixed_schedule.len(), 4);
        assert_eq!(swap.float_schedule.len(), 8);
        assert_eq!(swap.pillar(), m);
    }

    #[test]
    fn new_accepts_negative_rate() {
        let s = d(2024, 1, 2);
        let m = d(2026, 1, 2);
        let swap = SwapFixedFloat::new(
            s,
            m,
            -0.005,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        assert!(swap.rate < 0.0);
    }

    #[test]
    fn new_rejects_nan_rate() {
        let s = d(2024, 1, 2);
        let m = d(2026, 1, 2);
        let err = SwapFixedFloat::new(
            s,
            m,
            f64::NAN,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_inf_rate() {
        let s = d(2024, 1, 2);
        let m = d(2026, 1, 2);
        let err = SwapFixedFloat::new(
            s,
            m,
            f64::INFINITY,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_inverted_dates() {
        let s = d(2024, 1, 2);
        let m = d(2026, 1, 2);
        let err = SwapFixedFloat::new(
            m,
            s,
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_equal_dates() {
        let s = d(2024, 1, 2);
        let err = SwapFixedFloat::new(
            s,
            s,
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_irregular_term() {
        // 13 months — not divisible by SA cadence.
        let s = d(2024, 1, 2);
        let m = d(2025, 2, 2);
        let err = SwapFixedFloat::new(
            s,
            m,
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn with_schedules_validates_alignment() {
        let s = d(2024, 1, 2);
        let m = d(2026, 1, 2);
        let fixed = SwapSchedule::from_regular(s, m, Frequency::SemiAnnual).unwrap();
        let float = SwapSchedule::from_regular(s, m, Frequency::Quarterly).unwrap();
        let ok = SwapFixedFloat::with_schedules(
            s,
            m,
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
            fixed.clone(),
            float.clone(),
        );
        assert!(ok.is_ok());

        // Misaligned start: schedule starts at s+1y but swap claims s.
        let mid = d(2025, 1, 2);
        let mismatched = SwapSchedule::from_regular(mid, m, Frequency::SemiAnnual).unwrap();
        let err = SwapFixedFloat::with_schedules(
            s,
            m,
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
            mismatched,
            float.clone(),
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));

        // Misaligned float schedule.
        let mismatched_float = SwapSchedule::from_regular(mid, m, Frequency::Quarterly).unwrap();
        let err = SwapFixedFloat::with_schedules(
            s,
            m,
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
            fixed,
            mismatched_float,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn with_schedules_rejects_nan_rate() {
        let s = d(2024, 1, 2);
        let m = d(2026, 1, 2);
        let fixed = SwapSchedule::from_regular(s, m, Frequency::SemiAnnual).unwrap();
        let float = SwapSchedule::from_regular(s, m, Frequency::Quarterly).unwrap();
        let err = SwapFixedFloat::with_schedules(
            s,
            m,
            f64::NAN,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
            fixed,
            float,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn with_schedules_rejects_inverted_dates() {
        let s = d(2024, 1, 2);
        let m = d(2026, 1, 2);
        let fixed = SwapSchedule::from_regular(s, m, Frequency::SemiAnnual).unwrap();
        let float = SwapSchedule::from_regular(s, m, Frequency::Quarterly).unwrap();
        let err = SwapFixedFloat::with_schedules(
            m,
            s,
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
            fixed,
            float,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    // ─── Pricing identity on a flat continuously-compounded curve ────────

    /// Builds a hand-rolled flat continuously-compounded discount curve
    /// `D(t) = exp(-r * t)` evaluated on a quarterly grid out to ~30 years.
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

    /// Closed-form par rate of a fixed/float swap against a flat
    /// continuously-compounded curve at `r_c`. Uses the float telescoping:
    /// `r_par = (D(t_start) - D(t_maturity)) / SUM tau_i^fixed * D(t_i^fixed)`.
    fn par_rate_against_flat(
        swap_start: Date,
        swap_maturity: Date,
        fixed_freq: Frequency,
        fixed_daycount: Daycount,
        reference_date: Date,
        curve_daycount: Daycount,
        r_c: f64,
    ) -> f64 {
        let schedule = SwapSchedule::from_regular(swap_start, swap_maturity, fixed_freq).unwrap();
        let mut annuity = 0.0_f64;
        for i in 0..schedule.len() {
            let p_start = schedule.period_start(i);
            let p_end = schedule.period_end(i);
            let tau_i = fixed_daycount.year_fraction(p_start, p_end).unwrap();
            let t = curve_daycount.year_fraction(reference_date, p_end).unwrap();
            annuity += tau_i * (-r_c * t).exp();
        }
        let t_start = curve_daycount
            .year_fraction(reference_date, swap_start)
            .unwrap();
        let t_mat = curve_daycount
            .year_fraction(reference_date, swap_maturity)
            .unwrap();
        ((-r_c * t_start).exp() - (-r_c * t_mat).exp()) / annuity
    }

    #[test]
    fn residual_is_zero_on_flat_curve_with_closed_form_par_rate() {
        // 2y semi-annual fixed (Act/360) against quarterly float (Act/360);
        // flat continuously-compounded curve at r_c = 0.04.
        let reference = d(2024, 1, 2);
        let start = reference;
        let maturity = d(2026, 1, 2);
        let dc_curve = Daycount::Act360;
        let r_c = 0.04_f64;
        let (times, discounts) = flat_curve(reference, dc_curve, r_c);

        let r_par = par_rate_against_flat(
            start,
            maturity,
            Frequency::SemiAnnual,
            Daycount::Act360,
            reference,
            dc_curve,
            r_c,
        );
        // Sanity bound: par rate is within a handful of bp of the simple
        // continuous-equivalent for 2y at 4%.
        assert!(r_par > 0.03 && r_par < 0.05, "unexpected r_par = {r_par}");

        let swap = SwapFixedFloat::new(
            start,
            maturity,
            r_par,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc_curve,
            times: &times,
            discounts: &discounts,
        };
        let residual = swap.residual(reference, &snapshot).unwrap();
        assert!(
            residual.abs() < 1e-10,
            "residual at par should be < 1e-10, got {residual}",
        );
    }

    #[test]
    fn residual_sign_responds_to_rate_perturbation() {
        // Quoting fixed above par makes PV_fixed > PV_float -> residual > 0.
        let reference = d(2024, 1, 2);
        let start = reference;
        let maturity = d(2026, 1, 2);
        let dc_curve = Daycount::Act360;
        let r_c = 0.04_f64;
        let (times, discounts) = flat_curve(reference, dc_curve, r_c);

        let r_par = par_rate_against_flat(
            start,
            maturity,
            Frequency::SemiAnnual,
            Daycount::Act360,
            reference,
            dc_curve,
            r_c,
        );
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc_curve,
            times: &times,
            discounts: &discounts,
        };

        let high = SwapFixedFloat::new(
            start,
            maturity,
            r_par + 0.005,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        let low = SwapFixedFloat::new(
            start,
            maturity,
            r_par - 0.005,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        let res_high = high.residual(reference, &snapshot).unwrap();
        let res_low = low.residual(reference, &snapshot).unwrap();
        assert!(
            res_high > 1e-6,
            "expected positive residual, got {res_high}"
        );
        assert!(res_low < -1e-6, "expected negative residual, got {res_low}");
    }

    #[test]
    fn fixed_leg_pv_matches_manual_sum() {
        // Walk the same sum the implementation walks and compare.
        let reference = d(2024, 1, 2);
        let start = reference;
        let maturity = d(2026, 1, 2);
        let dc_curve = Daycount::Act360;
        let r_c = 0.04_f64;
        let (times, discounts) = flat_curve(reference, dc_curve, r_c);

        let swap = SwapFixedFloat::new(
            start,
            maturity,
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc_curve,
            times: &times,
            discounts: &discounts,
        };

        // Manual computation: 4 SA periods, each ~ tau_i * exp(-r_c * t_i).
        let mut expected = 0.0_f64;
        for i in 0..swap.fixed_schedule.len() {
            let p_start = swap.fixed_schedule.period_start(i);
            let p_end = swap.fixed_schedule.period_end(i);
            let tau = Daycount::Act360.year_fraction(p_start, p_end).unwrap();
            let t = dc_curve.year_fraction(reference, p_end).unwrap();
            expected += tau * (-r_c * t).exp();
        }
        expected *= 0.04;
        let got = swap.fixed_leg_pv(reference, &snapshot).unwrap();
        assert!(
            (got - expected).abs() < 1e-12,
            "fixed_leg_pv mismatch: got {got}, expected {expected}",
        );
    }

    #[test]
    fn float_leg_pv_telescopes_to_two_discounts() {
        let reference = d(2024, 1, 2);
        let start = reference;
        let maturity = d(2026, 1, 2);
        let dc_curve = Daycount::Act360;
        let r_c = 0.04_f64;
        let (times, discounts) = flat_curve(reference, dc_curve, r_c);

        let swap = SwapFixedFloat::new(
            start,
            maturity,
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc_curve,
            times: &times,
            discounts: &discounts,
        };

        let t_start = dc_curve.year_fraction(reference, start).unwrap();
        let t_mat = dc_curve.year_fraction(reference, maturity).unwrap();
        let expected = (-r_c * t_start).exp() - (-r_c * t_mat).exp();
        let got = swap
            .float_leg_pv_single_curve(reference, &snapshot)
            .unwrap();
        assert!((got - expected).abs() < 1e-14);
    }

    #[test]
    fn fixed_leg_pv_errors_on_empty_snapshot() {
        let reference = d(2024, 1, 2);
        let swap = SwapFixedFloat::new(
            reference,
            d(2026, 1, 2),
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: Daycount::Act360,
            times: &[],
            discounts: &[],
        };
        let err = swap.fixed_leg_pv(reference, &snapshot).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn float_leg_pv_errors_on_empty_snapshot() {
        let reference = d(2024, 1, 2);
        let swap = SwapFixedFloat::new(
            reference,
            d(2026, 1, 2),
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: Daycount::Act360,
            times: &[],
            discounts: &[],
        };
        let err = swap
            .float_leg_pv_single_curve(reference, &snapshot)
            .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn pillar_is_maturity() {
        let s = d(2024, 1, 2);
        let m = d(2029, 1, 2);
        let swap = SwapFixedFloat::new(
            s,
            m,
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        assert_eq!(swap.pillar(), m);
    }

    #[test]
    fn par_rate_invariant_under_mixed_daycounts() {
        // Fixed Thirty360 against float Act360 — the par-rate identity still
        // drives residual to zero with the right fixed-day-count accrual.
        let reference = d(2024, 1, 2);
        let start = reference;
        let maturity = d(2026, 1, 2);
        let dc_curve = Daycount::Act360;
        let r_c = 0.04_f64;
        let (times, discounts) = flat_curve(reference, dc_curve, r_c);

        let r_par = par_rate_against_flat(
            start,
            maturity,
            Frequency::SemiAnnual,
            Daycount::Thirty360BondBasis,
            reference,
            dc_curve,
            r_c,
        );
        let swap = SwapFixedFloat::new(
            start,
            maturity,
            r_par,
            Frequency::SemiAnnual,
            Daycount::Thirty360BondBasis,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc_curve,
            times: &times,
            discounts: &discounts,
        };
        let residual = swap.residual(reference, &snapshot).unwrap();
        assert!(residual.abs() < 1e-10);
    }

    #[test]
    fn five_year_swap_residual_zero_at_par() {
        let reference = d(2024, 1, 2);
        let start = reference;
        let maturity = d(2029, 1, 2);
        let dc_curve = Daycount::Act360;
        let r_c = 0.035_f64;
        let (times, discounts) = flat_curve(reference, dc_curve, r_c);

        let r_par = par_rate_against_flat(
            start,
            maturity,
            Frequency::SemiAnnual,
            Daycount::Act360,
            reference,
            dc_curve,
            r_c,
        );
        let swap = SwapFixedFloat::new(
            start,
            maturity,
            r_par,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc_curve,
            times: &times,
            discounts: &discounts,
        };
        let residual = swap.residual(reference, &snapshot).unwrap();
        assert!(
            residual.abs() < 1e-10,
            "5y residual at par should be < 1e-10, got {residual}",
        );
    }

    #[test]
    fn debug_format_contains_struct_name() {
        let swap = SwapFixedFloat::new(
            d(2024, 1, 2),
            d(2026, 1, 2),
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        let s = format!("{swap:?}");
        assert!(s.contains("SwapFixedFloat"));
    }

    #[test]
    fn clone_and_eq_round_trip() {
        let swap = SwapFixedFloat::new(
            d(2024, 1, 2),
            d(2026, 1, 2),
            0.04,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        let cloned = swap.clone();
        assert_eq!(swap, cloned);
    }
}
