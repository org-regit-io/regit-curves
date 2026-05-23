// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Tenor / cross-currency basis-swap instrument.
//!
//! A basis swap exchanges two floating legs of different tenors (or
//! different currencies, in the cross-currency case). One leg pays a
//! short-tenor index (e.g. 3M IBOR), the other pays a long-tenor index
//! (e.g. 6M IBOR), and the quote is the **spread** added to the short-tenor
//! leg that equalises the two legs' present values. In the post-2008
//! multi-curve framework, basis-swap quotes pin a tenor-projection curve to
//! the OIS discount curve.
//!
//! ## Pricing identity
//!
//! Let leg `j ∈ {a, b}` have payment schedule `(t_0^j, t_1^j, ..., t_N^j)`
//! and per-period accrual `tau_i^j`. Under a discount curve `D` and a
//! projection curve `P_j` (one per tenor), the float-leg PV is
//!
//! ```text
//! PV_float(leg_j) = SUM_i tau_i^j * F_j(t_{i-1}^j, t_i^j) * D(t_i^j),
//! ```
//!
//! where `F_j` is the simply-compounded forward rate implied by `P_j`. The
//! basis-swap quote is the spread `s` on `leg_a` such that
//!
//! ```text
//! PV_float(leg_a) + s * A_a = PV_float(leg_b),
//! ```
//!
//! with the spread annuity
//!
//! ```text
//! A_a = SUM_i tau_i^a * D(t_i^a).
//! ```
//!
//! ## Single-curve mode
//!
//! In the single-curve world both legs project on the same discount curve,
//! so each float-leg PV **telescopes** to `D(start) - D(maturity)`. When
//! both legs share the same `start` and `maturity`, the two float PVs are
//! identical and the residual collapses to
//!
//! ```text
//! residual = spread * A_a.
//! ```
//!
//! That single-curve residual is **deliberately uninformative**: it pins
//! the spread to zero independent of the curve, which is the correct
//! single-curve answer (basis spreads are a multi-curve phenomenon). The
//! informative multi-curve residual, with independent projection curves per
//! `index_tenor`, lives in `multi_curve.rs`. The instrument is implemented
//! here **structurally** so that the single-curve and multi-curve
//! bootstraps share the same data type.
//!
//! # References
//!
//! - Mercurio, F., "Interest Rates and The Credit Crunch: New Formulas and
//!   Market Models", Bloomberg Portfolio Research Paper No. 2010-01-FRONTIERS
//!   (2009), §3. Basis-swap pricing under the multi-curve framework.
//! - Bianchetti, M., "Two Curves, One Price", *Risk Magazine*, August 2010,
//!   pp. 66-72. Discount / forward curve separation pinned by basis-swap
//!   quotes.
//! - Ametrano, F. M. & Bianchetti, M., "Everything You Always Wanted to Know
//!   About Multiple Interest Rate Curve Bootstrapping but Were Afraid to
//!   Ask", SSRN 2219548 (2013), §4. Tenor-curve bootstrap from basis swaps.

use crate::errors::BootstrapError;
use crate::types::{Date, Daycount, Frequency, Tenor};

use super::{CurveSnapshot, InstrumentLike, SwapSchedule};

/// One leg of a basis swap.
///
/// A basis leg is a floating-rate leg defined by its accrual schedule and the
/// index tenor (e.g. 3M IBOR) that determines which tenor-projection curve
/// would price it in a multi-curve setting. The leg stores the pre-built
/// [`SwapSchedule`] computed from `(start, maturity, freq)`.
///
/// Fields:
///
/// - `start` — leg start date (first accrual period's start).
/// - `maturity` — leg maturity date (last payment date).
/// - `freq` — payment frequency (one payment per period).
/// - `daycount` — day-count convention applied to every period accrual `tau_i`.
/// - `index_tenor` — the tenor of the floating index that fixes the
///   forward rate (3M, 6M, ...). Held for the multi-curve projection-curve
///   selection; the single-curve residual ignores it.
/// - `schedule` — the regular schedule generated from `(start, maturity, freq)`.
///
/// Constructed via [`BasisLeg::new`].
///
/// # Examples
///
/// ```
/// use regit_curves::instruments::basis_swap::BasisLeg;
/// use regit_curves::types::{Date, Daycount, Frequency, Tenor, TenorUnit};
///
/// let start    = Date::from_ymd(2024, 1, 2).unwrap();
/// let maturity = Date::from_ymd(2029, 1, 2).unwrap();
/// let leg = BasisLeg::new(
///     start,
///     maturity,
///     Frequency::Quarterly,
///     Daycount::Act360,
///     Tenor::new(3, TenorUnit::Months),
/// )
/// .unwrap();
/// // 5y quarterly -> 20 periods.
/// assert_eq!(leg.schedule.len(), 20);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BasisLeg {
    /// Leg start date (first accrual period's start).
    pub start: Date,
    /// Leg maturity (last payment date).
    pub maturity: Date,
    /// Payment frequency.
    pub freq: Frequency,
    /// Day-count convention applied to every period accrual `tau_i`.
    pub daycount: Daycount,
    /// Tenor of the floating index that fixes the forward rate.
    pub index_tenor: Tenor,
    /// Regular schedule generated from `(start, maturity, freq)`.
    pub schedule: SwapSchedule,
}

/// A basis swap — `leg_a + spread` vs `leg_b`.
///
/// By convention the spread is applied to `leg_a` (the short-tenor leg in
/// typical IBOR-basis quotes). The basis-swap par equation pins the spread
/// `s` such that the two projected float legs (plus `s` on `leg_a`) have
/// equal present value.
///
/// In the single-curve world the float-leg PVs telescope to
/// `D(start) - D(maturity)` on a shared curve, so the residual reduces to
/// `spread * A_a`. The informative multi-curve residual — which carries
/// the information that pins the projection curve — lives in
/// `multi_curve.rs`.
///
/// Fields:
///
/// - `leg_a` — the leg that carries the basis spread (typically the
///   short-tenor leg).
/// - `leg_b` — the other leg of the basis swap.
/// - `spread` — the quoted basis spread (decimal rate units, e.g. `0.0010`
///   for 10 bp). Applied to `leg_a` by convention.
///
/// Constructed via [`BasisSwap::new`].
///
/// # Examples
///
/// ```
/// use regit_curves::instruments::basis_swap::{BasisLeg, BasisSwap};
/// use regit_curves::types::{Date, Daycount, Frequency, Tenor, TenorUnit};
///
/// let start    = Date::from_ymd(2024, 1, 2).unwrap();
/// let maturity = Date::from_ymd(2029, 1, 2).unwrap();
/// let leg_a = BasisLeg::new(
///     start, maturity, Frequency::Quarterly, Daycount::Act360,
///     Tenor::new(3, TenorUnit::Months),
/// ).unwrap();
/// let leg_b = BasisLeg::new(
///     start, maturity, Frequency::SemiAnnual, Daycount::Act360,
///     Tenor::new(6, TenorUnit::Months),
/// ).unwrap();
/// let bs = BasisSwap::new(leg_a, leg_b, 0.0010).unwrap();
/// assert_eq!(bs.pillar(), maturity);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BasisSwap {
    /// The leg that carries the basis spread.
    pub leg_a: BasisLeg,
    /// The other leg of the basis swap.
    pub leg_b: BasisLeg,
    /// Quoted basis spread (decimal rate units). Applied to [`BasisSwap::leg_a`].
    pub spread: f64,
}

impl BasisLeg {
    /// Constructs a basis-swap leg after validating its invariants.
    ///
    /// Validation:
    ///
    /// - `start` must be strictly before `maturity`.
    /// - `index_tenor.count` must be strictly positive (a tenor must be a
    ///   non-degenerate forward window).
    /// - The regular schedule built from `(start, maturity, freq)` must
    ///   succeed — i.e. the term must be a whole-number multiple of the
    ///   period length under [`SwapSchedule::from_regular`]'s contract.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `start >= maturity`, if
    ///   `index_tenor.count <= 0`, or if the schedule generator rejects the
    ///   `(start, maturity, freq)` triple as not regular.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::basis_swap::BasisLeg;
    /// use regit_curves::types::{Date, Daycount, Frequency, Tenor, TenorUnit};
    /// use regit_curves::BootstrapError;
    ///
    /// let s = Date::from_ymd(2024, 1, 2).unwrap();
    /// let m = Date::from_ymd(2029, 1, 2).unwrap();
    /// assert!(BasisLeg::new(
    ///     s, m, Frequency::Quarterly, Daycount::Act360,
    ///     Tenor::new(3, TenorUnit::Months),
    /// ).is_ok());
    /// // Inverted dates rejected:
    /// assert!(matches!(
    ///     BasisLeg::new(
    ///         m, s, Frequency::Quarterly, Daycount::Act360,
    ///         Tenor::new(3, TenorUnit::Months),
    ///     ).unwrap_err(),
    ///     BootstrapError::InvalidInstrument { .. },
    /// ));
    /// // Zero index-tenor rejected:
    /// assert!(matches!(
    ///     BasisLeg::new(
    ///         s, m, Frequency::Quarterly, Daycount::Act360,
    ///         Tenor::new(0, TenorUnit::Months),
    ///     ).unwrap_err(),
    ///     BootstrapError::InvalidInstrument { .. },
    /// ));
    /// ```
    pub fn new(
        start: Date,
        maturity: Date,
        freq: Frequency,
        daycount: Daycount,
        index_tenor: Tenor,
    ) -> Result<Self, BootstrapError> {
        if start.days_between(maturity) <= 0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "basis leg start must be strictly before maturity",
            });
        }
        if index_tenor.count <= 0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "basis leg index_tenor.count must be strictly positive",
            });
        }
        let schedule = SwapSchedule::from_regular(start, maturity, freq)?;
        Ok(Self {
            start,
            maturity,
            freq,
            daycount,
            index_tenor,
            schedule,
        })
    }

    /// Returns the float-leg PV under the single-curve telescoping identity
    ///
    /// ```text
    /// PV_float(leg) = D(leg.start) - D(leg.maturity).
    /// ```
    ///
    /// This is the classical single-curve simplification: each period's
    /// projected float coupon `tau_i * F(t_{i-1}, t_i) * D(t_i)` equals
    /// `D(t_{i-1}) - D(t_i)` when the projection and discount curves
    /// coincide, and the sum telescopes.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::Type`] if a day-count query against the curve's
    ///   day-count convention fails.
    /// - [`BootstrapError::InvalidInstrument`] if the curve snapshot is
    ///   empty or returns a non-positive discount factor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // The float-leg PV is exposed via the basis-swap residual; see the
    /// // module-level tests for an end-to-end example against a flat curve.
    /// ```
    pub(crate) fn float_pv_single_curve(
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
        if d_start <= 0.0 || d_maturity <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "non-positive discount factor in curve snapshot",
            });
        }
        Ok(d_start - d_maturity)
    }

    /// Returns the spread annuity
    ///
    /// ```text
    /// A = SUM_i tau_i * D(t_i),
    /// ```
    ///
    /// where the sum runs over the leg's payment periods, `tau_i` is the
    /// year fraction of period `i` under the leg's day-count convention, and
    /// `D(t_i)` is the curve's discount factor at the period's payment date.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::Type`] if any day-count query fails (e.g. an
    ///   unsupported [`Daycount::Business252`]).
    /// - [`BootstrapError::InvalidInstrument`] if the curve snapshot is
    ///   empty or returns a non-positive discount factor at any payment date.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::basis_swap::BasisLeg;
    /// use regit_curves::types::{Date, Daycount, Frequency, Tenor, TenorUnit};
    ///
    /// let s = Date::from_ymd(2024, 1, 2).unwrap();
    /// let m = Date::from_ymd(2025, 1, 2).unwrap();
    /// let leg = BasisLeg::new(
    ///     s, m, Frequency::Quarterly, Daycount::Act360,
    ///     Tenor::new(3, TenorUnit::Months),
    /// ).unwrap();
    /// // Schedule covers four quarterly periods.
    /// assert_eq!(leg.schedule.len(), 4);
    /// ```
    pub(crate) fn annuity(
        &self,
        _reference_date: Date,
        curve: &CurveSnapshot<'_>,
    ) -> Result<f64, BootstrapError> {
        let mut a = 0.0_f64;
        for i in 0..self.schedule.len() {
            let p_start = self.schedule.period_start(i);
            let p_end = self.schedule.period_end(i);
            let tau = self.daycount.year_fraction(p_start, p_end)?;
            let t_pay = curve.daycount.year_fraction(curve.reference_date, p_end)?;
            let d_pay = curve
                .discount_at(t_pay)
                .ok_or(BootstrapError::InvalidInstrument {
                    at_index: 0,
                    reason: "curve snapshot is empty",
                })?;
            if d_pay <= 0.0 {
                return Err(BootstrapError::InvalidInstrument {
                    at_index: 0,
                    reason: "non-positive discount factor in curve snapshot",
                });
            }
            a += tau * d_pay;
        }
        Ok(a)
    }
}

impl BasisSwap {
    /// Constructs a basis swap after validating its invariants.
    ///
    /// Validation:
    ///
    /// - `spread` must be finite.
    /// - Both legs must share the same `start` date (the standard market
    ///   convention for tenor basis swaps; cross-tenor legs with mis-aligned
    ///   starts are not supported by this constructor).
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `spread` is not finite or
    ///   if `leg_a.start != leg_b.start`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::basis_swap::{BasisLeg, BasisSwap};
    /// use regit_curves::types::{Date, Daycount, Frequency, Tenor, TenorUnit};
    /// use regit_curves::BootstrapError;
    ///
    /// let s = Date::from_ymd(2024, 1, 2).unwrap();
    /// let m = Date::from_ymd(2029, 1, 2).unwrap();
    /// let leg_a = BasisLeg::new(
    ///     s, m, Frequency::Quarterly, Daycount::Act360,
    ///     Tenor::new(3, TenorUnit::Months),
    /// ).unwrap();
    /// let leg_b = BasisLeg::new(
    ///     s, m, Frequency::SemiAnnual, Daycount::Act360,
    ///     Tenor::new(6, TenorUnit::Months),
    /// ).unwrap();
    /// assert!(BasisSwap::new(leg_a.clone(), leg_b.clone(), 0.0010).is_ok());
    /// // Non-finite spread rejected:
    /// assert!(matches!(
    ///     BasisSwap::new(leg_a, leg_b, f64::NAN).unwrap_err(),
    ///     BootstrapError::InvalidInstrument { .. },
    /// ));
    /// ```
    pub fn new(leg_a: BasisLeg, leg_b: BasisLeg, spread: f64) -> Result<Self, BootstrapError> {
        if !spread.is_finite() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "basis swap spread must be finite",
            });
        }
        if leg_a.start != leg_b.start {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "basis swap legs must share the same start date",
            });
        }
        Ok(Self {
            leg_a,
            leg_b,
            spread,
        })
    }

    /// The instrument's pillar date — the later of the two leg maturities.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::basis_swap::{BasisLeg, BasisSwap};
    /// use regit_curves::types::{Date, Daycount, Frequency, Tenor, TenorUnit};
    ///
    /// let s   = Date::from_ymd(2024, 1, 2).unwrap();
    /// let m_a = Date::from_ymd(2028, 1, 2).unwrap();
    /// let m_b = Date::from_ymd(2029, 1, 2).unwrap();
    /// let leg_a = BasisLeg::new(
    ///     s, m_a, Frequency::Quarterly, Daycount::Act360,
    ///     Tenor::new(3, TenorUnit::Months),
    /// ).unwrap();
    /// let leg_b = BasisLeg::new(
    ///     s, m_b, Frequency::SemiAnnual, Daycount::Act360,
    ///     Tenor::new(6, TenorUnit::Months),
    /// ).unwrap();
    /// let bs = BasisSwap::new(leg_a, leg_b, 0.0).unwrap();
    /// assert_eq!(bs.pillar(), m_b);
    /// ```
    #[must_use]
    #[inline]
    pub fn pillar(&self) -> Date {
        if self.leg_a.maturity.serial() >= self.leg_b.maturity.serial() {
            self.leg_a.maturity
        } else {
            self.leg_b.maturity
        }
    }

    /// Returns the single-curve residual
    ///
    /// ```text
    /// residual = PV_float(leg_a) + spread * A_a - PV_float(leg_b),
    /// ```
    ///
    /// evaluated against the supplied [`CurveSnapshot`]. In the single-curve
    /// world, both legs project on the same curve and their float PVs each
    /// telescope to `D(start) - D(maturity)`. When the two legs share the
    /// same `start` and `maturity` (the typical market case) the two float
    /// PVs cancel and the residual reduces to `spread * A_a`.
    ///
    /// This residual is **deliberately uninformative** in the single-curve
    /// regime — it pins the spread to zero independent of curve shape, which
    /// is the correct single-curve answer. The informative multi-curve
    /// residual lives in `multi_curve.rs`, where each leg projects on its
    /// own tenor-projection curve.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::Type`] if a day-count query fails on either leg.
    /// - [`BootstrapError::InvalidInstrument`] if the curve snapshot is
    ///   empty or returns a non-positive discount factor.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::basis_swap::{BasisLeg, BasisSwap};
    /// use regit_curves::types::{Date, Daycount, Frequency, Tenor, TenorUnit};
    ///
    /// let s = Date::from_ymd(2024, 1, 2).unwrap();
    /// let m = Date::from_ymd(2025, 1, 2).unwrap();
    /// let leg = BasisLeg::new(
    ///     s, m, Frequency::Quarterly, Daycount::Act360,
    ///     Tenor::new(3, TenorUnit::Months),
    /// ).unwrap();
    /// let bs = BasisSwap::new(leg.clone(), leg, 0.0).unwrap();
    /// assert_eq!(bs.pillar(), m);
    /// ```
    pub(crate) fn single_curve_residual(
        &self,
        reference_date: Date,
        curve: &CurveSnapshot<'_>,
    ) -> Result<f64, BootstrapError> {
        let pv_a = self.leg_a.float_pv_single_curve(reference_date, curve)?;
        let pv_b = self.leg_b.float_pv_single_curve(reference_date, curve)?;
        let annuity_a = self.leg_a.annuity(reference_date, curve)?;
        Ok(pv_a + self.spread * annuity_a - pv_b)
    }
}

impl InstrumentLike for BasisSwap {
    #[inline]
    fn pillar(&self) -> Date {
        BasisSwap::pillar(self)
    }

    fn residual(
        &self,
        reference_date: Date,
        curve: &CurveSnapshot<'_>,
    ) -> Result<f64, BootstrapError> {
        self.single_curve_residual(reference_date, curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruments::CurveSnapshot;
    use crate::types::TenorUnit;

    fn d(y: i32, m: u32, day: u32) -> Date {
        Date::from_ymd(y, m, day).unwrap()
    }

    /// Builds a hand-rolled flat continuously-compounded discount curve
    /// `D(t) = exp(-r * t)` on a regular quarterly grid covering 30y.
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

    fn make_leg(start: Date, maturity: Date, freq: Frequency, tenor_months: i32) -> BasisLeg {
        BasisLeg::new(
            start,
            maturity,
            freq,
            Daycount::Act360,
            Tenor::new(tenor_months, TenorUnit::Months),
        )
        .unwrap()
    }

    // ─── Construction & validation ───────────────────────────────────────

    #[test]
    fn new_accepts_3m_vs_6m_5y_basis_swap() {
        let s = d(2024, 1, 2);
        let m = d(2029, 1, 2);
        let leg_a = make_leg(s, m, Frequency::Quarterly, 3);
        let leg_b = make_leg(s, m, Frequency::SemiAnnual, 6);
        let bs = BasisSwap::new(leg_a, leg_b, 0.0010).unwrap();
        assert_eq!(bs.leg_a.schedule.len(), 20);
        assert_eq!(bs.leg_b.schedule.len(), 10);
        assert!((bs.spread - 0.0010).abs() < 1e-15);
    }

    #[test]
    fn new_rejects_non_finite_spread() {
        let s = d(2024, 1, 2);
        let m = d(2029, 1, 2);
        let leg_a = make_leg(s, m, Frequency::Quarterly, 3);
        let leg_b = make_leg(s, m, Frequency::SemiAnnual, 6);
        let err = BasisSwap::new(leg_a.clone(), leg_b.clone(), f64::NAN).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
        let err = BasisSwap::new(leg_a, leg_b, f64::INFINITY).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_mismatched_leg_starts() {
        let s_a = d(2024, 1, 2);
        let s_b = d(2024, 1, 3);
        let m = d(2029, 1, 2);
        let leg_a = make_leg(s_a, m, Frequency::Quarterly, 3);
        // Build leg_b directly with a different start.
        let leg_b = BasisLeg {
            start: s_b,
            maturity: m,
            freq: Frequency::SemiAnnual,
            daycount: Daycount::Act360,
            index_tenor: Tenor::new(6, TenorUnit::Months),
            schedule: SwapSchedule::from_dates(&[s_b, d(2024, 7, 3), d(2025, 1, 3)]).unwrap(),
        };
        let err = BasisSwap::new(leg_a, leg_b, 0.0).unwrap_err();
        assert!(matches!(
            err,
            BootstrapError::InvalidInstrument {
                reason: r,
                ..
            } if r.contains("same start"),
        ));
    }

    #[test]
    fn basis_leg_rejects_inverted_dates() {
        let err = BasisLeg::new(
            d(2029, 1, 2),
            d(2024, 1, 2),
            Frequency::Quarterly,
            Daycount::Act360,
            Tenor::new(3, TenorUnit::Months),
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn basis_leg_rejects_zero_index_tenor() {
        let err = BasisLeg::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            Frequency::Quarterly,
            Daycount::Act360,
            Tenor::new(0, TenorUnit::Months),
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn basis_leg_rejects_negative_index_tenor() {
        let err = BasisLeg::new(
            d(2024, 1, 2),
            d(2029, 1, 2),
            Frequency::Quarterly,
            Daycount::Act360,
            Tenor::new(-3, TenorUnit::Months),
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    // ─── Pillar / accessors ──────────────────────────────────────────────

    #[test]
    fn pillar_returns_max_leg_maturity() {
        let s = d(2024, 1, 2);
        let m_a = d(2028, 1, 2);
        let m_b = d(2029, 1, 2);
        let leg_a = make_leg(s, m_a, Frequency::Quarterly, 3);
        let leg_b = make_leg(s, m_b, Frequency::SemiAnnual, 6);
        let bs = BasisSwap::new(leg_a, leg_b, 0.0).unwrap();
        assert_eq!(bs.pillar(), m_b);
        assert_eq!(InstrumentLike::pillar(&bs), m_b);
    }

    #[test]
    fn pillar_returns_leg_a_when_longer() {
        let s = d(2024, 1, 2);
        let m_a = d(2030, 1, 2);
        let m_b = d(2029, 1, 2);
        let leg_a = make_leg(s, m_a, Frequency::SemiAnnual, 3);
        let leg_b = make_leg(s, m_b, Frequency::SemiAnnual, 6);
        let bs = BasisSwap::new(leg_a, leg_b, 0.0).unwrap();
        assert_eq!(bs.pillar(), m_a);
    }

    // ─── Telescoping single-curve PV ─────────────────────────────────────

    #[test]
    fn float_pv_single_curve_telescopes_to_d_start_minus_d_maturity() {
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.04_f64;
        let (times, discounts) = flat_curve(reference, daycount, r_c);
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };

        let leg = make_leg(reference, d(2029, 1, 2), Frequency::Quarterly, 3);
        let pv = leg.float_pv_single_curve(reference, &snapshot).unwrap();

        let t_start = daycount.year_fraction(reference, reference).unwrap();
        let t_mat = daycount.year_fraction(reference, d(2029, 1, 2)).unwrap();
        let expected = (-r_c * t_start).exp() - (-r_c * t_mat).exp();
        assert!((pv - expected).abs() < 1e-12);
    }

    // ─── Annuity numerical target ────────────────────────────────────────

    #[test]
    fn single_curve_residual_collapses_to_spread_times_annuity() {
        // Both legs share the same start and maturity -> the two telescoped
        // float PVs are identical and the residual reduces to s * A_a.
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.04_f64;
        let (times, discounts) = flat_curve(reference, daycount, r_c);
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };

        let start = reference;
        let maturity = d(2029, 1, 2);
        let leg_a = make_leg(start, maturity, Frequency::Quarterly, 3);
        let leg_b = make_leg(start, maturity, Frequency::Quarterly, 3);
        let spread = 0.0010_f64;
        let bs = BasisSwap::new(leg_a.clone(), leg_b, spread).unwrap();

        let annuity_a = leg_a.annuity(reference, &snapshot).unwrap();
        let residual = bs.single_curve_residual(reference, &snapshot).unwrap();

        let expected = spread * annuity_a;
        assert!(
            (residual - expected).abs() < 1e-12,
            "residual {residual} != spread * annuity_a {expected}",
        );
    }

    #[test]
    fn instrument_like_residual_matches_single_curve_residual() {
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.04_f64;
        let (times, discounts) = flat_curve(reference, daycount, r_c);
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };

        let leg_a = make_leg(reference, d(2029, 1, 2), Frequency::Quarterly, 3);
        let leg_b = make_leg(reference, d(2029, 1, 2), Frequency::SemiAnnual, 6);
        let bs = BasisSwap::new(leg_a, leg_b, 0.0010).unwrap();

        let via_trait = InstrumentLike::residual(&bs, reference, &snapshot).unwrap();
        let direct = bs.single_curve_residual(reference, &snapshot).unwrap();
        assert!((via_trait - direct).abs() < 1e-15);
    }

    #[test]
    fn zero_spread_basis_swap_residual_is_zero_with_matched_maturities() {
        // Equal legs (3M-vs-3M, same start/maturity) and zero spread -> the
        // two telescoped float PVs cancel exactly.
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.03_f64;
        let (times, discounts) = flat_curve(reference, daycount, r_c);
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };

        let leg_a = make_leg(reference, d(2029, 1, 2), Frequency::Quarterly, 3);
        let leg_b = make_leg(reference, d(2029, 1, 2), Frequency::Quarterly, 3);
        let bs = BasisSwap::new(leg_a, leg_b, 0.0).unwrap();

        let residual = bs.single_curve_residual(reference, &snapshot).unwrap();
        assert!(
            residual.abs() < 1e-12,
            "matched-leg zero-spread residual must be zero, got {residual}",
        );
    }

    #[test]
    fn residual_errors_on_empty_curve_snapshot() {
        let reference = d(2024, 1, 2);
        let leg_a = make_leg(reference, d(2029, 1, 2), Frequency::Quarterly, 3);
        let leg_b = make_leg(reference, d(2029, 1, 2), Frequency::SemiAnnual, 6);
        let bs = BasisSwap::new(leg_a, leg_b, 0.0010).unwrap();

        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: Daycount::Act360,
            times: &[],
            discounts: &[],
        };
        let err = bs.single_curve_residual(reference, &snapshot).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn annuity_matches_direct_sum_over_periods() {
        // Cross-check `annuity` against a manual telescoped-free summation
        // built independently from the same schedule, day-count, and curve.
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let r_c = 0.04_f64;
        let (times, discounts) = flat_curve(reference, daycount, r_c);
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };

        let leg = make_leg(reference, d(2029, 1, 2), Frequency::Quarterly, 3);
        let computed = leg.annuity(reference, &snapshot).unwrap();

        let mut manual = 0.0_f64;
        for i in 0..leg.schedule.len() {
            let p_start = leg.schedule.period_start(i);
            let p_end = leg.schedule.period_end(i);
            let tau = daycount.year_fraction(p_start, p_end).unwrap();
            let t = daycount.year_fraction(reference, p_end).unwrap();
            let d_pay = snapshot.discount_at(t).unwrap();
            manual += tau * d_pay;
        }
        assert!((computed - manual).abs() < 1e-15);
        // Sanity: positive, less than maturity (5y).
        assert!(computed > 0.0 && computed < 5.0);
    }

    #[test]
    fn debug_clone_eq_round_trip() {
        // `#[derive(Debug, Clone, PartialEq)]` smoke test for both structs.
        let s = d(2024, 1, 2);
        let m = d(2029, 1, 2);
        let leg = make_leg(s, m, Frequency::Quarterly, 3);
        let leg_clone = leg.clone();
        assert_eq!(leg, leg_clone);
        assert!(format!("{leg:?}").contains("BasisLeg"));

        let bs = BasisSwap::new(leg.clone(), leg, 0.0010).unwrap();
        let bs_clone = bs.clone();
        assert_eq!(bs, bs_clone);
        assert!(format!("{bs:?}").contains("BasisSwap"));
    }
}
