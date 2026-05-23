// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Short-term interest-rate (STIR) future instrument.
//!
//! A STIR future is quoted as a **price** (e.g. `95.0` for an implied rate of
//! `5%`). The implied **forward** rate over the futures' underlying rate
//! period `[start, end]` is obtained from the quoted price by subtracting the
//! caller-supplied **convexity adjustment** (in rate units, i.e. decimal):
//!
//! ```text
//! r_quoted = (100 - price) / 100
//! r_fwd    = r_quoted - convexity_adjustment
//! ```
//!
//! Once the forward rate is known the instrument pins the discount curve on
//! `[start, end]` by the same simply-compounded identity used for an FRA:
//!
//! ```text
//! D(start) / D(end) = 1 + r_fwd * tau(start, end),
//! ```
//!
//! where `tau` is the year fraction under the future's day-count convention.
//!
//! # Convexity adjustment policy
//!
//! `convexity_adjustment` is a **caller-supplied scalar in rate units**. The
//! model that produces it — typically a short-rate model (Hull-White,
//! Black-Karasinski) or an HJM / SABR construction calibrated to caps or
//! swaptions — is **out of scope for this crate**. See `WORKING.md` §6, open
//! question 1: convexity-adjustment models live in a future, separately
//! audited crate. Passing `0.0` disables the adjustment.
//!
//! A futures price near or above `100` is permitted (negative-rate
//! environments — EUR, CHF, JPY); no clamping is applied.
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §2. The forward-rate
//!   pricing identity used here.
//! - Hull, J. C., *Options, Futures, and Other Derivatives*, 10th edition,
//!   Pearson (2018), §6.3 — "Eurodollar Futures". Standard reference for the
//!   convexity-adjustment relationship between futures and forward rates.
//! - Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling*, Volume I:
//!   Foundations and Vanilla Models, Atlantic Financial Press (2010), §6.3.
//!   Detailed treatment of futures-vs-forward convexity under Gaussian
//!   short-rate dynamics. The convexity-adjustment **model** itself is not
//!   implemented here — only consumed as a caller-supplied scalar.

use crate::errors::{BootstrapError, TypeError};
use crate::types::{Date, Daycount};

use super::{CurveSnapshot, InstrumentLike};

/// A short-term interest-rate (STIR) future.
///
/// Fields:
///
/// - `start` — period start: the futures expiry / IMM date and the first day
///   of the underlying rate period.
/// - `end` — period end: `start + index_tenor` (e.g. `start + 3M` for a
///   3-month-LIBOR / SOFR future).
/// - `price` — quoted futures price. A price of `95.0` implies a **quoted**
///   rate of `5%` (i.e. `(100 - 95) / 100`). Prices near or above `100` are
///   accepted (negative-rate environment).
/// - `convexity_adjustment` — caller-supplied scalar in rate units, subtracted
///   from the quoted rate to give the **forward** rate. Pass `0.0` to
///   disable. See the module header for the policy.
/// - `daycount` — the day-count convention used to compute the accrual `tau`.
///
/// Constructed via [`Future::new`], which validates: `price.is_finite()`,
/// `convexity_adjustment.is_finite()`, `start < end`.
///
/// # Examples
///
/// ```
/// use regit_curves::instruments::Future;
/// use regit_curves::types::{Date, Daycount};
///
/// // 3-month IMM Mar 2024 STIR future: 2024-03-20 -> 2024-06-19 (91 days),
/// // quoted at 95.00 with a 5bp convexity adjustment.
/// let start = Date::from_ymd(2024, 3, 20).unwrap();
/// let end   = Date::from_ymd(2024, 6, 19).unwrap();
/// let fut = Future::new(start, end, 95.0, 0.0005, Daycount::Act360).unwrap();
/// // Quoted rate: 5%. Implied forward rate: 5% - 5bp = 4.95%.
/// assert!((fut.quoted_rate() - 0.05).abs() < 1e-15);
/// assert!((fut.implied_forward_rate() - 0.0495).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Future {
    /// Period start — the futures expiry / IMM date and the first day of the
    /// underlying rate period.
    pub start: Date,
    /// Period end — `start + index_tenor` (e.g. `start + 3M`).
    pub end: Date,
    /// Quoted futures price (e.g. `95.0` implies a quoted rate of `5%`).
    pub price: f64,
    /// Convexity adjustment in rate units (decimal). Subtracted from the
    /// quoted rate to obtain the forward rate. Caller-supplied; the model
    /// that produces it is out of scope for this crate. Pass `0.0` to
    /// disable.
    pub convexity_adjustment: f64,
    /// Day-count convention used to compute the accrual `tau`.
    pub daycount: Daycount,
}

impl Future {
    /// Constructs a STIR future after validating its invariants.
    ///
    /// Validation:
    ///
    /// - `price` must be finite. Prices near or above `100` are permitted
    ///   (negative-rate quotes).
    /// - `convexity_adjustment` must be finite. Pass `0.0` to disable.
    /// - `start` must be strictly before `end` (a future must have a
    ///   non-degenerate underlying rate period).
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `price` or
    ///   `convexity_adjustment` is not finite, or if `start >= end`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::Future;
    /// use regit_curves::types::{Date, Daycount};
    /// use regit_curves::BootstrapError;
    ///
    /// let start = Date::from_ymd(2024, 3, 20).unwrap();
    /// let end   = Date::from_ymd(2024, 6, 19).unwrap();
    /// assert!(Future::new(start, end, 95.0, 0.0, Daycount::Act360).is_ok());
    /// // Inverted dates rejected:
    /// assert!(matches!(
    ///     Future::new(end, start, 95.0, 0.0, Daycount::Act360).unwrap_err(),
    ///     BootstrapError::InvalidInstrument { .. },
    /// ));
    /// ```
    pub fn new(
        start: Date,
        end: Date,
        price: f64,
        convexity_adjustment: f64,
        daycount: Daycount,
    ) -> Result<Self, BootstrapError> {
        if !price.is_finite() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "future price must be finite",
            });
        }
        if !convexity_adjustment.is_finite() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "future convexity adjustment must be finite",
            });
        }
        if start.days_between(end) <= 0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "future start must be strictly before end",
            });
        }
        Ok(Self {
            start,
            end,
            price,
            convexity_adjustment,
            daycount,
        })
    }

    /// Implied rate from the quoted price alone, ignoring the convexity
    /// adjustment:
    ///
    /// ```text
    /// r_quoted = (100 - price) / 100.
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::Future;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let start = Date::from_ymd(2024, 3, 20).unwrap();
    /// let end   = Date::from_ymd(2024, 6, 19).unwrap();
    /// let fut = Future::new(start, end, 95.0, 0.0005, Daycount::Act360).unwrap();
    /// assert!((fut.quoted_rate() - 0.05).abs() < 1e-15);
    /// ```
    #[inline]
    #[must_use]
    pub fn quoted_rate(&self) -> f64 {
        (100.0 - self.price) / 100.0
    }

    /// Implied **forward** rate over `[start, end]`:
    ///
    /// ```text
    /// r_fwd = (100 - price) / 100 - convexity_adjustment.
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::Future;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let start = Date::from_ymd(2024, 3, 20).unwrap();
    /// let end   = Date::from_ymd(2024, 6, 19).unwrap();
    /// // Price 95.0 -> 5% quoted; subtract 5bp convexity -> 4.95%.
    /// let fut = Future::new(start, end, 95.0, 0.0005, Daycount::Act360).unwrap();
    /// assert!((fut.implied_forward_rate() - 0.0495).abs() < 1e-15);
    /// ```
    #[inline]
    #[must_use]
    pub fn implied_forward_rate(&self) -> f64 {
        self.quoted_rate() - self.convexity_adjustment
    }

    /// Year fraction across the future's underlying rate period
    /// `[start, end]`.
    ///
    /// # Errors
    ///
    /// - [`TypeError::InvalidTenor`] if the day-count convention requires a
    ///   calendar it has not been supplied (e.g. [`Daycount::Business252`]).
    /// - [`TypeError::NonPositiveRange`] only if the constructor was bypassed
    ///   (the constructor validates `start < end`).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::Future;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let start = Date::from_ymd(2024, 3, 20).unwrap();
    /// let end   = Date::from_ymd(2024, 6, 19).unwrap();
    /// let fut = Future::new(start, end, 95.0, 0.0, Daycount::Act360).unwrap();
    /// // 91 days under Act/360.
    /// assert!((fut.accrual().unwrap() - 91.0 / 360.0).abs() < 1e-15);
    /// ```
    pub fn accrual(&self) -> Result<f64, TypeError> {
        self.daycount.year_fraction(self.start, self.end)
    }

    /// Returns the discount factor implied at [`Future::end`] given the
    /// discount factor at [`Future::start`]:
    ///
    /// ```text
    /// D(end) = D(start) / (1 + r_fwd * tau).
    /// ```
    ///
    /// `r_fwd` is [`Future::implied_forward_rate`].
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `discount_at_start` is
    ///   non-finite or non-positive, or if `1 + r_fwd * tau` is non-positive
    ///   (pathological deeply-negative-rate input).
    /// - [`BootstrapError::Type`] if the day-count query fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::Future;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let start = Date::from_ymd(2024, 3, 20).unwrap();
    /// let end   = Date::from_ymd(2024, 6, 19).unwrap();
    /// let fut = Future::new(start, end, 95.0, 0.0005, Daycount::Act360).unwrap();
    /// let d_end = fut.implied_discount(1.0).unwrap();
    /// // r_fwd = 0.0495; tau = 91/360.
    /// let expected = 1.0 / (1.0 + 0.0495 * 91.0 / 360.0);
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
        let growth = 1.0 + self.implied_forward_rate() * tau;
        if !growth.is_finite() || growth <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "non-positive accrual factor (1 + r_fwd * tau)",
            });
        }
        Ok(discount_at_start / growth)
    }
}

impl InstrumentLike for Future {
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
        // Residual: D(start) / D(end) - (1 + r_fwd * tau).
        // Zero at the bootstrap solution.
        Ok(d_start / d_end - (1.0 + self.implied_forward_rate() * tau))
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
    fn new_accepts_valid_future() {
        let fut = Future::new(
            d(2024, 3, 20),
            d(2024, 6, 19),
            95.0,
            0.0005,
            Daycount::Act360,
        )
        .unwrap();
        assert_eq!(fut.start, d(2024, 3, 20));
        assert_eq!(fut.end, d(2024, 6, 19));
        assert!((fut.price - 95.0).abs() < 1e-15);
        assert!((fut.convexity_adjustment - 0.0005).abs() < 1e-15);
    }

    #[test]
    fn new_accepts_zero_convexity_adjustment() {
        let fut = Future::new(d(2024, 3, 20), d(2024, 6, 19), 95.0, 0.0, Daycount::Act360).unwrap();
        assert!((fut.implied_forward_rate() - fut.quoted_rate()).abs() < 1e-15);
    }

    #[test]
    fn new_accepts_negative_rate_environment_price_above_100() {
        // EUR / CHF / JPY STIR futures: a price above 100 implies a negative
        // quoted rate. We do not clamp.
        let fut =
            Future::new(d(2024, 3, 20), d(2024, 6, 19), 100.5, 0.0, Daycount::Act360).unwrap();
        assert!(fut.quoted_rate() < 0.0);
        assert!((fut.quoted_rate() + 0.005).abs() < 1e-15);
    }

    #[test]
    fn new_accepts_price_exactly_100() {
        let fut =
            Future::new(d(2024, 3, 20), d(2024, 6, 19), 100.0, 0.0, Daycount::Act360).unwrap();
        assert!((fut.quoted_rate() - 0.0).abs() < 1e-15);
    }

    #[test]
    fn new_rejects_nan_price() {
        let err = Future::new(
            d(2024, 3, 20),
            d(2024, 6, 19),
            f64::NAN,
            0.0,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_inf_price() {
        let err = Future::new(
            d(2024, 3, 20),
            d(2024, 6, 19),
            f64::INFINITY,
            0.0,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_nan_convexity_adjustment() {
        let err = Future::new(
            d(2024, 3, 20),
            d(2024, 6, 19),
            95.0,
            f64::NAN,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_inf_convexity_adjustment() {
        let err = Future::new(
            d(2024, 3, 20),
            d(2024, 6, 19),
            95.0,
            f64::NEG_INFINITY,
            Daycount::Act360,
        )
        .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_inverted_dates() {
        let err =
            Future::new(d(2024, 6, 19), d(2024, 3, 20), 95.0, 0.0, Daycount::Act360).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn new_rejects_equal_dates() {
        // A future must have a strictly non-degenerate underlying rate period
        // (unlike a deposit, which permits a zero-day accrual).
        let err =
            Future::new(d(2024, 3, 20), d(2024, 3, 20), 95.0, 0.0, Daycount::Act360).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    // ─── quoted_rate / implied_forward_rate ──────────────────────────────

    #[test]
    fn quoted_rate_matches_price_definition() {
        let fut = Future::new(d(2024, 3, 20), d(2024, 6, 19), 95.0, 0.0, Daycount::Act360).unwrap();
        assert!((fut.quoted_rate() - 0.05).abs() < 1e-15);
    }

    #[test]
    fn implied_forward_rate_subtracts_convexity_adjustment() {
        let fut = Future::new(
            d(2024, 3, 20),
            d(2024, 6, 19),
            95.0,
            0.0005,
            Daycount::Act360,
        )
        .unwrap();
        // 5% - 5bp = 4.95%.
        assert!((fut.implied_forward_rate() - 0.0495).abs() < 1e-15);
    }

    // ─── Year-fraction helper ────────────────────────────────────────────

    #[test]
    fn accrual_matches_imm_mar_2024_act360() {
        // IMM Mar 2024 -> IMM Jun 2024: 2024-03-20 to 2024-06-19 = 91 days.
        let fut = Future::new(d(2024, 3, 20), d(2024, 6, 19), 95.0, 0.0, Daycount::Act360).unwrap();
        let tau = fut.accrual().unwrap();
        assert!((tau - 91.0 / 360.0).abs() < 1e-15);
    }

    #[test]
    fn accrual_propagates_business252_error() {
        let fut = Future::new(
            d(2024, 3, 20),
            d(2024, 6, 19),
            95.0,
            0.0,
            Daycount::Business252,
        )
        .unwrap();
        let err = fut.accrual().unwrap_err();
        assert!(matches!(err, TypeError::InvalidTenor { .. }));
    }

    // ─── Pricing identity: implied_discount ──────────────────────────────

    #[test]
    fn implied_discount_basic_formula() {
        // IMM Mar 2024 future at 95.0 with 5bp convexity adjustment.
        // r_fwd = 4.95%; tau = 91/360.
        let fut = Future::new(
            d(2024, 3, 20),
            d(2024, 6, 19),
            95.0,
            0.0005,
            Daycount::Act360,
        )
        .unwrap();
        let d_end = fut.implied_discount(1.0).unwrap();
        let tau = 91.0_f64 / 360.0;
        let expected = 1.0 / (1.0 + 0.0495 * tau);
        assert!((d_end - expected).abs() < 1e-15);
    }

    #[test]
    fn implied_discount_zero_convexity_matches_quoted_rate() {
        let fut = Future::new(d(2024, 3, 20), d(2024, 6, 19), 95.0, 0.0, Daycount::Act360).unwrap();
        let d_end = fut.implied_discount(1.0).unwrap();
        let tau = 91.0_f64 / 360.0;
        let expected = 1.0 / (1.0 + 0.05 * tau);
        assert!((d_end - expected).abs() < 1e-15);
    }

    #[test]
    fn implied_discount_scales_linearly_in_d_start() {
        // D(end) = D(start) / (1 + r * tau), so D(end) is linear in D(start).
        let fut = Future::new(d(2024, 3, 20), d(2024, 6, 19), 95.0, 0.0, Daycount::Act360).unwrap();
        let d1 = fut.implied_discount(1.0).unwrap();
        let d2 = fut.implied_discount(0.5).unwrap();
        assert!((d2 - 0.5 * d1).abs() < 1e-15);
    }

    #[test]
    fn implied_discount_rejects_non_finite_d_start() {
        let fut = Future::new(d(2024, 3, 20), d(2024, 6, 19), 95.0, 0.0, Daycount::Act360).unwrap();
        assert!(matches!(
            fut.implied_discount(f64::NAN).unwrap_err(),
            BootstrapError::InvalidInstrument { .. },
        ));
        assert!(matches!(
            fut.implied_discount(f64::INFINITY).unwrap_err(),
            BootstrapError::InvalidInstrument { .. },
        ));
    }

    #[test]
    fn implied_discount_rejects_non_positive_d_start() {
        let fut = Future::new(d(2024, 3, 20), d(2024, 6, 19), 95.0, 0.0, Daycount::Act360).unwrap();
        assert!(matches!(
            fut.implied_discount(0.0).unwrap_err(),
            BootstrapError::InvalidInstrument { .. },
        ));
        assert!(matches!(
            fut.implied_discount(-0.5).unwrap_err(),
            BootstrapError::InvalidInstrument { .. },
        ));
    }

    #[test]
    fn implied_discount_rejects_non_positive_growth() {
        // Pathological deeply-negative forward rate where (1 + r_fwd * tau) <= 0.
        // tau = 91/360 ≈ 0.2528; price = 1000 gives quoted_rate = -9.0, so
        // 1 + (-9.0) * 0.2528 ≈ -1.275.
        let fut = Future::new(
            d(2024, 3, 20),
            d(2024, 6, 19),
            1000.0,
            0.0,
            Daycount::Act360,
        )
        .unwrap();
        let err = fut.implied_discount(1.0).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn implied_discount_propagates_business252_error() {
        let fut = Future::new(
            d(2024, 3, 20),
            d(2024, 6, 19),
            95.0,
            0.0,
            Daycount::Business252,
        )
        .unwrap();
        let err = fut.implied_discount(1.0).unwrap_err();
        assert!(matches!(err, BootstrapError::Type(_)));
    }

    // ─── Residual against a self-consistent curve ────────────────────────

    /// Builds a hand-rolled tabulated discount curve consistent with a given
    /// forward rate `r_fwd` on `[start, end]` and a flat continuously-
    /// compounded rate `r_c` elsewhere. Specifically we lay down knots at the
    /// curve anchor, at `start`, and at `end` such that
    /// `D(start) / D(end) = 1 + r_fwd * tau(start, end)` is honoured exactly.
    fn curve_consistent_with_forward(
        reference: Date,
        daycount: Daycount,
        start: Date,
        end: Date,
        r_fwd: f64,
        r_c: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let t_start = daycount.year_fraction(reference, start).unwrap();
        let t_end = daycount.year_fraction(reference, end).unwrap();
        let tau = daycount.year_fraction(start, end).unwrap();
        let d_start = (-r_c * t_start).exp();
        let d_end = d_start / (1.0 + r_fwd * tau);
        // Anchor at reference (D = 1) plus the two pillar points; this is all
        // the residual computation looks up.
        (vec![0.0, t_start, t_end], vec![1.0, d_start, d_end])
    }

    #[test]
    fn future_residual_is_zero_on_self_consistent_curve() {
        // IMM Mar 2024 numerical test target from the pass spec:
        // start = 2024-03-20, end = 2024-06-19 (91 days, Act/360),
        // price = 95.0, convexity_adjustment = 0.0005.
        // r_fwd = 5% - 0.05% = 4.95%.
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let start = d(2024, 3, 20);
        let end = d(2024, 6, 19);
        let fut = Future::new(start, end, 95.0, 0.0005, daycount).unwrap();
        let r_fwd = fut.implied_forward_rate();
        assert!((r_fwd - 0.0495).abs() < 1e-15);

        let (times, discounts) =
            curve_consistent_with_forward(reference, daycount, start, end, r_fwd, 0.04);
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };
        let residual = fut.residual(reference, &snapshot).unwrap();
        assert!(
            residual.abs() < 1e-12,
            "residual on self-consistent curve must be zero to 1e-12, got {residual}",
        );
    }

    #[test]
    fn future_residual_sign_responds_to_price_perturbation() {
        // A lower price quotes a HIGHER implied rate. With the curve held
        // fixed at the par r_fwd, raising the implied rate by overshooting
        // the price increases `1 + r_fwd * tau` and drives the residual
        // `D(start)/D(end) - (1 + r_fwd * tau)` negative.
        let reference = d(2024, 1, 2);
        let daycount = Daycount::Act360;
        let start = d(2024, 3, 20);
        let end = d(2024, 6, 19);
        let par_fut = Future::new(start, end, 95.0, 0.0005, daycount).unwrap();
        let r_fwd_par = par_fut.implied_forward_rate();

        let (times, discounts) =
            curve_consistent_with_forward(reference, daycount, start, end, r_fwd_par, 0.04);
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount,
            times: &times,
            discounts: &discounts,
        };

        // Drop the quoted price by 50bp -> implied rate rises by 50bp ->
        // residual goes negative.
        let mispriced = Future::new(start, end, 94.5, 0.0005, daycount).unwrap();
        let residual = mispriced.residual(reference, &snapshot).unwrap();
        assert!(residual < -1e-6);
    }

    #[test]
    fn future_residual_errors_on_empty_curve_snapshot() {
        let reference = d(2024, 1, 2);
        let fut = Future::new(d(2024, 3, 20), d(2024, 6, 19), 95.0, 0.0, Daycount::Act360).unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: Daycount::Act360,
            times: &[],
            discounts: &[],
        };
        let err = fut.residual(reference, &snapshot).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn future_residual_propagates_business252_error() {
        // The instrument's `accrual` is queried before any curve lookup, so
        // a `Business252` instrument surfaces the day-count error first.
        let reference = d(2024, 1, 2);
        let fut = Future::new(
            d(2024, 3, 20),
            d(2024, 6, 19),
            95.0,
            0.0,
            Daycount::Business252,
        )
        .unwrap();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: Daycount::Act360,
            times: &[0.0_f64, 1.0],
            discounts: &[1.0_f64, 0.95],
        };
        let err = fut.residual(reference, &snapshot).unwrap_err();
        assert!(matches!(err, BootstrapError::Type(_)));
    }

    // ─── Pillar accessor ─────────────────────────────────────────────────

    #[test]
    fn future_pillar_is_end_date() {
        let fut = Future::new(d(2024, 3, 20), d(2024, 6, 19), 95.0, 0.0, Daycount::Act360).unwrap();
        assert_eq!(fut.pillar(), d(2024, 6, 19));
    }

    // ─── Round-trip identity check ───────────────────────────────────────

    #[test]
    fn future_discount_roundtrip_through_growth_factor() {
        // If we know D(start), implied_discount produces D(end) such that
        // D(start) / D(end) == 1 + r_fwd * tau exactly.
        let fut = Future::new(
            d(2024, 3, 20),
            d(2024, 6, 19),
            95.0,
            0.0005,
            daycount_for_test(),
        )
        .unwrap();
        let d_s = 0.9876;
        let d_e = fut.implied_discount(d_s).unwrap();
        let tau = fut.accrual().unwrap();
        assert!((d_s / d_e - (1.0 + 0.0495 * tau)).abs() < 1e-15);
    }

    fn daycount_for_test() -> Daycount {
        Daycount::Act360
    }
}
