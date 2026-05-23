// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Swap-leg payment schedules.
//!
//! A [`SwapSchedule`] is a precomputed sequence of period start / end dates
//! defining the accrual periods of a swap leg. The bootstrap engine reads
//! these dates as-is — no calendar-aware adjustment is applied inside this
//! crate (see WORKING.md §6: holiday calendars are jurisdiction-specific and
//! intentionally out-of-scope).
//!
//! The [`SwapSchedule::from_regular`] helper builds a strictly regular
//! schedule whose period length divides the total swap term exactly. For
//! irregular schedules (stub first/last periods, modified-following business-
//! day adjustments) the caller composes the schedule itself from `Date`
//! arithmetic and passes it via [`SwapSchedule::from_dates`].
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §2.
//! - ISDA, *2006 ISDA Definitions*, §4.6 ("Calculation Period").

use crate::errors::BootstrapError;
use crate::types::{Date, Frequency, Tenor, TenorUnit};

/// A swap-leg payment schedule: a sequence of contiguous accrual periods
/// `[d_0, d_1), [d_1, d_2), ..., [d_{n-1}, d_n)`.
///
/// The schedule stores `n + 1` boundary dates with `dates[0]` the leg start
/// and `dates[n]` the leg maturity; period `i` (zero-indexed, `0 <= i < n`)
/// spans `dates[i]..dates[i+1]`. The leg's payment cash flows fall on
/// `dates[1..=n]`.
///
/// # Invariants
///
/// - `dates.len() >= 2` (at least one period).
/// - `dates[i] < dates[i + 1]` for every `i` (strictly increasing dates).
///
/// # Examples
///
/// ```
/// use regit_curves::instruments::SwapSchedule;
/// use regit_curves::types::{Date, Frequency};
///
/// let start    = Date::from_ymd(2024, 1, 2).unwrap();
/// let maturity = Date::from_ymd(2026, 1, 2).unwrap();
/// // 2y semi-annual schedule -> 4 periods.
/// let sch = SwapSchedule::from_regular(start, maturity, Frequency::SemiAnnual).unwrap();
/// assert_eq!(sch.len(), 4);
/// assert_eq!(sch.period_start(0), start);
/// assert_eq!(sch.payment_date(3), maturity);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapSchedule {
    /// Period boundary dates: `dates[0]` start, `dates[n]` maturity.
    dates: Vec<Date>,
}

impl SwapSchedule {
    /// Builds a regular schedule from a `(start, maturity)` pair and a payment
    /// frequency.
    ///
    /// The total term `maturity - start` is split into `n` equal periods of
    /// length `12 / freq.periods_per_year()` months, generated forwards from
    /// `start`. The function requires that the resulting `n`-th period
    /// boundary fall exactly on `maturity` (i.e. the schedule must be truly
    /// regular). For irregular schedules use [`SwapSchedule::from_dates`].
    ///
    /// `freq = Frequency::OnceAtMaturity` produces a single period
    /// `[start, maturity)`.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `start >= maturity`, or if
    ///   the regular generator's final date does not match `maturity` exactly
    ///   (i.e. the schedule is not regular at the requested frequency).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::SwapSchedule;
    /// use regit_curves::types::{Date, Frequency};
    ///
    /// let s = Date::from_ymd(2024, 6, 15).unwrap();
    /// let m = Date::from_ymd(2025, 6, 15).unwrap();
    /// // 1y annual -> single period.
    /// let sch = SwapSchedule::from_regular(s, m, Frequency::Annual).unwrap();
    /// assert_eq!(sch.len(), 1);
    /// ```
    pub fn from_regular(
        start: Date,
        maturity: Date,
        freq: Frequency,
    ) -> Result<Self, BootstrapError> {
        if start.serial() >= maturity.serial() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "schedule start must precede maturity",
            });
        }
        if matches!(freq, Frequency::OnceAtMaturity) {
            return Ok(Self {
                dates: vec![start, maturity],
            });
        }
        let n = freq.periods_per_year();
        if n == 0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "frequency periods_per_year must be positive",
            });
        }
        // Each period is `12 / n` months long (n in {1, 2, 4, 12}).
        let months_per_period = i32::try_from(12 / n).unwrap_or(1);
        let mut dates = vec![start];
        let mut k: i32 = 1;
        loop {
            let total_months =
                months_per_period
                    .checked_mul(k)
                    .ok_or(BootstrapError::InvalidInstrument {
                        at_index: 0,
                        reason: "schedule overflow",
                    })?;
            let next = Tenor::new(total_months, TenorUnit::Months).add_to(start);
            if next.serial() > maturity.serial() {
                return Err(BootstrapError::InvalidInstrument {
                    at_index: 0,
                    reason: "schedule is not regular at the requested frequency",
                });
            }
            dates.push(next);
            if next.serial() == maturity.serial() {
                break;
            }
            k = k.checked_add(1).ok_or(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "schedule overflow",
            })?;
            if k > 10_000 {
                return Err(BootstrapError::InvalidInstrument {
                    at_index: 0,
                    reason: "schedule exceeded 10000 periods",
                });
            }
        }
        Ok(Self { dates })
    }

    /// Builds a schedule directly from a sequence of boundary dates.
    ///
    /// Requires at least two dates, strictly increasing.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if fewer than two dates are
    ///   supplied or if any two consecutive dates are not strictly increasing.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::SwapSchedule;
    /// use regit_curves::types::Date;
    ///
    /// let d0 = Date::from_ymd(2024, 1, 2).unwrap();
    /// let d1 = Date::from_ymd(2024, 7, 2).unwrap();
    /// let d2 = Date::from_ymd(2025, 1, 2).unwrap();
    /// let sch = SwapSchedule::from_dates(&[d0, d1, d2]).unwrap();
    /// assert_eq!(sch.len(), 2);
    /// ```
    pub fn from_dates(dates: &[Date]) -> Result<Self, BootstrapError> {
        if dates.len() < 2 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "schedule requires at least two boundary dates",
            });
        }
        for w in dates.windows(2) {
            if w[0].serial() >= w[1].serial() {
                return Err(BootstrapError::InvalidInstrument {
                    at_index: 0,
                    reason: "schedule dates must be strictly increasing",
                });
            }
        }
        Ok(Self {
            dates: dates.to_vec(),
        })
    }

    /// Returns the number of accrual periods (one fewer than the number of
    /// boundary dates).
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.dates.len().saturating_sub(1)
    }

    /// Returns `true` if the schedule has no periods. Always `false` for a
    /// successfully constructed `SwapSchedule`.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the start date of the leg (= `dates[0]`).
    #[must_use]
    #[inline]
    pub fn start(&self) -> Date {
        self.dates[0]
    }

    /// Returns the maturity (= last boundary date).
    #[must_use]
    #[inline]
    pub fn maturity(&self) -> Date {
        self.dates[self.dates.len() - 1]
    }

    /// Returns the start of period `i` (`0 <= i < len()`).
    ///
    /// # Panics
    ///
    /// Panics in debug builds only on out-of-range `i`. Callers loop over
    /// `0..self.len()` and so do not exercise that path.
    #[must_use]
    #[inline]
    pub fn period_start(&self, i: usize) -> Date {
        self.dates[i]
    }

    /// Returns the end (= payment date) of period `i`.
    #[must_use]
    #[inline]
    pub fn period_end(&self, i: usize) -> Date {
        self.dates[i + 1]
    }

    /// Alias for [`SwapSchedule::period_end`] — the payment date of period
    /// `i`.
    #[must_use]
    #[inline]
    pub fn payment_date(&self, i: usize) -> Date {
        self.dates[i + 1]
    }

    /// Returns all boundary dates as a slice (length `len() + 1`).
    #[must_use]
    #[inline]
    pub fn dates(&self) -> &[Date] {
        &self.dates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> Date {
        Date::from_ymd(y, m, day).unwrap()
    }

    #[test]
    fn regular_2y_semi_annual_has_four_periods() {
        let s = d(2024, 1, 2);
        let m = d(2026, 1, 2);
        let sch = SwapSchedule::from_regular(s, m, Frequency::SemiAnnual).unwrap();
        assert_eq!(sch.len(), 4);
        assert_eq!(sch.start(), s);
        assert_eq!(sch.maturity(), m);
        assert_eq!(sch.period_end(0), d(2024, 7, 2));
        assert_eq!(sch.period_end(1), d(2025, 1, 2));
        assert_eq!(sch.period_end(2), d(2025, 7, 2));
        assert_eq!(sch.period_end(3), m);
    }

    #[test]
    fn regular_1y_annual_single_period() {
        let s = d(2024, 6, 15);
        let m = d(2025, 6, 15);
        let sch = SwapSchedule::from_regular(s, m, Frequency::Annual).unwrap();
        assert_eq!(sch.len(), 1);
        assert_eq!(sch.period_start(0), s);
        assert_eq!(sch.period_end(0), m);
    }

    #[test]
    fn regular_3y_quarterly_has_twelve_periods() {
        let s = d(2024, 1, 2);
        let m = d(2027, 1, 2);
        let sch = SwapSchedule::from_regular(s, m, Frequency::Quarterly).unwrap();
        assert_eq!(sch.len(), 12);
    }

    #[test]
    fn regular_5y_monthly_has_sixty_periods() {
        let s = d(2024, 1, 2);
        let m = d(2029, 1, 2);
        let sch = SwapSchedule::from_regular(s, m, Frequency::Monthly).unwrap();
        assert_eq!(sch.len(), 60);
    }

    #[test]
    fn once_at_maturity_single_period() {
        let s = d(2024, 1, 2);
        let m = d(2025, 1, 2);
        let sch = SwapSchedule::from_regular(s, m, Frequency::OnceAtMaturity).unwrap();
        assert_eq!(sch.len(), 1);
    }

    #[test]
    fn irregular_schedule_rejected_at_regular_constructor() {
        // 13 months at semi-annual cadence is not regular.
        let s = d(2024, 1, 2);
        let m = d(2025, 2, 2);
        let err = SwapSchedule::from_regular(s, m, Frequency::SemiAnnual).unwrap_err();
        assert!(matches!(
            err,
            BootstrapError::InvalidInstrument {
                reason: r,
                ..
            } if r.contains("not regular")
        ));
    }

    #[test]
    fn start_after_maturity_rejected() {
        let s = d(2026, 1, 2);
        let m = d(2024, 1, 2);
        let err = SwapSchedule::from_regular(s, m, Frequency::Annual).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn equal_start_and_maturity_rejected() {
        let s = d(2024, 1, 2);
        let err = SwapSchedule::from_regular(s, s, Frequency::Annual).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn from_dates_accepts_strictly_increasing() {
        let dates = vec![d(2024, 1, 2), d(2024, 7, 2), d(2025, 1, 2)];
        let sch = SwapSchedule::from_dates(&dates).unwrap();
        assert_eq!(sch.len(), 2);
        assert_eq!(sch.dates(), &dates[..]);
    }

    #[test]
    fn from_dates_rejects_short() {
        let dates = vec![d(2024, 1, 2)];
        let err = SwapSchedule::from_dates(&dates).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn from_dates_rejects_non_increasing() {
        let dates = vec![d(2024, 1, 2), d(2024, 1, 2), d(2025, 1, 2)];
        let err = SwapSchedule::from_dates(&dates).unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn is_empty_false_after_construction() {
        let s = d(2024, 1, 2);
        let m = d(2025, 1, 2);
        let sch = SwapSchedule::from_regular(s, m, Frequency::Annual).unwrap();
        assert!(!sch.is_empty());
    }
}
