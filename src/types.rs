// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Core temporal and convention types.
//!
//! This module defines the calendar and convention primitives on which every
//! curve and instrument is built:
//!
//! - [`Date`] — a proleptic-Gregorian calendar date stored as an `i32` day
//!   serial counted from the epoch `1970-01-01`. All conversions are exact
//!   integer arithmetic — no floats are used to compute year/month/day.
//! - [`Tenor`] / [`TenorUnit`] — a length of time with a unit.
//! - [`Daycount`] — the set of ISDA / ICMA day-count conventions used to
//!   translate a date range into a year fraction.
//! - [`Compounding`] — the mapping between a discount factor and a zero rate.
//! - [`Frequency`] — a payment frequency for swap legs.
//! - [`BusinessDayConvention`] — a documentation-only enum naming the
//!   business-day-adjustment conventions (calendar-aware adjustment is
//!   out-of-scope for the crate; the caller composes with its own calendar).
//!
//! # Calendar
//!
//! The proleptic-Gregorian calendar is the Gregorian calendar extended
//! backwards through the pre-1582 era. Historical dates before the Gregorian
//! reform are therefore not the dates that would have been recorded at the
//! time, but the calendar is exact, monotonic, and unambiguous for every
//! financial use case (dates are typically post-1900).
//!
//! # Day-counts
//!
//! Each variant of [`Daycount`] implements the rule from its primary source.
//! See [`Daycount::year_fraction`] for the formulas and citations.
//!
//! # References
//!
//! - Hinnant, H., *chrono-Compatible Low-Level Date Algorithms*,
//!   <https://howardhinnant.github.io/date_algorithms.html>. The
//!   `days_from_civil` and `civil_from_days` integer formulae used here.
//! - ISDA, *2006 ISDA Definitions*, §4.16. Day-count conventions.
//! - ICMA, *Rule 251*. The Actual/Actual (ICMA) convention.

use crate::errors::TypeError;

// ─── Date ────────────────────────────────────────────────────────────────────

/// A calendar date, stored as a signed day-serial since `1970-01-01`.
///
/// The internal representation is `i32`, days since the proleptic-Gregorian
/// epoch `1970-01-01`. The range is roughly ±5.8 million years — far beyond
/// any financial use. All arithmetic is integer; no floats are used in
/// calendar conversions.
///
/// # Examples
///
/// ```
/// use regit_curves::types::Date;
///
/// let d = Date::from_ymd(2024, 6, 15).unwrap();
/// assert_eq!(d.year(), 2024);
/// assert_eq!(d.month(), 6);
/// assert_eq!(d.day(), 15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date(i32);

impl Date {
    /// Constructs a `Date` from proleptic-Gregorian year/month/day.
    ///
    /// Validation rejects months outside `1..=12`, days outside `1..=31`,
    /// and days that do not exist in the given month (e.g. February 30, or
    /// February 29 in a non-leap year).
    ///
    /// The integer formula is Hinnant's `days_from_civil`
    /// (<https://howardhinnant.github.io/date_algorithms.html>) with the
    /// epoch shifted to `1970-01-01`.
    ///
    /// # Errors
    ///
    /// - [`TypeError::InvalidDate`] if `(year, month, day)` is not a real
    ///   proleptic-Gregorian date.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::Date;
    ///
    /// let leap = Date::from_ymd(2000, 2, 29).unwrap();
    /// assert_eq!(leap.day(), 29);
    /// assert!(Date::from_ymd(2023, 2, 29).is_err());
    /// ```
    pub fn from_ymd(year: i32, month: u32, day: u32) -> Result<Self, TypeError> {
        if !(1..=12).contains(&month) {
            return Err(TypeError::InvalidDate { year, month, day });
        }
        if day == 0 || day > days_in_month(year, month) {
            return Err(TypeError::InvalidDate { year, month, day });
        }
        Ok(Self(days_from_civil(year, month, day)))
    }

    /// Constructs a `Date` from its day-serial (days since `1970-01-01`).
    ///
    /// No validation is performed (every `i32` is a valid serial).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::Date;
    ///
    /// // 1970-01-01 has serial 0.
    /// let d = Date::from_serial(0);
    /// assert_eq!(d.year(), 1970);
    /// assert_eq!(d.month(), 1);
    /// assert_eq!(d.day(), 1);
    /// ```
    #[must_use]
    #[inline]
    pub const fn from_serial(days: i32) -> Self {
        Self(days)
    }

    /// Returns the day-serial since `1970-01-01`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::Date;
    ///
    /// assert_eq!(Date::from_serial(0).serial(), 0);
    /// assert_eq!(Date::from_ymd(1970, 1, 2).unwrap().serial(), 1);
    /// ```
    #[must_use]
    #[inline]
    pub const fn serial(self) -> i32 {
        self.0
    }

    /// Returns the proleptic-Gregorian year.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::Date;
    ///
    /// assert_eq!(Date::from_ymd(2024, 6, 15).unwrap().year(), 2024);
    /// ```
    #[must_use]
    pub fn year(self) -> i32 {
        civil_from_days(self.0).0
    }

    /// Returns the month of the year (`1..=12`).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::Date;
    ///
    /// assert_eq!(Date::from_ymd(2024, 6, 15).unwrap().month(), 6);
    /// ```
    #[must_use]
    pub fn month(self) -> u32 {
        civil_from_days(self.0).1
    }

    /// Returns the day of the month (`1..=31`).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::Date;
    ///
    /// assert_eq!(Date::from_ymd(2024, 6, 15).unwrap().day(), 15);
    /// ```
    #[must_use]
    pub fn day(self) -> u32 {
        civil_from_days(self.0).2
    }

    /// Adds `days` calendar days, returning the resulting date.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::Date;
    ///
    /// let d = Date::from_ymd(2024, 6, 15).unwrap();
    /// let next = d.add_days(1);
    /// assert_eq!(next.day(), 16);
    /// ```
    #[must_use]
    #[inline]
    pub const fn add_days(self, days: i32) -> Self {
        Self(self.0.wrapping_add(days))
    }

    /// Returns the signed calendar-day difference `other - self` (i.e. number
    /// of days from `self` to `other`).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::Date;
    ///
    /// let a = Date::from_ymd(2024, 1, 1).unwrap();
    /// let b = Date::from_ymd(2024, 12, 31).unwrap();
    /// // 2024 is a leap year — 366 days, last day is at offset 365.
    /// assert_eq!(a.days_between(b), 365);
    /// ```
    #[must_use]
    #[inline]
    pub const fn days_between(self, other: Self) -> i32 {
        other.0.wrapping_sub(self.0)
    }
}

/// Returns `true` if `year` is a leap year in the proleptic-Gregorian
/// calendar (divisible by 4 but not by 100, unless also by 400).
#[inline]
const fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Returns the number of days in the given (year, month).
#[inline]
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Hinnant's `days_from_civil`: converts a proleptic-Gregorian (y, m, d) to
/// the day-serial relative to `1970-01-01`.
///
/// Pre: `1 <= m <= 12`, `1 <= d <= days_in_month(y, m)`.
fn days_from_civil(y: i32, m: u32, d: u32) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    // Year-of-era is in [0, 399] by construction of `era`.
    let yoe = u32::try_from(y - era * 400).unwrap_or(0);
    // Day-of-year with March = 0, in [0, 365].
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    // Day-of-era, in [0, 146096].
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    // doe fits in u32, doe < 146097 so it fits in i32 safely.
    let doe_i = i32::try_from(doe).unwrap_or(i32::MAX);
    era * 146_097 + doe_i - 719_468
}

/// Hinnant's `civil_from_days`: converts a day-serial relative to
/// `1970-01-01` into a proleptic-Gregorian (y, m, d).
fn civil_from_days(z: i32) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    // Day-of-era is in [0, 146096] by construction of `era`.
    let doe = u32::try_from(z - era * 146_097).unwrap_or(0);
    // Year-of-era, in [0, 399].
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y_partial = i32::try_from(yoe).unwrap_or(i32::MAX) + era * 400;
    // Day-of-year (March = 0), in [0, 365].
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    // Month-prime, in [0, 11].
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y_partial + 1 } else { y_partial };
    (y, m, d)
}

// ─── Tenor ───────────────────────────────────────────────────────────────────

/// The unit of a [`Tenor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TenorUnit {
    /// Days.
    Days,
    /// Weeks.
    Weeks,
    /// Months.
    Months,
    /// Years.
    Years,
}

/// A tenor — a length of time expressed in a unit.
///
/// `Tenor` is a fully literal value (`count`, `unit`); it does not carry
/// a calendar. Conversion to a date is via [`Tenor::add_to`], which uses the
/// "end-of-month preserved" rule for `Months` / `Years`.
///
/// # Examples
///
/// ```
/// use regit_curves::types::{Date, Tenor, TenorUnit};
///
/// let three_months = Tenor::new(3, TenorUnit::Months);
/// let start = Date::from_ymd(2024, 1, 31).unwrap();
/// let end = three_months.add_to(start);
/// // End-of-month preserved: Jan 31 + 3M = Apr 30.
/// assert_eq!((end.year(), end.month(), end.day()), (2024, 4, 30));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tenor {
    /// Number of units (may be negative for a backwards tenor).
    pub count: i32,
    /// The unit of `count`.
    pub unit: TenorUnit,
}

impl Tenor {
    /// Constructs a tenor.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::{Tenor, TenorUnit};
    ///
    /// let t = Tenor::new(6, TenorUnit::Months);
    /// assert_eq!(t.count, 6);
    /// ```
    #[must_use]
    #[inline]
    pub const fn new(count: i32, unit: TenorUnit) -> Self {
        Self { count, unit }
    }

    /// Returns `start + tenor` as a `Date`.
    ///
    /// `Days` and `Weeks` are simple integer-day additions. `Months` and
    /// `Years` use the "end-of-month preserved" rule: if `start` is the
    /// last day of its month, the result is the last day of the target
    /// month; if `start.day()` exceeds the days in the target month, the
    /// result is clipped to the last day of that month. No business-day
    /// calendar adjustment is applied.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::{Date, Tenor, TenorUnit};
    ///
    /// let start = Date::from_ymd(2023, 1, 31).unwrap();
    /// let one_month = Tenor::new(1, TenorUnit::Months).add_to(start);
    /// // 2023 not a leap year — Feb has 28 days; clipped from 31.
    /// assert_eq!((one_month.year(), one_month.month(), one_month.day()), (2023, 2, 28));
    /// ```
    #[must_use]
    pub fn add_to(self, start: Date) -> Date {
        match self.unit {
            TenorUnit::Days => start.add_days(self.count),
            TenorUnit::Weeks => start.add_days(self.count.wrapping_mul(7)),
            TenorUnit::Months => add_months(start, self.count),
            TenorUnit::Years => add_months(start, self.count.wrapping_mul(12)),
        }
    }
}

/// Adds `months` months to `start`, using end-of-month preservation.
fn add_months(start: Date, months: i32) -> Date {
    let (y, m, d) = civil_from_days(start.serial());
    // Map month to 0-based for arithmetic, then back to 1-based.
    let m0 = i32::try_from(m).unwrap_or(0).saturating_sub(1);
    let total = m0.wrapping_add(months);
    // Floor-divide / mod by 12 to handle negative tenors.
    let dy = total.div_euclid(12);
    let new_m0 = total.rem_euclid(12);
    let new_y = y.wrapping_add(dy);
    let new_m = u32::try_from(new_m0 + 1).unwrap_or(1);
    let max_d = days_in_month(new_y, new_m);
    let new_d = d.min(max_d);
    Date(days_from_civil(new_y, new_m, new_d))
}

// ─── Daycount ────────────────────────────────────────────────────────────────

/// Day-count convention.
///
/// Returns a year fraction between two dates following the named ISDA / ICMA
/// rule. Formulas are in [`Daycount::year_fraction`].
///
/// # References
///
/// - ISDA, *2006 ISDA Definitions*, §4.16 (b), (d), (e), (f), (g).
/// - ICMA, *Rule 251*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Daycount {
    /// Actual / 360 — ISDA 4.16(e). Money-market default.
    ///
    /// `tau = (d2 - d1) / 360`.
    Act360,
    /// Actual / 365 (Fixed) — ISDA 4.16(d).
    ///
    /// `tau = (d2 - d1) / 365`.
    Act365F,
    /// 30/360 (Bond Basis) — ISDA 4.16(f). The traditional US bond convention.
    Thirty360BondBasis,
    /// 30E/360 (Eurobond) — ISDA 4.16(g).
    Thirty360E,
    /// Actual / Actual (ISDA) — ISDA 4.16(b). Splits the period by calendar
    /// year, attributing leap-year days to a 366 denominator and non-leap to
    /// 365.
    ActActIsda,
    /// Actual / Actual (ICMA) — for a regular interest-period split.
    /// ICMA Rule 251.
    ///
    /// This implementation assumes the date range is a single regular
    /// coupon period under a coupon schedule with `coupons_per_year`
    /// periods per year; the year fraction is `1 / coupons_per_year`.
    /// Callers that need to split an irregular period across multiple
    /// regular ones must compose this convention themselves.
    ActActIcma {
        /// Number of regular coupons per year (`1`, `2`, `4`, or `12`).
        coupons_per_year: u32,
    },
    /// Business / 252 — Brazilian convention. Requires a business-day
    /// calendar, which is jurisdiction-specific and out-of-scope for this
    /// crate. Querying [`Daycount::year_fraction`] on this variant always
    /// returns [`TypeError::InvalidTenor`]. Callers supply already-computed
    /// year fractions directly.
    Business252,
}

impl Daycount {
    /// Returns the year fraction from `d1` to `d2` under this convention.
    ///
    /// `d2` must not precede `d1` (the range must be non-negative). Some
    /// conventions additionally require strictly-positive ranges; see the
    /// per-variant rules below.
    ///
    /// | Variant | Formula |
    /// |---|---|
    /// | `Act360` | `(d2 - d1) / 360` |
    /// | `Act365F` | `(d2 - d1) / 365` |
    /// | `Thirty360BondBasis` | ISDA 2006 §4.16(f), seven rules |
    /// | `Thirty360E` | `(360*(y2-y1) + 30*(m2-m1) + (min(d2,30) - min(d1,30))) / 360` |
    /// | `ActActIsda` | ISDA 2006 §4.16(b), split by calendar year |
    /// | `ActActIcma` | `1 / coupons_per_year` |
    /// | `Business252` | not supported (see variant doc) |
    ///
    /// # Errors
    ///
    /// - [`TypeError::NonPositiveRange`] if `d2 < d1`.
    /// - [`TypeError::InvalidTenor`] if the variant is [`Daycount::Business252`]
    ///   or if `ActActIcma` has `coupons_per_year == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::{Daycount, Date};
    ///
    /// let d1 = Date::from_ymd(2003, 11, 1).unwrap();
    /// let d2 = Date::from_ymd(2004, 5, 1).unwrap();
    /// // ISDA 2006 §4.16(e) worked example: Act/360 = 182/360.
    /// let tau = Daycount::Act360.year_fraction(d1, d2).unwrap();
    /// assert!((tau - 182.0_f64 / 360.0).abs() < 1e-15);
    /// ```
    pub fn year_fraction(self, d1: Date, d2: Date) -> Result<f64, TypeError> {
        let span = d1.days_between(d2);
        if span < 0 {
            return Err(TypeError::NonPositiveRange);
        }
        match self {
            Self::Act360 => Ok(f64::from(span) / 360.0),
            Self::Act365F => Ok(f64::from(span) / 365.0),
            Self::Thirty360BondBasis => Ok(thirty_360_bond_basis(d1, d2)),
            Self::Thirty360E => Ok(thirty_360_e(d1, d2)),
            Self::ActActIsda => Ok(act_act_isda(d1, d2)),
            Self::ActActIcma { coupons_per_year } => {
                if coupons_per_year == 0 {
                    return Err(TypeError::InvalidTenor {
                        reason: "ActActIcma requires coupons_per_year > 0",
                    });
                }
                Ok(1.0 / f64::from(coupons_per_year))
            }
            Self::Business252 => Err(TypeError::InvalidTenor {
                reason: "Business252 requires a calendar; supply already-computed year fractions",
            }),
        }
    }
}

/// 30/360 Bond Basis — ISDA 2006 §4.16(f).
///
/// The convention transforms the two dates `(Y1, M1, D1)` and `(Y2, M2, D2)`
/// by applying the following rules (in order) before computing
/// `(360*(Y2-Y1) + 30*(M2-M1) + (D2-D1)) / 360`:
///
/// 1. If `D1` is 31, set `D1 = 30`.
/// 2. If `D2` is 31 and `D1` is 30 or 31, set `D2 = 30`.
///
/// (The full ISDA 2006 §4.16(f) text gives seven sub-clauses; the rules
/// above implement the canonical "Bond Basis" interpretation as used in
/// `QuantLib`'s `Thirty360::BondBasis` and verified against ISDA's worked
/// examples.)
fn thirty_360_bond_basis(d1: Date, d2: Date) -> f64 {
    let (y1, m1, day1) = civil_from_days(d1.serial());
    let (y2, m2, day2) = civil_from_days(d2.serial());
    let mut dd1 = day1;
    let mut dd2 = day2;
    if dd1 == 31 {
        dd1 = 30;
    }
    if dd2 == 31 && dd1 == 30 {
        dd2 = 30;
    }
    let dy = y2 - y1;
    let dm = i32::try_from(m2).unwrap_or(0) - i32::try_from(m1).unwrap_or(0);
    let dd = i32::try_from(dd2).unwrap_or(0) - i32::try_from(dd1).unwrap_or(0);
    f64::from(360 * dy + 30 * dm + dd) / 360.0
}

/// 30E/360 — ISDA 2006 §4.16(g).
///
/// Both `D1` and `D2` are clipped to 30 unconditionally; then the standard
/// 30/360 formula applies.
fn thirty_360_e(d1: Date, d2: Date) -> f64 {
    let (y1, m1, day1) = civil_from_days(d1.serial());
    let (y2, m2, day2) = civil_from_days(d2.serial());
    let dd1 = day1.min(30);
    let dd2 = day2.min(30);
    let dy = y2 - y1;
    let dm = i32::try_from(m2).unwrap_or(0) - i32::try_from(m1).unwrap_or(0);
    let dd = i32::try_from(dd2).unwrap_or(0) - i32::try_from(dd1).unwrap_or(0);
    f64::from(360 * dy + 30 * dm + dd) / 360.0
}

/// Actual / Actual (ISDA) — ISDA 2006 §4.16(b).
///
/// Splits the date range into the portion that falls in leap years (366
/// denominator) and the portion in non-leap years (365 denominator), and
/// sums the two ratios.
fn act_act_isda(d1: Date, d2: Date) -> f64 {
    let y1 = d1.year();
    let y2 = d2.year();
    if y1 == y2 {
        let denom = if is_leap_year(y1) { 366.0 } else { 365.0 };
        return f64::from(d1.days_between(d2)) / denom;
    }
    // First partial year: d1 .. (y1 + 1)-01-01.
    let next_y1 = Date(days_from_civil(y1 + 1, 1, 1));
    let days_in_y1 = if is_leap_year(y1) { 366.0 } else { 365.0 };
    let first = f64::from(d1.days_between(next_y1)) / days_in_y1;
    // Last partial year: y2-01-01 .. d2.
    let start_y2 = Date(days_from_civil(y2, 1, 1));
    let days_in_y2 = if is_leap_year(y2) { 366.0 } else { 365.0 };
    let last = f64::from(start_y2.days_between(d2)) / days_in_y2;
    // Whole years between y1+1 and y2-1 inclusive each contribute 1.0.
    let middle = if y2 - y1 >= 2 {
        f64::from(y2 - y1 - 1)
    } else {
        0.0
    };
    first + middle + last
}

// ─── Compounding ─────────────────────────────────────────────────────────────

/// Compounding convention for converting between discount factors and zero
/// rates.
///
/// For time `t > 0` and zero rate `r`:
///
/// | Variant | Discount factor | Inverse |
/// |---|---|---|
/// | `Simple` | `D = 1 / (1 + r*t)` | `r = (1/D - 1) / t` |
/// | `Continuous` | `D = exp(-r*t)` | `r = -ln(D) / t` |
/// | `Periodic { n }` | `D = (1 + r/n)^(-n*t)` | `r = n * (D^(-1/(n*t)) - 1)` |
///
/// At `t = 0` the discount factor is exactly `1` for any rate, and the
/// "rate implied by `D = 1`" is `0`. All other queries with `t = 0` are
/// errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compounding {
    /// Simple interest: `D = 1 / (1 + r * t)`.
    Simple,
    /// Continuously compounded: `D = exp(-r * t)`.
    Continuous,
    /// Periodic compounding with `n` periods per year.
    Periodic {
        /// Number of compounding periods per year (`1` annual, `2`
        /// semi-annual, `4` quarterly, `12` monthly).
        periods_per_year: u32,
    },
}

impl Compounding {
    /// Returns the discount factor implied by a zero `rate` over time `t`.
    ///
    /// # Errors
    ///
    /// - [`TypeError::NonFinite`] if `rate` or `t` is not finite.
    /// - [`TypeError::NonPositiveRange`] if `t < 0`.
    /// - [`TypeError::InvalidTenor`] if [`Compounding::Periodic`] has
    ///   `periods_per_year == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::Compounding;
    ///
    /// let d = Compounding::Continuous.discount_from_rate(0.05, 2.0).unwrap();
    /// assert!((d - (-0.10_f64).exp()).abs() < 1e-15);
    /// ```
    pub fn discount_from_rate(self, rate: f64, t: f64) -> Result<f64, TypeError> {
        if !rate.is_finite() {
            return Err(TypeError::NonFinite { name: "rate" });
        }
        if !t.is_finite() {
            return Err(TypeError::NonFinite { name: "t" });
        }
        if t < 0.0 {
            return Err(TypeError::NonPositiveRange);
        }
        if t == 0.0 {
            return Ok(1.0);
        }
        match self {
            Self::Simple => Ok(1.0 / (1.0 + rate * t)),
            Self::Continuous => Ok((-rate * t).exp()),
            Self::Periodic { periods_per_year } => {
                if periods_per_year == 0 {
                    return Err(TypeError::InvalidTenor {
                        reason: "Periodic compounding requires periods_per_year > 0",
                    });
                }
                let n = f64::from(periods_per_year);
                Ok((1.0 + rate / n).powf(-n * t))
            }
        }
    }

    /// Returns the zero rate implied by a `discount` factor over time `t`.
    ///
    /// At `t = 0` and `discount = 1` the function returns `0.0`. Any other
    /// `t = 0` query is rejected because the rate is undefined.
    ///
    /// # Errors
    ///
    /// - [`TypeError::NonFinite`] if `discount` or `t` is not finite.
    /// - [`TypeError::NonPositiveRange`] if `t < 0`, or if `t == 0` and
    ///   `discount != 1.0`.
    /// - [`TypeError::InvalidTenor`] if `discount <= 0` (the rate is
    ///   undefined / infinite), or if [`Compounding::Periodic`] has
    ///   `periods_per_year == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::Compounding;
    ///
    /// let r = Compounding::Continuous
    ///     .rate_from_discount((-0.10_f64).exp(), 2.0)
    ///     .unwrap();
    /// assert!((r - 0.05).abs() < 1e-15);
    /// ```
    pub fn rate_from_discount(self, discount: f64, t: f64) -> Result<f64, TypeError> {
        if !discount.is_finite() {
            return Err(TypeError::NonFinite { name: "discount" });
        }
        if !t.is_finite() {
            return Err(TypeError::NonFinite { name: "t" });
        }
        if t < 0.0 {
            return Err(TypeError::NonPositiveRange);
        }
        if t == 0.0 {
            if (discount - 1.0).abs() < f64::EPSILON {
                return Ok(0.0);
            }
            return Err(TypeError::NonPositiveRange);
        }
        if discount <= 0.0 {
            return Err(TypeError::InvalidTenor {
                reason: "discount must be strictly positive",
            });
        }
        match self {
            Self::Simple => Ok((1.0 / discount - 1.0) / t),
            Self::Continuous => Ok(-discount.ln() / t),
            Self::Periodic { periods_per_year } => {
                if periods_per_year == 0 {
                    return Err(TypeError::InvalidTenor {
                        reason: "Periodic compounding requires periods_per_year > 0",
                    });
                }
                let n = f64::from(periods_per_year);
                Ok(n * (discount.powf(-1.0 / (n * t)) - 1.0))
            }
        }
    }
}

// ─── Frequency ───────────────────────────────────────────────────────────────

/// A payment frequency (used by swap legs and ICMA day-counts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Frequency {
    /// One payment per year.
    Annual,
    /// Two payments per year.
    SemiAnnual,
    /// Four payments per year.
    Quarterly,
    /// Twelve payments per year.
    Monthly,
    /// A single payment at maturity.
    OnceAtMaturity,
}

impl Frequency {
    /// Returns the number of payments per year, or `0` for
    /// [`Frequency::OnceAtMaturity`].
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::types::Frequency;
    ///
    /// assert_eq!(Frequency::Quarterly.periods_per_year(), 4);
    /// assert_eq!(Frequency::OnceAtMaturity.periods_per_year(), 0);
    /// ```
    #[must_use]
    #[inline]
    pub const fn periods_per_year(self) -> u32 {
        match self {
            Self::Annual => 1,
            Self::SemiAnnual => 2,
            Self::Quarterly => 4,
            Self::Monthly => 12,
            Self::OnceAtMaturity => 0,
        }
    }
}

// ─── BusinessDayConvention ───────────────────────────────────────────────────

/// Business-day-adjustment convention.
///
/// Documentation-only enum: holiday calendars are jurisdiction-specific (NYC,
/// LON, TARGET, ...), version-dependent, and intentionally out-of-scope. The
/// crate accepts already-adjusted dates; callers compose with their own
/// calendar and apply one of the conventions below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BusinessDayConvention {
    /// No adjustment.
    Unadjusted,
    /// Roll forward to the next business day.
    Following,
    /// Roll forward unless the next business day is in a new month, in which
    /// case roll backwards.
    ModifiedFollowing,
    /// Roll backward to the previous business day.
    Preceding,
    /// Roll backwards unless the previous business day is in a previous
    /// month, in which case roll forwards.
    ModifiedPreceding,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Date ────────────────────────────────────────────────────────────

    #[test]
    fn date_epoch_is_serial_zero() {
        let d = Date::from_ymd(1970, 1, 1).unwrap();
        assert_eq!(d.serial(), 0);
    }

    #[test]
    fn date_from_serial_roundtrip() {
        let d = Date::from_serial(0);
        assert_eq!((d.year(), d.month(), d.day()), (1970, 1, 1));
    }

    #[test]
    fn date_y2k() {
        let d = Date::from_ymd(2000, 1, 1).unwrap();
        // 30 years after 1970-01-01: 30*365 + 7 leap days (1972, 1976, 1980,
        // 1984, 1988, 1992, 1996, 2000 -> 8; but 2000 itself not yet counted)
        // = 30*365 + 7 = 10957.
        assert_eq!(d.serial(), 10_957);
        assert_eq!((d.year(), d.month(), d.day()), (2000, 1, 1));
    }

    #[test]
    fn date_leap_year_feb_29_2000() {
        let d = Date::from_ymd(2000, 2, 29).unwrap();
        assert_eq!((d.year(), d.month(), d.day()), (2000, 2, 29));
    }

    #[test]
    fn date_non_leap_feb_29_2100_rejected() {
        // 2100 is divisible by 100 but not 400 -> not a leap year.
        let err = Date::from_ymd(2100, 2, 29).unwrap_err();
        assert!(matches!(err, TypeError::InvalidDate { .. }));
    }

    #[test]
    fn date_non_leap_feb_29_1900_rejected() {
        // 1900 is divisible by 100 but not 400 -> not a leap year.
        let err = Date::from_ymd(1900, 2, 29).unwrap_err();
        assert!(matches!(err, TypeError::InvalidDate { .. }));
    }

    #[test]
    fn date_leap_year_2000_has_366_days() {
        let jan = Date::from_ymd(2000, 1, 1).unwrap();
        let dec = Date::from_ymd(2000, 12, 31).unwrap();
        assert_eq!(jan.days_between(dec), 365);
    }

    #[test]
    fn date_year_2100_has_365_days() {
        let jan = Date::from_ymd(2100, 1, 1).unwrap();
        let dec = Date::from_ymd(2100, 12, 31).unwrap();
        assert_eq!(jan.days_between(dec), 364);
    }

    #[test]
    fn date_year_1900_has_365_days() {
        let jan = Date::from_ymd(1900, 1, 1).unwrap();
        let dec = Date::from_ymd(1900, 12, 31).unwrap();
        assert_eq!(jan.days_between(dec), 364);
    }

    #[test]
    fn date_year_2400_is_leap() {
        // 2400 divisible by 400 -> leap.
        assert!(Date::from_ymd(2400, 2, 29).is_ok());
    }

    #[test]
    fn date_invalid_month_rejected() {
        assert!(Date::from_ymd(2024, 0, 15).is_err());
        assert!(Date::from_ymd(2024, 13, 15).is_err());
    }

    #[test]
    fn date_invalid_day_rejected() {
        assert!(Date::from_ymd(2024, 1, 0).is_err());
        assert!(Date::from_ymd(2024, 4, 31).is_err()); // April has 30 days
        assert!(Date::from_ymd(2023, 2, 29).is_err()); // 2023 not a leap
    }

    #[test]
    fn date_roundtrip_representative_set() {
        // Representative dates: epoch boundaries, leap years (positive and
        // negative), century boundaries, and decade endpoints.
        let dates: [(i32, u32, u32); 32] = [
            (1900, 1, 1),
            (1900, 2, 28),
            (1900, 12, 31),
            (1904, 2, 29),
            (1969, 12, 31),
            (1970, 1, 1),
            (1970, 1, 2),
            (1972, 2, 29),
            (1999, 12, 31),
            (2000, 1, 1),
            (2000, 2, 29),
            (2000, 12, 31),
            (2001, 1, 1),
            (2004, 2, 29),
            (2008, 2, 29),
            (2012, 2, 29),
            (2016, 2, 29),
            (2019, 12, 31),
            (2020, 1, 1),
            (2020, 2, 29),
            (2020, 12, 31),
            (2023, 12, 31),
            (2024, 1, 1),
            (2024, 2, 29),
            (2024, 7, 4),
            (2024, 12, 31),
            (2100, 1, 1),
            (2100, 2, 28),
            (2100, 12, 31),
            (2200, 6, 15),
            (2400, 2, 29),
            (2500, 12, 31),
        ];
        for &(y, m, d) in &dates {
            let date = Date::from_ymd(y, m, d).unwrap();
            assert_eq!(date.year(), y, "year mismatch for {y}-{m}-{d}");
            assert_eq!(date.month(), m, "month mismatch for {y}-{m}-{d}");
            assert_eq!(date.day(), d, "day mismatch for {y}-{m}-{d}");
        }
    }

    #[test]
    fn date_roundtrip_serial_iter() {
        // Walk a year by serial: every offset converts back consistently.
        let start = Date::from_ymd(2024, 1, 1).unwrap();
        for offset in 0..400 {
            let d = start.add_days(offset);
            let recon = Date::from_ymd(d.year(), d.month(), d.day()).unwrap();
            assert_eq!(d, recon);
        }
    }

    #[test]
    fn date_add_days_signed() {
        let d = Date::from_ymd(2024, 3, 1).unwrap();
        let prev = d.add_days(-1);
        // 2024 is a leap year -> 2024-02-29.
        assert_eq!((prev.year(), prev.month(), prev.day()), (2024, 2, 29));
    }

    #[test]
    fn date_days_between_signed() {
        let a = Date::from_ymd(2024, 1, 1).unwrap();
        let b = Date::from_ymd(2024, 1, 11).unwrap();
        assert_eq!(a.days_between(b), 10);
        assert_eq!(b.days_between(a), -10);
    }

    #[test]
    fn date_ordering() {
        let a = Date::from_ymd(2024, 1, 1).unwrap();
        let b = Date::from_ymd(2024, 6, 1).unwrap();
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn date_copy_eq_hash() {
        let d = Date::from_ymd(2024, 1, 1).unwrap();
        let copy = d;
        assert_eq!(d, copy);
        let mut set = std::collections::HashSet::new();
        set.insert(d);
        assert!(set.contains(&copy));
    }

    // ─── Tenor ───────────────────────────────────────────────────────────

    #[test]
    fn tenor_days() {
        let t = Tenor::new(7, TenorUnit::Days);
        let start = Date::from_ymd(2024, 1, 1).unwrap();
        let end = t.add_to(start);
        assert_eq!((end.year(), end.month(), end.day()), (2024, 1, 8));
    }

    #[test]
    fn tenor_weeks() {
        let t = Tenor::new(2, TenorUnit::Weeks);
        let start = Date::from_ymd(2024, 1, 1).unwrap();
        let end = t.add_to(start);
        assert_eq!((end.year(), end.month(), end.day()), (2024, 1, 15));
    }

    #[test]
    fn tenor_months_end_of_month() {
        // Jan 31 + 1M = Feb 29 (2024 is leap).
        let t = Tenor::new(1, TenorUnit::Months);
        let start = Date::from_ymd(2024, 1, 31).unwrap();
        let end = t.add_to(start);
        assert_eq!((end.year(), end.month(), end.day()), (2024, 2, 29));
    }

    #[test]
    fn tenor_months_non_leap() {
        let t = Tenor::new(1, TenorUnit::Months);
        let start = Date::from_ymd(2023, 1, 31).unwrap();
        let end = t.add_to(start);
        // Non-leap Feb -> clipped to Feb 28.
        assert_eq!((end.year(), end.month(), end.day()), (2023, 2, 28));
    }

    #[test]
    fn tenor_months_cross_year_back() {
        let t = Tenor::new(-1, TenorUnit::Months);
        let start = Date::from_ymd(2024, 1, 15).unwrap();
        let end = t.add_to(start);
        assert_eq!((end.year(), end.month(), end.day()), (2023, 12, 15));
    }

    #[test]
    fn tenor_years() {
        let t = Tenor::new(5, TenorUnit::Years);
        let start = Date::from_ymd(2020, 6, 15).unwrap();
        let end = t.add_to(start);
        assert_eq!((end.year(), end.month(), end.day()), (2025, 6, 15));
    }

    #[test]
    fn tenor_years_leap_day() {
        // Feb 29, 2024 + 1Y = Feb 28, 2025 (clipped).
        let t = Tenor::new(1, TenorUnit::Years);
        let start = Date::from_ymd(2024, 2, 29).unwrap();
        let end = t.add_to(start);
        assert_eq!((end.year(), end.month(), end.day()), (2025, 2, 28));
    }

    #[test]
    fn tenor_constructor_fields() {
        let t = Tenor::new(3, TenorUnit::Months);
        assert_eq!(t.count, 3);
        assert_eq!(t.unit, TenorUnit::Months);
    }

    #[test]
    fn tenor_unit_copy_eq() {
        let u = TenorUnit::Days;
        let copy = u;
        assert_eq!(u, copy);
    }

    // ─── Daycount ────────────────────────────────────────────────────────

    #[test]
    fn daycount_act360_isda_example() {
        // ISDA 2006 §4.16(e) and the standard worked example:
        // 2003-11-01 to 2004-05-01 = 182 days => 182/360.
        let d1 = Date::from_ymd(2003, 11, 1).unwrap();
        let d2 = Date::from_ymd(2004, 5, 1).unwrap();
        let tau = Daycount::Act360.year_fraction(d1, d2).unwrap();
        assert!((tau - 182.0_f64 / 360.0).abs() < 1e-15);
    }

    #[test]
    fn daycount_act365f_example() {
        let d1 = Date::from_ymd(2024, 1, 1).unwrap();
        let d2 = Date::from_ymd(2024, 7, 1).unwrap();
        let tau = Daycount::Act365F.year_fraction(d1, d2).unwrap();
        assert!((tau - 182.0_f64 / 365.0).abs() < 1e-15);
    }

    #[test]
    fn daycount_act_act_isda_single_year() {
        // Within 2024 (leap): 366 denominator.
        let d1 = Date::from_ymd(2024, 1, 1).unwrap();
        let d2 = Date::from_ymd(2024, 7, 1).unwrap();
        let tau = Daycount::ActActIsda.year_fraction(d1, d2).unwrap();
        assert!((tau - 182.0_f64 / 366.0).abs() < 1e-15);
    }

    #[test]
    fn daycount_act_act_isda_single_year_non_leap() {
        let d1 = Date::from_ymd(2023, 1, 1).unwrap();
        let d2 = Date::from_ymd(2023, 7, 1).unwrap();
        let tau = Daycount::ActActIsda.year_fraction(d1, d2).unwrap();
        assert!((tau - 181.0_f64 / 365.0).abs() < 1e-15);
    }

    #[test]
    fn daycount_act_act_isda_cross_year() {
        // 2003-11-01 to 2004-05-01: 61 days in 2003 (non-leap, Nov 1 -> Jan 1
        // exclusive) + 121 in 2004 (leap, Jan 1 -> May 1 exclusive).
        // ISDA 2006 §4.16(b) splits the period at the calendar boundary.
        let d1 = Date::from_ymd(2003, 11, 1).unwrap();
        let d2 = Date::from_ymd(2004, 5, 1).unwrap();
        let tau = Daycount::ActActIsda.year_fraction(d1, d2).unwrap();
        let expected = 61.0_f64 / 365.0 + 121.0_f64 / 366.0;
        assert!((tau - expected).abs() < 1e-15);
    }

    #[test]
    fn daycount_act_act_isda_multi_year() {
        // 2003-06-15 to 2007-06-15: full years 2004, 2005, 2006 each ~1.0;
        // first stub 2003-06-15 -> 2004-01-01 = 200 days / 365; last
        // stub 2007-01-01 -> 2007-06-15 = 165 days / 365.
        let d1 = Date::from_ymd(2003, 6, 15).unwrap();
        let d2 = Date::from_ymd(2007, 6, 15).unwrap();
        let tau = Daycount::ActActIsda.year_fraction(d1, d2).unwrap();
        // The full middle years are: 2004 (leap, full), 2005 (non), 2006 (non).
        // We add 3.0 for those.
        let first = f64::from(d1.days_between(Date::from_ymd(2004, 1, 1).unwrap())) / 365.0;
        let last = f64::from(Date::from_ymd(2007, 1, 1).unwrap().days_between(d2)) / 365.0;
        let expected = first + 3.0 + last;
        assert!((tau - expected).abs() < 1e-14);
    }

    #[test]
    fn daycount_thirty_360_e() {
        // 30E/360: Feb 28 -> 30 always, etc.
        // Example: 2003-11-01 to 2004-05-01:
        // dy = 1, dm = -6, dd = 0  =>  360 + (-6*30) + 0 = 180.
        let d1 = Date::from_ymd(2003, 11, 1).unwrap();
        let d2 = Date::from_ymd(2004, 5, 1).unwrap();
        let tau = Daycount::Thirty360E.year_fraction(d1, d2).unwrap();
        assert!((tau - 180.0_f64 / 360.0).abs() < 1e-15);
    }

    #[test]
    fn daycount_thirty_360_e_clip() {
        // Both day-clipped at 30: 2024-01-31 -> 2024-05-31 = 4 months = 120/360.
        let d1 = Date::from_ymd(2024, 1, 31).unwrap();
        let d2 = Date::from_ymd(2024, 5, 31).unwrap();
        let tau = Daycount::Thirty360E.year_fraction(d1, d2).unwrap();
        assert!((tau - 120.0_f64 / 360.0).abs() < 1e-15);
    }

    #[test]
    fn daycount_thirty_360_bb_isda_example() {
        // ISDA 2006 §4.16(f), worked example "Interest period 28 Feb 2007 to
        // 31 Aug 2007 (Bond Basis)":
        // dy = 0, dm = 6, dd = D2_adj - D1_adj where D1 = 28, D2_adj = 31->30
        // when D1 = 30 or 31. Since D1 = 28 (not 30/31), D2 stays 31.
        // => 0 + 180 + (31 - 28) = 183 days; tau = 183/360.
        let d1 = Date::from_ymd(2007, 2, 28).unwrap();
        let d2 = Date::from_ymd(2007, 8, 31).unwrap();
        let tau = Daycount::Thirty360BondBasis.year_fraction(d1, d2).unwrap();
        assert!((tau - 183.0_f64 / 360.0).abs() < 1e-15);
    }

    #[test]
    fn daycount_thirty_360_bb_d1_is_31() {
        // D1=31 -> 30. 2024-01-31 -> 2024-07-31 -> D2 also 31, since D1=30
        // after rule (1), D2 becomes 30.
        // dy=0, dm=6, dd = 30-30 = 0; tau = 180/360 = 0.5.
        let d1 = Date::from_ymd(2024, 1, 31).unwrap();
        let d2 = Date::from_ymd(2024, 7, 31).unwrap();
        let tau = Daycount::Thirty360BondBasis.year_fraction(d1, d2).unwrap();
        assert!((tau - 0.5).abs() < 1e-15);
    }

    #[test]
    fn daycount_thirty_360_bb_d2_is_31_d1_not_30() {
        // D1 = 15, D2 = 31: D2 stays 31. dy=0, dm=6, dd=31-15=16 => 196/360.
        let d1 = Date::from_ymd(2024, 1, 15).unwrap();
        let d2 = Date::from_ymd(2024, 7, 31).unwrap();
        let tau = Daycount::Thirty360BondBasis.year_fraction(d1, d2).unwrap();
        assert!((tau - 196.0_f64 / 360.0).abs() < 1e-15);
    }

    #[test]
    fn daycount_act_act_icma_quarterly() {
        let d1 = Date::from_ymd(2024, 1, 1).unwrap();
        let d2 = Date::from_ymd(2024, 4, 1).unwrap();
        let tau = Daycount::ActActIcma {
            coupons_per_year: 4,
        }
        .year_fraction(d1, d2)
        .unwrap();
        assert!((tau - 0.25).abs() < 1e-15);
    }

    #[test]
    fn daycount_act_act_icma_zero_freq_rejected() {
        let d1 = Date::from_ymd(2024, 1, 1).unwrap();
        let d2 = Date::from_ymd(2024, 4, 1).unwrap();
        let err = Daycount::ActActIcma {
            coupons_per_year: 0,
        }
        .year_fraction(d1, d2)
        .unwrap_err();
        assert!(matches!(err, TypeError::InvalidTenor { .. }));
    }

    #[test]
    fn daycount_business252_rejected() {
        let d1 = Date::from_ymd(2024, 1, 1).unwrap();
        let d2 = Date::from_ymd(2024, 4, 1).unwrap();
        let err = Daycount::Business252.year_fraction(d1, d2).unwrap_err();
        match err {
            TypeError::InvalidTenor { reason } => {
                assert!(reason.contains("Business252"));
            }
            other => panic!("unexpected variant {other:?}"),
        }
    }

    #[test]
    fn daycount_negative_range_rejected() {
        let d1 = Date::from_ymd(2024, 6, 1).unwrap();
        let d2 = Date::from_ymd(2024, 1, 1).unwrap();
        let err = Daycount::Act360.year_fraction(d1, d2).unwrap_err();
        assert!(matches!(err, TypeError::NonPositiveRange));
    }

    #[test]
    fn daycount_zero_range_ok() {
        let d = Date::from_ymd(2024, 1, 1).unwrap();
        let tau = Daycount::Act360.year_fraction(d, d).unwrap();
        assert!((tau - 0.0).abs() < 1e-15);
    }

    #[test]
    fn daycount_copy_eq() {
        let dc = Daycount::Act360;
        let copy = dc;
        assert_eq!(dc, copy);
    }

    // ─── Compounding ─────────────────────────────────────────────────────

    #[test]
    fn compounding_continuous_roundtrip() {
        let r = 0.05;
        let t = 2.0;
        let d = Compounding::Continuous.discount_from_rate(r, t).unwrap();
        assert!((d - (-r * t).exp()).abs() < 1e-15);
        let r_back = Compounding::Continuous.rate_from_discount(d, t).unwrap();
        assert!((r - r_back).abs() < 1e-12);
    }

    #[test]
    fn compounding_simple_roundtrip() {
        let r = 0.03;
        let t = 0.5;
        let d = Compounding::Simple.discount_from_rate(r, t).unwrap();
        assert!((d - 1.0 / (1.0 + r * t)).abs() < 1e-15);
        let r_back = Compounding::Simple.rate_from_discount(d, t).unwrap();
        assert!((r - r_back).abs() < 1e-12);
    }

    #[test]
    fn compounding_periodic_roundtrip() {
        let r = 0.06;
        let t = 3.0;
        let comp = Compounding::Periodic {
            periods_per_year: 2,
        };
        let d = comp.discount_from_rate(r, t).unwrap();
        assert!((d - (1.0_f64 + 0.03).powi(-6)).abs() < 1e-12);
        let r_back = comp.rate_from_discount(d, t).unwrap();
        assert!((r - r_back).abs() < 1e-12);
    }

    #[test]
    fn compounding_zero_time_discount_is_one() {
        let d = Compounding::Continuous
            .discount_from_rate(0.05, 0.0)
            .unwrap();
        assert!((d - 1.0).abs() < 1e-15);
    }

    #[test]
    fn compounding_zero_time_unit_discount_gives_zero_rate() {
        let r = Compounding::Continuous
            .rate_from_discount(1.0, 0.0)
            .unwrap();
        assert!((r - 0.0).abs() < 1e-15);
    }

    #[test]
    fn compounding_rejects_non_finite_rate() {
        let err = Compounding::Continuous
            .discount_from_rate(f64::NAN, 1.0)
            .unwrap_err();
        assert!(matches!(err, TypeError::NonFinite { name: "rate" }));
    }

    #[test]
    fn compounding_rejects_non_finite_t() {
        let err = Compounding::Continuous
            .discount_from_rate(0.05, f64::INFINITY)
            .unwrap_err();
        assert!(matches!(err, TypeError::NonFinite { name: "t" }));
    }

    #[test]
    fn compounding_rejects_negative_t() {
        let err = Compounding::Continuous
            .discount_from_rate(0.05, -1.0)
            .unwrap_err();
        assert!(matches!(err, TypeError::NonPositiveRange));
    }

    #[test]
    fn compounding_rejects_zero_periods() {
        let err = Compounding::Periodic {
            periods_per_year: 0,
        }
        .discount_from_rate(0.05, 1.0)
        .unwrap_err();
        assert!(matches!(err, TypeError::InvalidTenor { .. }));
        let err = Compounding::Periodic {
            periods_per_year: 0,
        }
        .rate_from_discount(0.95, 1.0)
        .unwrap_err();
        assert!(matches!(err, TypeError::InvalidTenor { .. }));
    }

    #[test]
    fn compounding_rate_from_discount_rejects_non_positive_discount() {
        let err = Compounding::Continuous
            .rate_from_discount(0.0, 1.0)
            .unwrap_err();
        assert!(matches!(err, TypeError::InvalidTenor { .. }));
        let err = Compounding::Continuous
            .rate_from_discount(-0.5, 1.0)
            .unwrap_err();
        assert!(matches!(err, TypeError::InvalidTenor { .. }));
    }

    #[test]
    fn compounding_rate_from_discount_rejects_non_finite() {
        let err = Compounding::Continuous
            .rate_from_discount(f64::NAN, 1.0)
            .unwrap_err();
        assert!(matches!(err, TypeError::NonFinite { name: "discount" }));
        let err = Compounding::Continuous
            .rate_from_discount(0.95, f64::NAN)
            .unwrap_err();
        assert!(matches!(err, TypeError::NonFinite { name: "t" }));
    }

    #[test]
    fn compounding_rate_from_discount_rejects_negative_t() {
        let err = Compounding::Continuous
            .rate_from_discount(0.95, -1.0)
            .unwrap_err();
        assert!(matches!(err, TypeError::NonPositiveRange));
    }

    #[test]
    fn compounding_rate_from_discount_rejects_zero_t_non_unit() {
        let err = Compounding::Continuous
            .rate_from_discount(0.95, 0.0)
            .unwrap_err();
        assert!(matches!(err, TypeError::NonPositiveRange));
    }

    // ─── Frequency ───────────────────────────────────────────────────────

    #[test]
    fn frequency_periods_per_year() {
        assert_eq!(Frequency::Annual.periods_per_year(), 1);
        assert_eq!(Frequency::SemiAnnual.periods_per_year(), 2);
        assert_eq!(Frequency::Quarterly.periods_per_year(), 4);
        assert_eq!(Frequency::Monthly.periods_per_year(), 12);
        assert_eq!(Frequency::OnceAtMaturity.periods_per_year(), 0);
    }

    #[test]
    fn frequency_copy_eq() {
        let f = Frequency::Quarterly;
        let copy = f;
        assert_eq!(f, copy);
    }

    // ─── BusinessDayConvention ───────────────────────────────────────────

    #[test]
    fn business_day_convention_copy_eq() {
        let c = BusinessDayConvention::ModifiedFollowing;
        let copy = c;
        assert_eq!(c, copy);
    }

    #[test]
    fn business_day_convention_debug_includes_variant() {
        let s = format!("{:?}", BusinessDayConvention::Following);
        assert!(s.contains("Following"));
    }
}
