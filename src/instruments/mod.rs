// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Bootstrap instruments — deposits, FRAs, futures, vanilla and OIS swaps,
//! basis swaps.
//!
//! Every instrument carries the dates and quote needed to evaluate it, plus
//! the day-count convention. The bootstrap engine drives a candidate
//! discount factor at the instrument's pillar (its latest date) so that the
//! instrument's residual against the in-progress curve drops to zero.
//!
//! Instruments are exposed at the top level via the [`Instrument`] enum;
//! the bootstrap engine dispatches on the enum, not on a trait object. The
//! internal trait `InstrumentLike` (private to the crate, wired by the
//! bootstrap engine) gives the engine a uniform handle into every variant.
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §2.
//! - ISDA, *2006 ISDA Definitions*, §4.6 and §7.1.

use crate::errors::BootstrapError;
use crate::interpolation::{Interpolator, LogLinear};
use crate::types::{Date, Daycount};

pub mod basis_swap;
pub mod bond;
pub mod deposit;
pub mod fra;
pub mod future;
pub mod ois_swap;
pub mod schedule;
pub mod swap_fixed_float;

pub use basis_swap::{BasisLeg, BasisSwap};
pub use bond::Bond;
pub use deposit::Deposit;
pub use fra::Fra;
pub use future::Future;
pub use ois_swap::OisSwap;
pub use schedule::SwapSchedule;
pub use swap_fixed_float::SwapFixedFloat;

/// The set of instruments that the bootstrap engine knows how to re-price.
///
/// Marked `#[non_exhaustive]` so adding variants is non-breaking. The enum
/// dispatches `InstrumentLike::pillar` and `InstrumentLike::residual` to the
/// variant's underlying instrument; the bootstrap engine iterates these to
/// drive each instrument's residual to zero.
///
/// # Examples
///
/// ```
/// use regit_curves::instruments::{Deposit, Instrument};
/// use regit_curves::types::{Date, Daycount};
///
/// let fixing  = Date::from_ymd(2024, 1, 2).unwrap();
/// let payment = Date::from_ymd(2024, 4, 2).unwrap();
/// let dep = Deposit::new(fixing, payment, 0.05, Daycount::Act360).unwrap();
/// let inst = Instrument::Deposit(dep);
/// matches!(inst, Instrument::Deposit(_));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Instrument {
    /// A coupon-bearing bond (see [`Bond`]).
    Bond(Bond),
    /// A money-market deposit (see [`Deposit`]).
    Deposit(Deposit),
    /// A forward-rate agreement (see [`Fra`]).
    Fra(Fra),
    /// A STIR future (see [`Future`]).
    Future(Future),
    /// A vanilla fixed-floating interest-rate swap (see [`SwapFixedFloat`]).
    SwapFixedFloat(SwapFixedFloat),
    /// An OIS (overnight-indexed) swap (see [`OisSwap`]).
    OisSwap(OisSwap),
    /// A tenor / cross-currency basis swap (see [`BasisSwap`]).
    BasisSwap(BasisSwap),
}

impl Instrument {
    /// Returns the instrument's pillar date — the latest date it constrains.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::{Deposit, Instrument};
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let fixing  = Date::from_ymd(2024, 1, 2).unwrap();
    /// let payment = Date::from_ymd(2024, 4, 2).unwrap();
    /// let dep = Deposit::new(fixing, payment, 0.05, Daycount::Act360).unwrap();
    /// let inst = Instrument::Deposit(dep);
    /// assert_eq!(inst.pillar(), payment);
    /// ```
    #[must_use]
    pub fn pillar(&self) -> Date {
        match self {
            Self::Bond(b) => InstrumentLike::pillar(b),
            Self::Deposit(d) => InstrumentLike::pillar(d),
            Self::Fra(f) => InstrumentLike::pillar(f),
            Self::Future(f) => InstrumentLike::pillar(f),
            Self::SwapFixedFloat(s) => InstrumentLike::pillar(s),
            Self::OisSwap(s) => InstrumentLike::pillar(s),
            Self::BasisSwap(b) => InstrumentLike::pillar(b),
        }
    }
}

/// Internal trait — every instrument variant implements it so the bootstrap
/// engine can dispatch uniformly.
///
/// Wired by the bootstrap engine. The trait is `pub(crate)` and users
/// construct instruments via the [`Instrument`] enum, not through the trait
/// directly.
#[allow(dead_code)]
pub(crate) trait InstrumentLike {
    /// The latest date this instrument constrains — used both to order
    /// instruments and to identify the curve pillar that the instrument
    /// pins.
    fn pillar(&self) -> Date;

    /// Re-prices the instrument against a [`CurveSnapshot`] and returns the
    /// residual. Zero at the bootstrap solution. The residual is on the
    /// same scale as the quoted price (rate units for rate-quoted
    /// instruments).
    ///
    /// # Errors
    ///
    /// Returns a [`BootstrapError`] when the curve snapshot or the
    /// instrument's day-count convention cannot deliver a finite residual.
    fn residual(
        &self,
        reference_date: Date,
        curve: &CurveSnapshot<'_>,
    ) -> Result<f64, BootstrapError>;
}

/// Minimal handle into an in-progress discount curve.
///
/// `CurveSnapshot` is the **internal** view the bootstrap engine hands to
/// an instrument's residual computation. It is deliberately minimal: the
/// public [`crate::curves::DiscountCurve`] is the user-facing curve type,
/// while `CurveSnapshot` is the thin in-progress view used during the
/// bootstrap iteration so that each instrument's `residual` is a real,
/// testable computation against the partially-built table.
///
/// The discount-factor lookup is **flat-extrapolating piecewise log-linear**
/// in `t` (which is equivalent to piecewise-constant continuously-compounded
/// zero rate — see [`crate::interpolation::log_linear`]).
#[allow(dead_code)]
pub(crate) struct CurveSnapshot<'a> {
    /// Curve anchor.
    pub(crate) reference_date: Date,
    /// Day-count used to convert dates to year fractions on the t-axis.
    pub(crate) daycount: Daycount,
    /// Knot times (year fractions from `reference_date`).
    pub(crate) times: &'a [f64],
    /// Knot discount factors.
    pub(crate) discounts: &'a [f64],
}

impl CurveSnapshot<'_> {
    /// Looks up the curve's discount factor at year fraction `t` by
    /// flat-extrapolating piecewise log-linear interpolation through the
    /// `(times, discounts)` table.
    ///
    /// Returns `None` if the table is empty or shorter than two knots, or if
    /// the table fails the log-linear invariants (non-positive discount
    /// factor, non-monotone times). The public [`crate::curves::DiscountCurve`]
    /// enforces these invariants at construction time; this snapshot is the
    /// in-progress view the bootstrap engine uses before the final curve is
    /// finalised.
    pub(crate) fn discount_at(&self, t: f64) -> Option<f64> {
        if self.times.is_empty() || self.times.len() != self.discounts.len() {
            return None;
        }
        if self.times.len() == 1 {
            return Some(self.discounts[0]);
        }
        let knots: Vec<(f64, f64)> = self
            .times
            .iter()
            .zip(self.discounts.iter())
            .map(|(&t, &d)| (t, d))
            .collect();
        let interp = LogLinear::build(&knots).ok()?;
        Some(interp.eval(t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> Date {
        Date::from_ymd(y, m, day).unwrap()
    }

    #[test]
    fn instrument_pillar_dispatches_to_variant() {
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), 0.05, Daycount::Act360).unwrap();
        let inst = Instrument::Deposit(dep);
        assert_eq!(inst.pillar(), d(2024, 4, 2));
    }

    #[test]
    fn instrument_clone_eq() {
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), 0.05, Daycount::Act360).unwrap();
        let a = Instrument::Deposit(dep);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn instrument_debug_includes_variant() {
        let dep = Deposit::new(d(2024, 1, 2), d(2024, 4, 2), 0.05, Daycount::Act360).unwrap();
        let s = format!("{:?}", Instrument::Deposit(dep));
        assert!(s.contains("Deposit"));
    }

    #[test]
    fn curve_snapshot_discount_at_returns_none_on_empty() {
        let snap = CurveSnapshot {
            reference_date: d(2024, 1, 2),
            daycount: Daycount::Act360,
            times: &[],
            discounts: &[],
        };
        assert!(snap.discount_at(1.0).is_none());
    }

    #[test]
    fn curve_snapshot_discount_at_returns_single_when_one_knot() {
        let snap = CurveSnapshot {
            reference_date: d(2024, 1, 2),
            daycount: Daycount::Act360,
            times: &[0.0],
            discounts: &[1.0],
        };
        let v = snap.discount_at(5.0).unwrap();
        assert!((v - 1.0).abs() < 1e-15);
    }

    #[test]
    fn curve_snapshot_discount_at_log_linear_through_knots() {
        // Reproduce knots exactly.
        let times = [0.0_f64, 1.0, 2.0];
        let disc = [1.0_f64, 0.95, 0.90];
        let snap = CurveSnapshot {
            reference_date: d(2024, 1, 2),
            daycount: Daycount::Act360,
            times: &times,
            discounts: &disc,
        };
        for (&t, &d) in times.iter().zip(disc.iter()) {
            assert!((snap.discount_at(t).unwrap() - d).abs() < 1e-15);
        }
        // Midpoint of [0, 1]: geometric mean.
        let mid = snap.discount_at(0.5).unwrap();
        assert!((mid - 0.95_f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn curve_snapshot_discount_at_returns_none_on_mismatched_lengths() {
        let snap = CurveSnapshot {
            reference_date: d(2024, 1, 2),
            daycount: Daycount::Act360,
            times: &[0.0_f64, 1.0],
            discounts: &[1.0_f64],
        };
        assert!(snap.discount_at(0.5).is_none());
    }

    #[test]
    fn curve_snapshot_discount_at_flat_extrapolation() {
        let times = [0.0_f64, 1.0];
        let disc = [1.0_f64, 0.95];
        let snap = CurveSnapshot {
            reference_date: d(2024, 1, 2),
            daycount: Daycount::Act360,
            times: &times,
            discounts: &disc,
        };
        // Below first knot: flat at 1.0.
        assert!((snap.discount_at(-1.0).unwrap() - 1.0).abs() < 1e-15);
        // Above last knot: flat at 0.95.
        assert!((snap.discount_at(2.0).unwrap() - 0.95).abs() < 1e-15);
    }
}
