// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Typed error enums for the three failure domains of `regit-curves`.
//!
//! All failure paths return a typed `Result` — no `panic!()`, no `unwrap()`,
//! no string errors. Each variant carries enough context for the caller to
//! decide how to recover.
//!
//! Three enums separate the three failure domains:
//!
//! - [`TypeError`] — invalid `Date`, `Tenor`, or `Daycount` queries.
//! - [`CurveError`] — invalid curve construction or evaluation.
//! - [`BootstrapError`] — bootstrap engine failures (non-convergence,
//!   ordering, instrument-level invariants).
//!
//! Natural conversions are provided via `From`:
//!
//! - `TypeError -> CurveError` (variant [`CurveError::Type`])
//! - `TypeError -> BootstrapError` (variant [`BootstrapError::Type`])
//! - `CurveError -> BootstrapError` (variant [`BootstrapError::Curve`])
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006). Section 2 motivates
//!   the curve invariants enforced by [`CurveError`].
//! - ISDA, *2006 ISDA Definitions*, §4.16. Day-count edge cases that
//!   [`TypeError`] reports.

use core::fmt;

// ─── Type-construction errors ────────────────────────────────────────────────

/// Error returned when constructing or querying a basic numeric / temporal
/// type ([`Date`](crate::types::Date), [`Tenor`](crate::types::Tenor),
/// [`Daycount`](crate::types::Daycount)).
///
/// # Examples
///
/// ```
/// use regit_curves::errors::TypeError;
///
/// let err = TypeError::InvalidDate { year: 2023, month: 2, day: 30 };
/// assert_eq!(
///     format!("{err}"),
///     "invalid calendar date: 2023-02-30",
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeError {
    /// [`Date::from_ymd`](crate::types::Date::from_ymd) received an invalid
    /// calendar date (out-of-range month/day, or a day that does not exist
    /// in the given month — e.g. February 30).
    InvalidDate {
        /// The supplied year.
        year: i32,
        /// The supplied month.
        month: u32,
        /// The supplied day.
        day: u32,
    },
    /// A non-positive day count was requested where a positive value is
    /// required (e.g. day-count year fraction across a zero or negative
    /// range, or a degenerate compounding `t <= 0` for `rate_from_discount`).
    NonPositiveRange,
    /// A non-finite (`NaN` or infinite) number was supplied where a finite
    /// value is required.
    NonFinite {
        /// Human-readable name of the offending input.
        name: &'static str,
    },
    /// A frequency or tenor count is invalid (zero, negative for a unit that
    /// requires positive, or otherwise out of range — including
    /// `Daycount::Business252` queried without a calendar).
    InvalidTenor {
        /// Human-readable reason.
        reason: &'static str,
    },
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDate { year, month, day } => {
                write!(f, "invalid calendar date: {year:04}-{month:02}-{day:02}")
            }
            Self::NonPositiveRange => write!(f, "day-count range must be strictly positive"),
            Self::NonFinite { name } => write!(f, "input {name} must be a finite number"),
            Self::InvalidTenor { reason } => write!(f, "invalid tenor: {reason}"),
        }
    }
}

impl std::error::Error for TypeError {}

// ─── Curve construction / evaluation errors ──────────────────────────────────

/// Error returned when constructing or evaluating a discount curve.
///
/// # Examples
///
/// ```
/// use regit_curves::errors::CurveError;
///
/// let err = CurveError::TooFewNodes { found: 1 };
/// assert_eq!(
///     format!("{err}"),
///     "curve needs at least two nodes, found 1",
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurveError {
    /// Fewer than two nodes were supplied.
    TooFewNodes {
        /// Number of nodes that were supplied.
        found: usize,
    },
    /// Node times are not strictly increasing.
    NodesNotIncreasing {
        /// Index of the first node that breaks the invariant.
        at_index: usize,
    },
    /// A discount factor at a node was not strictly positive.
    NonPositiveDiscount {
        /// Index of the offending node.
        at_index: usize,
        /// The offending value.
        value: f64,
    },
    /// The first node is not at `t = 0` or its discount factor is not `1`.
    AnchorNotUnit,
    /// `t` is negative or non-finite.
    InvalidTime {
        /// The offending value of `t`.
        t: f64,
    },
    /// A day-count query failed.
    Type(TypeError),
    /// Two construction points have the same `t` (used in slope / spline
    /// construction).
    DuplicateNode {
        /// The duplicated time.
        t: f64,
    },
}

impl fmt::Display for CurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewNodes { found } => {
                write!(f, "curve needs at least two nodes, found {found}")
            }
            Self::NodesNotIncreasing { at_index } => {
                write!(f, "node times not strictly increasing at index {at_index}")
            }
            Self::NonPositiveDiscount { at_index, value } => {
                write!(
                    f,
                    "discount factor at node {at_index} must be positive, got {value}"
                )
            }
            Self::AnchorNotUnit => write!(f, "anchor node must be (t=0, D=1)"),
            Self::InvalidTime { t } => write!(f, "invalid time t = {t}"),
            Self::Type(e) => write!(f, "type error in curve query: {e}"),
            Self::DuplicateNode { t } => write!(f, "duplicate node at t = {t}"),
        }
    }
}

impl std::error::Error for CurveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Type(e) => Some(e),
            _ => None,
        }
    }
}

impl From<TypeError> for CurveError {
    fn from(e: TypeError) -> Self {
        Self::Type(e)
    }
}

// ─── Bootstrap engine errors ────────────────────────────────────────────────

/// Error returned by the bootstrap engine.
///
/// # Examples
///
/// ```
/// use regit_curves::errors::BootstrapError;
///
/// let err = BootstrapError::LegDidNotConverge { at_index: 3, residual: 1.2e-9 };
/// let msg = format!("{err}");
/// assert!(msg.contains("3"));
/// assert!(msg.contains("converge"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BootstrapError {
    /// Instruments are not ordered by their primary anchor time, or two
    /// instruments share an anchor with no resolution rule.
    InstrumentsNotOrdered {
        /// Index of the first instrument that breaks the ordering.
        at_index: usize,
    },
    /// An instrument's primary anchor is on or before the previous anchor.
    NonIncreasingAnchor {
        /// Index of the offending instrument.
        at_index: usize,
    },
    /// The leg solver did not converge to a discount factor that re-prices
    /// the instrument within `tolerance` after `max_iterations`.
    LegDidNotConverge {
        /// Index of the leg that failed to converge.
        at_index: usize,
        /// The final residual reached.
        residual: f64,
    },
    /// Brent could not bracket a root for the leg.
    NoBracket {
        /// Index of the leg with no bracket.
        at_index: usize,
    },
    /// An instrument quote violates a domain constraint (rate not finite,
    /// negative when a positive value is required, etc.).
    InvalidInstrument {
        /// Index of the offending instrument.
        at_index: usize,
        /// Human-readable reason.
        reason: &'static str,
    },
    /// A curve error surfaced while building the interim curve.
    Curve(CurveError),
    /// A type error surfaced.
    Type(TypeError),
}

impl fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InstrumentsNotOrdered { at_index } => {
                write!(f, "instruments not ordered at index {at_index}")
            }
            Self::NonIncreasingAnchor { at_index } => {
                write!(
                    f,
                    "instrument anchor not strictly increasing at index {at_index}"
                )
            }
            Self::LegDidNotConverge { at_index, residual } => {
                write!(
                    f,
                    "bootstrap leg {at_index} did not converge: residual {residual:e}"
                )
            }
            Self::NoBracket { at_index } => {
                write!(f, "bootstrap leg {at_index} could not bracket a root")
            }
            Self::InvalidInstrument { at_index, reason } => {
                write!(f, "invalid instrument at index {at_index}: {reason}")
            }
            Self::Curve(e) => write!(f, "curve error during bootstrap: {e}"),
            Self::Type(e) => write!(f, "type error during bootstrap: {e}"),
        }
    }
}

impl std::error::Error for BootstrapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Curve(e) => Some(e),
            Self::Type(e) => Some(e),
            _ => None,
        }
    }
}

impl From<TypeError> for BootstrapError {
    fn from(e: TypeError) -> Self {
        Self::Type(e)
    }
}

impl From<CurveError> for BootstrapError {
    fn from(e: CurveError) -> Self {
        Self::Curve(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── TypeError ───────────────────────────────────────────────────────

    #[test]
    fn type_error_display_invalid_date() {
        let err = TypeError::InvalidDate {
            year: 2023,
            month: 2,
            day: 30,
        };
        assert_eq!(format!("{err}"), "invalid calendar date: 2023-02-30");
    }

    #[test]
    fn type_error_display_non_positive_range() {
        let err = TypeError::NonPositiveRange;
        assert!(format!("{err}").contains("positive"));
    }

    #[test]
    fn type_error_display_non_finite() {
        let err = TypeError::NonFinite { name: "rate" };
        assert!(format!("{err}").contains("rate"));
        assert!(format!("{err}").contains("finite"));
    }

    #[test]
    fn type_error_display_invalid_tenor() {
        let err = TypeError::InvalidTenor {
            reason: "Business252 requires a calendar",
        };
        assert!(format!("{err}").contains("Business252"));
    }

    #[test]
    fn type_error_is_error_trait() {
        let err: &dyn std::error::Error = &TypeError::NonPositiveRange;
        assert!(err.source().is_none());
    }

    #[test]
    fn type_error_copy_eq_hash() {
        let err = TypeError::NonFinite { name: "rate" };
        let copy = err;
        assert_eq!(err, copy);
        // Hash is derived; we use it via a HashSet check.
        let mut set = std::collections::HashSet::new();
        set.insert(err);
        assert!(set.contains(&copy));
    }

    #[test]
    fn type_error_debug() {
        assert!(format!("{:?}", TypeError::NonPositiveRange).contains("NonPositiveRange"));
    }

    // ─── CurveError ──────────────────────────────────────────────────────

    #[test]
    fn curve_error_display_all_variants() {
        assert!(format!("{}", CurveError::TooFewNodes { found: 1 }).contains("two nodes"));
        assert!(format!("{}", CurveError::NodesNotIncreasing { at_index: 4 }).contains('4'));
        assert!(
            format!(
                "{}",
                CurveError::NonPositiveDiscount {
                    at_index: 2,
                    value: -0.5,
                }
            )
            .contains("-0.5")
        );
        assert!(format!("{}", CurveError::AnchorNotUnit).contains("anchor"));
        assert!(format!("{}", CurveError::InvalidTime { t: -1.0 }).contains("-1"));
        assert!(format!("{}", CurveError::DuplicateNode { t: 0.5 }).contains("0.5"));
        assert!(
            format!("{}", CurveError::Type(TypeError::NonPositiveRange)).contains("type error")
        );
    }

    #[test]
    fn curve_error_from_type_and_source() {
        let te = TypeError::NonPositiveRange;
        let ce: CurveError = te.into();
        assert!(matches!(ce, CurveError::Type(_)));
        let dyn_err: &dyn std::error::Error = &ce;
        assert!(dyn_err.source().is_some());
    }

    #[test]
    fn curve_error_no_source_for_plain_variants() {
        let ce = CurveError::AnchorNotUnit;
        let dyn_err: &dyn std::error::Error = &ce;
        assert!(dyn_err.source().is_none());
    }

    #[test]
    fn curve_error_copy_eq() {
        let err = CurveError::TooFewNodes { found: 0 };
        let copy = err;
        assert_eq!(err, copy);
    }

    #[test]
    fn curve_error_debug() {
        assert!(format!("{:?}", CurveError::AnchorNotUnit).contains("AnchorNotUnit"));
    }

    // ─── BootstrapError ──────────────────────────────────────────────────

    #[test]
    fn bootstrap_error_display_all_variants() {
        assert!(format!("{}", BootstrapError::InstrumentsNotOrdered { at_index: 2 }).contains('2'));
        assert!(format!("{}", BootstrapError::NonIncreasingAnchor { at_index: 5 }).contains('5'));
        let m = format!(
            "{}",
            BootstrapError::LegDidNotConverge {
                at_index: 3,
                residual: 1.2e-9,
            }
        );
        assert!(m.contains('3'));
        assert!(m.contains("converge"));
        assert!(format!("{}", BootstrapError::NoBracket { at_index: 7 }).contains('7'));
        assert!(
            format!(
                "{}",
                BootstrapError::InvalidInstrument {
                    at_index: 1,
                    reason: "negative rate",
                }
            )
            .contains("negative rate")
        );
        assert!(
            format!("{}", BootstrapError::Curve(CurveError::AnchorNotUnit)).contains("curve error")
        );
        assert!(
            format!("{}", BootstrapError::Type(TypeError::NonPositiveRange)).contains("type error")
        );
    }

    #[test]
    fn bootstrap_error_from_type() {
        let te = TypeError::NonPositiveRange;
        let be: BootstrapError = te.into();
        assert!(matches!(be, BootstrapError::Type(_)));
        let dyn_err: &dyn std::error::Error = &be;
        assert!(dyn_err.source().is_some());
    }

    #[test]
    fn bootstrap_error_from_curve() {
        let ce = CurveError::AnchorNotUnit;
        let be: BootstrapError = ce.into();
        assert!(matches!(be, BootstrapError::Curve(_)));
        let dyn_err: &dyn std::error::Error = &be;
        assert!(dyn_err.source().is_some());
    }

    #[test]
    fn bootstrap_error_no_source_for_plain_variants() {
        let be = BootstrapError::NoBracket { at_index: 0 };
        let dyn_err: &dyn std::error::Error = &be;
        assert!(dyn_err.source().is_none());
    }

    #[test]
    fn bootstrap_error_copy_eq() {
        let err = BootstrapError::InstrumentsNotOrdered { at_index: 0 };
        let copy = err;
        assert_eq!(err, copy);
    }

    #[test]
    fn bootstrap_error_debug() {
        assert!(format!("{:?}", BootstrapError::NoBracket { at_index: 0 }).contains("NoBracket"));
    }

    #[test]
    fn bootstrap_error_chained_from_type_through_curve_is_not_automatic() {
        // Validate that we use the direct `From<TypeError>` rather than going
        // via `CurveError`, so the boxed source is `TypeError` (not wrapped).
        let te = TypeError::NonPositiveRange;
        let be: BootstrapError = te.into();
        assert!(matches!(be, BootstrapError::Type(_)));
    }
}
