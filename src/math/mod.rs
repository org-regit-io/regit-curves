// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Hand-rolled numerical primitives — no external math dependencies.
//!
//! The crate is zero-dependency, so every solver and root-finder is
//! implemented from its primary source. All routines are pure functions,
//! deterministic (same input produces bit-identical output), and `std`-only.
//!
//! # Contents
//!
//! - [`linear_solve::solve`] — dense Gaussian elimination with partial
//!   pivoting (Golub & Van Loan §3.4).
//! - [`linear_solve::solve_spd`] — symmetric positive-definite solve by
//!   Cholesky decomposition (Golub & Van Loan §4.2).
//! - [`tridiag::thomas`] — `O(n)` tridiagonal solve by the Thomas algorithm.
//! - [`brent::brent_root`] — bracketed root-finder by Brent's method
//!   (Brent 1973).
//!
//! Errors from these primitives surface through the local [`MathError`] enum;
//! `From<MathError> for CurveError` is provided so curve-level callers can
//! propagate with `?`.
//!
//! # References
//!
//! - Golub, G. H. & Van Loan, C. F., *Matrix Computations*, 4th ed.,
//!   Johns Hopkins (2013), Chapters 3 and 4.
//! - Brent, R. P., *Algorithms for Minimization Without Derivatives*,
//!   Prentice-Hall (1973), Chapter 4.
//! - Thomas, L. H., *Elliptic Problems in Linear Difference Equations over a
//!   Network*, Watson Sci. Comput. Lab. Report (Columbia 1949).

use core::fmt;

use crate::errors::CurveError;

pub mod brent;
pub mod linear_solve;
pub mod tridiag;

pub use brent::{BrentConfig, brent_root};
pub use linear_solve::{solve, solve_spd};
pub use tridiag::thomas;

/// Errors raised by the numerical primitives in [`math`](self).
///
/// These are domain errors — every routine is a pure function and never
/// panics on the inputs it accepts; "domain" means the input violates the
/// algorithm's preconditions (singular matrix, non-SPD matrix, bracket fails
/// to straddle a root, etc.).
///
/// # Examples
///
/// ```
/// use regit_curves::math::MathError;
///
/// let err = MathError::Singular;
/// assert_eq!(format!("{err}"), "matrix is singular");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MathError {
    /// A pivot collapsed below tolerance during elimination — the matrix is
    /// singular (up to round-off).
    Singular,
    /// Cholesky decomposition encountered a non-positive pivot — the matrix
    /// is not symmetric positive-definite.
    NotSpd,
    /// Inputs to a solver have inconsistent dimensions.
    DimensionMismatch,
    /// Iterative algorithm did not converge to the requested tolerance
    /// within the iteration cap.
    NoConvergence,
    /// `f(a)` and `f(b)` do not straddle zero, so a bracketed root-finder
    /// cannot make progress.
    BracketNotStraddling,
}

impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Singular => write!(f, "matrix is singular"),
            Self::NotSpd => write!(f, "matrix is not symmetric positive-definite"),
            Self::DimensionMismatch => write!(f, "inputs have inconsistent dimensions"),
            Self::NoConvergence => write!(f, "iterative algorithm did not converge"),
            Self::BracketNotStraddling => {
                write!(f, "f(a) and f(b) do not straddle zero")
            }
        }
    }
}

impl std::error::Error for MathError {}

impl From<MathError> for CurveError {
    fn from(e: MathError) -> Self {
        // Map every algorithmic failure to a curve-level error that the
        // outer caller can surface. The closest fit at the curve level is
        // `DuplicateNode { t: NaN }` for `Singular` (a duplicated time
        // typically produces a singular interpolation matrix); but more
        // generally these are programmer-side conditions, so we map to
        // `InvalidTime { t: NaN }` as a catch-all "the math could not be
        // performed" signal.
        match e {
            MathError::DimensionMismatch => Self::TooFewNodes { found: 0 },
            _ => Self::InvalidTime { t: f64::NAN },
        }
    }
}

/// Converts a `usize` index or count to `f64` losslessly.
///
/// Splits the value into a high and low `u32` half and recombines through
/// `f64::from`, both of which are exact conversions. The result is exact for
/// every `usize` below `2^53` (the `f64` mantissa width) — i.e. for every
/// grid index and count this crate produces — and avoids the precision-loss
/// `as`-cast lint entirely.
///
/// # Examples
///
/// ```
/// use regit_curves::math::index_to_f64;
///
/// assert_eq!(index_to_f64(0), 0.0);
/// assert_eq!(index_to_f64(42), 42.0);
/// assert_eq!(index_to_f64(1 << 30), (1u64 << 30) as f64);
/// ```
#[inline]
#[must_use]
pub fn index_to_f64(i: usize) -> f64 {
    let value = i as u64;
    let high = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(value & 0xFFFF_FFFF).unwrap_or(u32::MAX);
    f64::from(high) * 4_294_967_296.0 + f64::from(low)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn math_error_display_all_variants() {
        assert_eq!(format!("{}", MathError::Singular), "matrix is singular");
        assert!(format!("{}", MathError::NotSpd).contains("positive-definite"));
        assert!(format!("{}", MathError::DimensionMismatch).contains("dimensions"));
        assert!(format!("{}", MathError::NoConvergence).contains("converge"));
        assert!(format!("{}", MathError::BracketNotStraddling).contains("straddle"));
    }

    #[test]
    fn math_error_is_error_trait() {
        let err: &dyn std::error::Error = &MathError::Singular;
        assert!(err.source().is_none());
    }

    #[test]
    fn math_error_copy_eq_hash() {
        let err = MathError::NotSpd;
        let copy = err;
        assert_eq!(err, copy);
        let mut set = std::collections::HashSet::new();
        set.insert(err);
        assert!(set.contains(&copy));
    }

    #[test]
    fn math_error_debug() {
        assert!(format!("{:?}", MathError::Singular).contains("Singular"));
    }

    #[test]
    fn curve_error_from_math_error_singular() {
        let ce: CurveError = MathError::Singular.into();
        assert!(matches!(ce, CurveError::InvalidTime { .. }));
    }

    #[test]
    fn curve_error_from_math_error_dim_mismatch() {
        let ce: CurveError = MathError::DimensionMismatch.into();
        assert!(matches!(ce, CurveError::TooFewNodes { .. }));
    }

    #[test]
    fn index_to_f64_basic() {
        assert!((index_to_f64(0) - 0.0).abs() < f64::EPSILON);
        assert!((index_to_f64(1) - 1.0).abs() < f64::EPSILON);
        assert!((index_to_f64(1000) - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn index_to_f64_large() {
        let large = 1usize << 40;
        // 2^40 is exactly representable in `f64`, expressed here without an
        // `as` cast (which clippy `cast_precision_loss` flags).
        let expected = f64::from(1u32 << 30) * 1024.0;
        assert!((index_to_f64(large) - expected).abs() < f64::EPSILON);
    }
}
