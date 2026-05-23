// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Curve interpolation methods.
//!
//! Yield-curve construction needs a way to **interpolate between knots**. The
//! choice of interpolant determines whether the curve's discount factor `D`,
//! continuously-compounded zero rate `z`, or instantaneous forward `f` is the
//! "smooth" quantity — and therefore the qualitative behaviour of the curve
//! between pillars. The classical reference enumerates nine methods and
//! discusses their trade-offs (Hagan & West 2006, §3).
//!
//! Every interpolator in this crate is a knot-based interpolant
//! `(t_i, y_i) -> y(t)` with the same uniform interface:
//!
//! ```text
//! trait Interpolator {
//!     fn build(&[(t, y)]) -> Result<Self, CurveError>;
//!     fn eval(t)           -> f64;
//!     fn deriv(t)          -> Option<f64>;
//! }
//! ```
//!
//! The `t`-axis units are year fractions from the curve's reference date,
//! computed under the curve's own day-count convention. The `y`-axis unit
//! depends on the consumer (`y = D(t)` for a discount curve;
//! `y = z(t) * t` for a "linear in capitalisation factor" curve, etc.).
//!
//! # Variants
//!
//! All Hagan–West Methods 0-6 plus the monotone refinements (Fritsch–Carlson
//! 1980, Steffen 1990, Hyman 1983) are exposed via the [`Interpolation`] enum.
//! The enum is `#[non_exhaustive]` so adding variants is non-breaking.
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §3.

use crate::errors::CurveError;

pub mod convex_monotone;
pub mod cubic_spline;
pub mod hermite_bessel;
pub mod linear;
pub mod linear_in_zero;
pub mod log_linear;
pub mod monotone_cubic;
pub mod monotone_hyman;
pub mod monotone_steffen;
pub mod piecewise_constant_forward;

pub use convex_monotone::ConvexMonotone;
pub use cubic_spline::{CubicSpline, SplineBoundary};
pub use hermite_bessel::HermiteBessel;
pub use linear::Linear;
pub use linear_in_zero::LinearInZero;
pub use log_linear::LogLinear;
pub use monotone_cubic::MonotoneCubic;
pub use monotone_hyman::MonotoneHyman;
pub use monotone_steffen::MonotoneSteffen;
pub use piecewise_constant_forward::PiecewiseConstantForward;

/// Uniform interface implemented by every interpolation method.
///
/// Implementors store the knot data needed to evaluate the interpolant; once
/// constructed, [`Interpolator::eval`] is a pure function of `t`.
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::{Interpolator, LogLinear};
///
/// let interp = LogLinear::build(&[(0.0, 1.0), (1.0, 0.95)]).unwrap();
/// assert!((interp.eval(0.0) - 1.0).abs() < 1e-15);
/// ```
pub trait Interpolator {
    /// Builds an interpolant over the given knots.
    ///
    /// Knots are `(t, y)` pairs; `t` must be strictly increasing.
    /// Method-specific invariants (`y > 0` for log-linear, monotone `y` for
    /// monotone variants, etc.) are documented on each implementor.
    ///
    /// # Errors
    ///
    /// Returns a [`CurveError`] if the knots violate any invariant required
    /// by the implementor. See per-implementor docs for the variant list.
    fn build(knots: &[(f64, f64)]) -> Result<Self, CurveError>
    where
        Self: Sized;

    /// Evaluates the interpolant at `t`.
    ///
    /// Out-of-range `t` is handled by flat extrapolation in the value domain
    /// (the curve view layer translates that into the correct extrapolation
    /// for `D`, `z`, or `f`).
    fn eval(&self, t: f64) -> f64;

    /// First derivative `dy / dt`, where defined.
    ///
    /// Returns `None` for methods that are not C¹ at the requested point
    /// (e.g. piecewise-constant forward at a knot). The default implementation
    /// returns `None`; implementors override when their interpolant carries a
    /// well-defined derivative.
    fn deriv(&self, _t: f64) -> Option<f64> {
        None
    }
}

/// User-facing choice of interpolation method.
///
/// Marked `#[non_exhaustive]` so additional variants can be added without
/// breaking semver.
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::Interpolation;
///
/// let method = Interpolation::LogLinear;
/// assert_eq!(method, Interpolation::LogLinear);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Interpolation {
    /// Piecewise linear (see [`Linear`]). Hagan & West Method 0.
    Linear,
    /// Piecewise log-linear on `D` (see [`LogLinear`]). Hagan & West Method 1.
    LogLinear,
    /// Linear on the continuously-compounded zero rate (see [`LinearInZero`]).
    /// Hagan & West Method 2 — their recommended default.
    LinearInZero,
    /// Piecewise constant instantaneous forward (see
    /// [`PiecewiseConstantForward`]). Mathematically equivalent to log-linear
    /// on `D`; semantically reports the forward rather than the discount.
    PiecewiseConstantForward,
    /// C² cubic spline with a choice of boundary condition (see
    /// [`CubicSpline`]). Not-a-knot is the recommended boundary.
    CubicSpline(SplineBoundary),
    /// Hagan–West "monotone convex" interpolator (see [`ConvexMonotone`]).
    /// Hagan & West (2008) Method 7 — local, arbitrage-free, and
    /// non-negative-forward-preserving.
    ConvexMonotone,
    /// Bessel-slope Hermite cubic (see [`HermiteBessel`]). C¹ but not
    /// monotonicity-preserving.
    HermiteBessel,
    /// Fritsch–Carlson 1980 monotone piecewise cubic Hermite (see
    /// [`MonotoneCubic`]).
    MonotoneCubic,
    /// Hyman 1983 monotonicity-preserving filter applied to a cubic-spline
    /// base (see [`MonotoneHyman`]).
    MonotoneHyman,
    /// Steffen 1990 monotone cubic Hermite (see [`MonotoneSteffen`]).
    MonotoneSteffen,
}

impl Interpolation {
    /// Builds a concrete [`InterpolationImpl`] instance from this method choice
    /// and a knot list.
    ///
    /// Each variant dispatches to its underlying interpolator's `new` /
    /// `build` constructor — the only place in the crate where the choice of
    /// method is resolved into a concrete type. The discount-curve layer
    /// uses this to store a single polymorphic interpolant alongside the
    /// curve nodes.
    ///
    /// # Errors
    ///
    /// Returns whatever [`CurveError`] variant the underlying interpolator's
    /// constructor returns — typically [`CurveError::TooFewNodes`],
    /// [`CurveError::NodesNotIncreasing`], [`CurveError::DuplicateNode`],
    /// [`CurveError::NonPositiveDiscount`], or [`CurveError::InvalidTime`].
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::interpolation::Interpolation;
    ///
    /// let knots = [(0.0_f64, 1.0_f64), (1.0, 0.95), (2.0, 0.90)];
    /// let interp = Interpolation::LogLinear.build(&knots).unwrap();
    /// assert!((interp.eval(0.0) - 1.0).abs() < 1e-15);
    /// assert!((interp.eval(1.0) - 0.95).abs() < 1e-15);
    /// ```
    pub fn build(self, knots: &[(f64, f64)]) -> Result<InterpolationImpl, CurveError> {
        match self {
            Self::Linear => Linear::build(knots).map(InterpolationImpl::Linear),
            Self::LogLinear => LogLinear::build(knots).map(InterpolationImpl::LogLinear),
            Self::LinearInZero => LinearInZero::build(knots).map(InterpolationImpl::LinearInZero),
            Self::PiecewiseConstantForward => PiecewiseConstantForward::build(knots)
                .map(InterpolationImpl::PiecewiseConstantForward),
            Self::CubicSpline(boundary) => {
                CubicSpline::new(knots, boundary).map(InterpolationImpl::CubicSpline)
            }
            Self::ConvexMonotone => {
                ConvexMonotone::build(knots).map(InterpolationImpl::ConvexMonotone)
            }
            Self::HermiteBessel => {
                HermiteBessel::build(knots).map(InterpolationImpl::HermiteBessel)
            }
            Self::MonotoneCubic => {
                MonotoneCubic::build(knots).map(InterpolationImpl::MonotoneCubic)
            }
            Self::MonotoneHyman => {
                MonotoneHyman::build(knots).map(InterpolationImpl::MonotoneHyman)
            }
            Self::MonotoneSteffen => {
                MonotoneSteffen::build(knots).map(InterpolationImpl::MonotoneSteffen)
            }
        }
    }
}

/// A concrete interpolator instance — one of the nine implementations,
/// stored polymorphically.
///
/// Built from an [`Interpolation`] method choice and a knot list via
/// [`Interpolation::build`]. The discount-curve layer stores a single
/// `InterpolationImpl` alongside the curve nodes so that subsequent
/// `discount(t)` queries dispatch through a fixed concrete type without
/// rebuilding the interpolant per call.
///
/// # Examples
///
/// ```
/// use regit_curves::interpolation::{Interpolation, InterpolationImpl};
///
/// let knots = [(0.0_f64, 1.0_f64), (1.0, 0.95)];
/// let interp: InterpolationImpl = Interpolation::LogLinear.build(&knots).unwrap();
/// assert!((interp.eval(0.5) - 0.95_f64.sqrt()).abs() < 1e-15);
/// ```
#[derive(Debug, Clone)]
pub enum InterpolationImpl {
    /// Piecewise linear (see [`Linear`]).
    Linear(Linear),
    /// Piecewise log-linear (see [`LogLinear`]).
    LogLinear(LogLinear),
    /// Linear on the continuously-compounded zero rate (see [`LinearInZero`]).
    LinearInZero(LinearInZero),
    /// Piecewise constant instantaneous forward (see
    /// [`PiecewiseConstantForward`]).
    PiecewiseConstantForward(PiecewiseConstantForward),
    /// C² cubic spline (see [`CubicSpline`]).
    CubicSpline(CubicSpline),
    /// Hagan–West monotone-convex interpolator (see [`ConvexMonotone`]).
    ConvexMonotone(ConvexMonotone),
    /// Bessel-slope Hermite cubic (see [`HermiteBessel`]).
    HermiteBessel(HermiteBessel),
    /// Fritsch–Carlson 1980 monotone cubic (see [`MonotoneCubic`]).
    MonotoneCubic(MonotoneCubic),
    /// Hyman 1983 monotone filter (see [`MonotoneHyman`]).
    MonotoneHyman(MonotoneHyman),
    /// Steffen 1990 monotone cubic (see [`MonotoneSteffen`]).
    MonotoneSteffen(MonotoneSteffen),
}

impl InterpolationImpl {
    /// Evaluates the interpolant at `t`. Dispatches to the contained
    /// concrete interpolator's [`Interpolator::eval`].
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::interpolation::Interpolation;
    ///
    /// let knots = [(0.0_f64, 1.0_f64), (1.0, 0.95)];
    /// let interp = Interpolation::LogLinear.build(&knots).unwrap();
    /// assert!((interp.eval(0.0) - 1.0).abs() < 1e-15);
    /// ```
    #[must_use]
    pub fn eval(&self, t: f64) -> f64 {
        match self {
            Self::Linear(i) => i.eval(t),
            Self::LogLinear(i) => i.eval(t),
            Self::LinearInZero(i) => i.eval(t),
            Self::PiecewiseConstantForward(i) => i.eval(t),
            Self::CubicSpline(i) => i.eval(t),
            Self::ConvexMonotone(i) => i.eval(t),
            Self::HermiteBessel(i) => i.eval(t),
            Self::MonotoneCubic(i) => i.eval(t),
            Self::MonotoneHyman(i) => i.eval(t),
            Self::MonotoneSteffen(i) => i.eval(t),
        }
    }

    /// First derivative `dy / dt` at `t`, where defined. Dispatches to the
    /// contained concrete interpolator's [`Interpolator::deriv`].
    ///
    /// Returns `None` for methods that are not C¹ at the requested point
    /// (e.g. [`PiecewiseConstantForward`] at a knot).
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::interpolation::Interpolation;
    ///
    /// let knots = [(0.0_f64, 1.0_f64), (1.0, 0.95), (2.0, 0.90)];
    /// let interp = Interpolation::LogLinear.build(&knots).unwrap();
    /// // LogLinear has a well-defined right-derivative everywhere.
    /// assert!(interp.deriv(0.5).is_some());
    /// ```
    #[must_use]
    pub fn deriv(&self, t: f64) -> Option<f64> {
        match self {
            Self::Linear(i) => i.deriv(t),
            Self::LogLinear(i) => i.deriv(t),
            Self::LinearInZero(i) => i.deriv(t),
            Self::PiecewiseConstantForward(i) => i.deriv(t),
            Self::CubicSpline(i) => i.deriv(t),
            Self::ConvexMonotone(i) => i.deriv(t),
            Self::HermiteBessel(i) => i.deriv(t),
            Self::MonotoneCubic(i) => i.deriv(t),
            Self::MonotoneHyman(i) => i.deriv(t),
            Self::MonotoneSteffen(i) => i.deriv(t),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_log_linear_copy_eq() {
        let a = Interpolation::LogLinear;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn interpolation_debug_includes_variant() {
        let s = format!("{:?}", Interpolation::LogLinear);
        assert!(s.contains("LogLinear"));
    }

    #[test]
    fn trait_build_via_log_linear() {
        let interp = LogLinear::build(&[(0.0, 1.0), (1.0, 0.95)]).unwrap();
        assert!((interp.eval(0.0) - 1.0).abs() < 1e-15);
        assert!((interp.eval(1.0) - 0.95).abs() < 1e-15);
    }

    /// Mock implementor exercising the default `deriv` impl (`None`).
    struct ConstantOne;

    impl Interpolator for ConstantOne {
        fn build(_knots: &[(f64, f64)]) -> Result<Self, CurveError> {
            Ok(Self)
        }
        fn eval(&self, _t: f64) -> f64 {
            1.0
        }
    }

    #[test]
    fn trait_default_deriv_is_none() {
        let c = ConstantOne::build(&[]).unwrap();
        assert!(c.deriv(0.5).is_none());
        assert!((c.eval(0.5) - 1.0).abs() < 1e-15);
    }

    // ─── Interpolation::build dispatch (one test per variant) ────────────

    /// A representative set of `(t, D)` knots used to exercise every variant.
    /// All discount factors are `> 0` and times strictly increasing — the
    /// common-denominator invariants of every interpolator.
    fn standard_knots() -> [(f64, f64); 4] {
        [(0.0, 1.0), (0.5, 0.975), (1.0, 0.95), (2.0, 0.90)]
    }

    /// Asserts that every knot in `knots` is reproduced exactly (to 1e-12)
    /// by `interp.eval`. This is the canonical correctness check shared by
    /// every interpolator.
    fn assert_knots_reproduced(interp: &InterpolationImpl, knots: &[(f64, f64)]) {
        for &(t, y) in knots {
            let v = interp.eval(t);
            assert!(
                (v - y).abs() < 1e-12,
                "knot ({t}, {y}) not reproduced: got {v}"
            );
        }
    }

    #[test]
    fn interpolation_build_linear() {
        let knots = standard_knots();
        let interp = Interpolation::Linear.build(&knots).unwrap();
        assert!(matches!(interp, InterpolationImpl::Linear(_)));
        assert_knots_reproduced(&interp, &knots);
    }

    #[test]
    fn interpolation_build_log_linear() {
        let knots = standard_knots();
        let interp = Interpolation::LogLinear.build(&knots).unwrap();
        assert!(matches!(interp, InterpolationImpl::LogLinear(_)));
        assert_knots_reproduced(&interp, &knots);
    }

    #[test]
    fn interpolation_build_linear_in_zero() {
        let knots = standard_knots();
        let interp = Interpolation::LinearInZero.build(&knots).unwrap();
        assert!(matches!(interp, InterpolationImpl::LinearInZero(_)));
        assert_knots_reproduced(&interp, &knots);
    }

    #[test]
    fn interpolation_build_piecewise_constant_forward() {
        let knots = standard_knots();
        let interp = Interpolation::PiecewiseConstantForward
            .build(&knots)
            .unwrap();
        assert!(matches!(
            interp,
            InterpolationImpl::PiecewiseConstantForward(_)
        ));
        assert_knots_reproduced(&interp, &knots);
    }

    #[test]
    fn interpolation_build_cubic_spline_natural() {
        let knots = standard_knots();
        let interp = Interpolation::CubicSpline(SplineBoundary::Natural)
            .build(&knots)
            .unwrap();
        assert!(matches!(interp, InterpolationImpl::CubicSpline(_)));
        assert_knots_reproduced(&interp, &knots);
    }

    #[test]
    fn interpolation_build_cubic_spline_not_a_knot() {
        let knots = standard_knots();
        let interp = Interpolation::CubicSpline(SplineBoundary::NotAKnot)
            .build(&knots)
            .unwrap();
        assert!(matches!(interp, InterpolationImpl::CubicSpline(_)));
        assert_knots_reproduced(&interp, &knots);
    }

    #[test]
    fn interpolation_build_convex_monotone() {
        let knots = standard_knots();
        let interp = Interpolation::ConvexMonotone.build(&knots).unwrap();
        assert!(matches!(interp, InterpolationImpl::ConvexMonotone(_)));
        assert_knots_reproduced(&interp, &knots);
    }

    #[test]
    fn interpolation_build_hermite_bessel() {
        let knots = standard_knots();
        let interp = Interpolation::HermiteBessel.build(&knots).unwrap();
        assert!(matches!(interp, InterpolationImpl::HermiteBessel(_)));
        assert_knots_reproduced(&interp, &knots);
    }

    #[test]
    fn interpolation_build_monotone_cubic() {
        let knots = standard_knots();
        let interp = Interpolation::MonotoneCubic.build(&knots).unwrap();
        assert!(matches!(interp, InterpolationImpl::MonotoneCubic(_)));
        assert_knots_reproduced(&interp, &knots);
    }

    #[test]
    fn interpolation_build_monotone_hyman() {
        let knots = standard_knots();
        let interp = Interpolation::MonotoneHyman.build(&knots).unwrap();
        assert!(matches!(interp, InterpolationImpl::MonotoneHyman(_)));
        assert_knots_reproduced(&interp, &knots);
    }

    #[test]
    fn interpolation_build_monotone_steffen() {
        let knots = standard_knots();
        let interp = Interpolation::MonotoneSteffen.build(&knots).unwrap();
        assert!(matches!(interp, InterpolationImpl::MonotoneSteffen(_)));
        assert_knots_reproduced(&interp, &knots);
    }

    #[test]
    fn interpolation_build_propagates_too_few_nodes() {
        let err = Interpolation::LogLinear.build(&[(0.0, 1.0)]).unwrap_err();
        assert!(matches!(err, CurveError::TooFewNodes { found: 1 }));
    }

    #[test]
    fn interpolation_impl_deriv_dispatches() {
        let knots = standard_knots();
        let interp = Interpolation::LogLinear.build(&knots).unwrap();
        // LogLinear's deriv is well-defined at an interior non-knot.
        let d = interp.deriv(0.75).unwrap();
        assert!(d.is_finite());
    }

    #[test]
    fn interpolation_impl_clone_yields_equivalent_eval() {
        let knots = standard_knots();
        let interp = Interpolation::LogLinear.build(&knots).unwrap();
        let copy = interp.clone();
        assert!((interp.eval(0.75) - copy.eval(0.75)).abs() < 1e-15);
    }
}
