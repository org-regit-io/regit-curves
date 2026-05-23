// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Audit-grade interest-rate yield curve bootstrap and interpolation in pure
//! Rust.
//!
//! `regit-curves` bootstraps interest-rate yield curves from market
//! instruments (deposits, FRAs, futures, vanilla and OIS swaps, basis swaps),
//! interpolates between curve nodes with a documented family of methods, and
//! exposes the resulting curve as discount factor, zero rate, instantaneous
//! forward and par yield views — single-currency, single-curve and
//! multi-curve (post-2008 OIS-discounted).
//!
//! Designed for auditability: every formula is hand-rolled from primary paper
//! and standards sources with no external dependencies. A regulator, quant
//! auditor, or new engineer can trace every number to a citable derivation in
//! [`MATH.md`].
//!
//! [`MATH.md`]: https://github.com/org-regit-io/regit-curves/blob/main/MATH.md
//!
//! Part of [Regit OS](https://www.regit.io) — the operating system for
//! investment products. From Luxembourg.

#![forbid(unsafe_code)]

pub mod bootstrap;
pub mod curves;
pub mod errors;
pub mod instruments;
pub mod interpolation;
pub mod math;
pub mod multi_curve;
pub mod types;

// Re-exports (top-level ergonomic access)
pub use bootstrap::{Bootstrap, BootstrapConfig};
pub use curves::{DiscountCurve, ForwardCurve, ParCurve, ZeroCurve};
pub use errors::{BootstrapError, CurveError, TypeError};
pub use instruments::{
    BasisLeg, BasisSwap, Bond, Deposit, Fra, Future, Instrument, OisSwap, SwapFixedFloat,
    SwapSchedule,
};
pub use interpolation::{
    ConvexMonotone, CubicSpline, HermiteBessel, Interpolation, InterpolationImpl, Interpolator,
    Linear, LinearInZero, LogLinear, MonotoneCubic, MonotoneHyman, MonotoneSteffen,
    PiecewiseConstantForward, SplineBoundary,
};
pub use multi_curve::{MultiCurve, MultiCurveBootstrap};
pub use types::{BusinessDayConvention, Compounding, Date, Daycount, Frequency, Tenor, TenorUnit};
