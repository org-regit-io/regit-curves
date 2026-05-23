// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Yield-curve types — discount, zero, forward, par.
//!
//! A yield curve has four equivalent representations. They are connected by
//! the canonical identities (see Hagan & West 2006, §2):
//!
//! ```text
//! D(t)  =  exp(-z(t) * t)              (continuous compounding)
//! z(t)  = -ln D(t) / t                 (t > 0)
//! f(t)  = -d/dt ln D(t)                (instantaneous forward)
//! D(t)  =  exp(-INTEGRAL_0^t f(s) ds)
//! ```
//!
//! Any one of the four views fully determines the other three. For an
//! audit-grade implementation the natural design choice is to make **one** of
//! them **canonical** — to store the curve in that representation, and derive
//! the others on demand. We pick `D(t)` (the discount factor) as canonical
//! for three reasons:
//!
//! 1. `D(t)` is the only one whose value is **directly observable** in the
//!    market (zero-coupon bond price) without an integral or a derivative.
//! 2. `D(t)` is the only one whose anchor `D(0) = 1` is a trivial,
//!    convention-free identity. The zero rate `z(0)` is a `0/0` limit; the
//!    forward `f(0)` needs a one-sided derivative.
//! 3. All bootstrap instruments (deposits, FRAs, swaps) price as discount-
//!    factor products. The bootstrap engine solves for `D` at each pillar.
//!
//! Hagan & West (2006, §2) make the same choice explicitly: "the discount
//! function `D(t)` is the centrally interpolated object".
//!
//! # The view types
//!
//! [`DiscountCurve`] is the canonical store: it owns the knot times and
//! discount factors plus the chosen interpolant. The three sibling view
//! types — [`ZeroCurve`], [`ForwardCurve`], [`ParCurve`] — are lightweight
//! borrowers that reinterpret the same data:
//!
//! - [`ZeroCurve`] returns `z(t)` under a chosen [`crate::types::Compounding`].
//! - [`ForwardCurve`] returns simply-compounded forward rates over arbitrary
//!   `[t_1, t_2]` intervals and the instantaneous forward `f(t)`.
//! - [`ParCurve`] returns par-swap rates against the discount curve under a
//!   chosen payment [`crate::types::Frequency`] and accrual
//!   [`crate::types::Daycount`].
//!
//! Each view holds an immutable borrow `&'a DiscountCurve`. That keeps the
//! types tiny and avoids cloning the curve nodes; the cost is that a view
//! cannot outlive its parent curve. (An owned alternative may be added in a
//! future release.)
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §2 ("Defining the
//!   problem"). Identifies the discount function as the centrally interpolated
//!   object; gives the canonical identities between `D`, `z`, and `f`.
//! - Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling*, Vol. 1,
//!   Atlantic Financial Press (2010), §6 ("Curve building"). Single-curve
//!   par-swap-rate formula used by [`ParCurve`].

pub mod discount;
pub mod forward;
pub mod par;
pub mod zero;

pub use discount::DiscountCurve;
pub use forward::ForwardCurve;
pub use par::ParCurve;
pub use zero::ZeroCurve;
