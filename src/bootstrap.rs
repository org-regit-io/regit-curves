// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Sequential iterative bootstrap engine.
//!
//! The bootstrap engine constructs a [`DiscountCurve`] from a list of market
//! instruments (deposits, FRAs, futures, fixed-float swaps, OIS swaps, basis
//! swaps) by **sequentially solving for the discount factor at each
//! instrument's pillar date**, so that the instrument re-prices to within
//! `tolerance` against the in-progress curve.
//!
//! ```text
//! anchor:    (t_0, D_0) = (0, 1)
//! for k = 1, 2, ..., N:
//!     t_k    = daycount.year_fraction(reference_date, instruments[k-1].pillar())
//!     residual_fn(D) builds an interim CurveSnapshot over (t_0..t_{k-1}, t_k)
//!         with (D_0..D_{k-1}, D) and asks instruments[k-1] for its residual
//!     solve residual_fn(D_k) = 0 by Brent's method, bracketing around the
//!         previous-anchor forward extrapolation
//!     append (t_k, D_k) to the running curve
//! return DiscountCurve::from_times_and_discounts(reference_date, daycount,
//!                                                times, discounts, method)
//! ```
//!
//! For interpolation methods whose value at one pillar depends on the value
//! at later pillars (cubic spline, Hermite-Bessel, Hyman-filtered cubic), the
//! single-pass sweep is not sufficient — the global interpolant shifts as
//! later pillars are added. The engine handles this with an **outer
//! iteration**: starting from the single-pass solution it re-solves each
//! pillar against the latest curve until the maximum nodal change drops below
//! `iter_tol`. For local interpolants (linear, log-linear, linear-in-zero,
//! piecewise-constant forward, the monotone Hermite cubics) one pass
//! suffices.
//!
//! Convergence of the outer iteration on consistent market data follows from
//! the contractivity of the residual map under reasonable curve shapes; the
//! formal argument is laid out in Andersen & Piterbarg (2010, Vol. 1 §6.4)
//! and Hagan & West (2006, §3).
//!
//! # Re-pricing certificate
//!
//! On every successful `build`, every input instrument's residual against the
//! returned curve is `< config.tolerance`. This is the bootstrap's contract
//! with the caller: a curve returned by `Bootstrap::build` is one that
//! re-prices its inputs.
//!
//! # References
//!
//! - Hagan, P. S. & West, G., "Interpolation methods for curve construction",
//!   *Applied Mathematical Finance* 13(2):89-129 (2006), §3. The canonical
//!   "sequential anchor-by-anchor solve" formulation.
//! - Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling*,
//!   Volume I: Foundations and Vanilla Models, Atlantic Financial Press
//!   (2010), §6.4. Outer-iteration treatment for non-local interpolators in
//!   the single-currency bootstrap.

use crate::curves::DiscountCurve;
use crate::errors::BootstrapError;
use crate::instruments::{CurveSnapshot, Instrument, InstrumentLike};
use crate::interpolation::Interpolation;
use crate::math::MathError;
use crate::math::brent::{BrentConfig, brent_root};
use crate::types::{Date, Daycount};

/// Configuration for [`Bootstrap`].
///
/// The defaults match the values documented in `WORKING.md` §3.10:
/// `tolerance = 1e-12`, `max_iter = 100`, `bracket = 0.5`, `iterative = true`,
/// `iter_max = 8`, `iter_tol = 1e-14`.
///
/// # Examples
///
/// ```
/// use regit_curves::bootstrap::BootstrapConfig;
///
/// let cfg = BootstrapConfig::default();
/// assert!((cfg.tolerance - 1e-12).abs() < 1e-18);
/// assert_eq!(cfg.max_iter, 100);
/// assert!((cfg.bracket - 0.5).abs() < 1e-15);
/// assert!(cfg.iterative);
/// assert_eq!(cfg.iter_max, 8);
/// assert!((cfg.iter_tol - 1e-14).abs() < 1e-20);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapConfig {
    /// Per-leg residual tolerance handed to Brent as `ftol`.
    pub tolerance: f64,
    /// Per-leg iteration cap handed to Brent as `max_iter`.
    pub max_iter: u32,
    /// Initial half-width of the discount-factor bracket. The bracket is
    /// `[D_guess * exp(-bracket), D_guess * exp(+bracket)]` and is widened on
    /// failure by doubling up to five times.
    pub bracket: f64,
    /// Run the outer iteration for non-local interpolation methods. Has no
    /// effect for local methods (linear, log-linear, linear-in-zero,
    /// piecewise-constant forward, monotone cubic, Steffen).
    pub iterative: bool,
    /// Outer-iteration cap.
    pub iter_max: u32,
    /// Outer-iteration convergence tolerance on the maximum nodal change
    /// `max_i |D_i^{new} - D_i^{old}|`.
    pub iter_tol: f64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_iter: 100,
            bracket: 0.5,
            iterative: true,
            iter_max: 8,
            iter_tol: 1e-14,
        }
    }
}

/// Sequential iterative bootstrap engine.
///
/// Constructs a [`DiscountCurve`] from a list of [`Instrument`] quotes by
/// driving each instrument's residual to zero against the in-progress curve.
/// See the module-level documentation for the algorithm.
///
/// # Examples
///
/// ```
/// use regit_curves::bootstrap::{Bootstrap, BootstrapConfig};
/// use regit_curves::instruments::{Deposit, Instrument};
/// use regit_curves::interpolation::Interpolation;
/// use regit_curves::types::{Date, Daycount};
///
/// let reference = Date::from_ymd(2024, 1, 2).unwrap();
/// let dep1 = Deposit::new(
///     reference,
///     Date::from_ymd(2024, 4, 2).unwrap(),
///     0.05,
///     Daycount::Act360,
/// )
/// .unwrap();
/// let dep2 = Deposit::new(
///     reference,
///     Date::from_ymd(2024, 7, 2).unwrap(),
///     0.05,
///     Daycount::Act360,
/// )
/// .unwrap();
/// let instruments = [Instrument::Deposit(dep1), Instrument::Deposit(dep2)];
///
/// let bootstrap = Bootstrap::new(reference, Daycount::Act360);
/// let curve = bootstrap
///     .build(&instruments, Interpolation::LogLinear)
///     .unwrap();
/// assert_eq!(curve.reference_date(), reference);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bootstrap {
    /// Curve anchor / reference date.
    pub reference_date: Date,
    /// Day-count convention for the curve's `t`-axis.
    pub daycount: Daycount,
    /// Solver configuration.
    pub config: BootstrapConfig,
}

impl Bootstrap {
    /// Constructs a bootstrap engine with [`BootstrapConfig::default`].
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::bootstrap::Bootstrap;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let bs = Bootstrap::new(reference, Daycount::Act360);
    /// assert_eq!(bs.reference_date, reference);
    /// assert_eq!(bs.daycount, Daycount::Act360);
    /// ```
    #[must_use]
    pub fn new(reference_date: Date, daycount: Daycount) -> Self {
        Self {
            reference_date,
            daycount,
            config: BootstrapConfig::default(),
        }
    }

    /// Returns the bootstrap engine with the supplied configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::bootstrap::{Bootstrap, BootstrapConfig};
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let cfg = BootstrapConfig {
    ///     tolerance: 1e-10,
    ///     ..BootstrapConfig::default()
    /// };
    /// let bs = Bootstrap::new(reference, Daycount::Act360).with_config(cfg);
    /// assert!((bs.config.tolerance - 1e-10).abs() < 1e-18);
    /// ```
    #[must_use]
    pub fn with_config(mut self, config: BootstrapConfig) -> Self {
        self.config = config;
        self
    }

    /// Builds a [`DiscountCurve`] that re-prices every instrument in
    /// `instruments` to within [`BootstrapConfig::tolerance`].
    ///
    /// Instruments must be ordered by [`Instrument::pillar`], strictly
    /// increasing. The pillar of every instrument must lie strictly after
    /// [`Bootstrap::reference_date`].
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `instruments` is empty, or if
    ///   any pillar is on or before the reference date.
    /// - [`BootstrapError::NonIncreasingAnchor`] if pillars are not strictly
    ///   increasing.
    /// - [`BootstrapError::NoBracket`] if no sign change can be found in the
    ///   discount-factor search interval for a leg, even after widening.
    /// - [`BootstrapError::LegDidNotConverge`] if Brent fails to converge, or
    ///   if the outer iteration fails to converge within
    ///   [`BootstrapConfig::iter_max`] passes.
    /// - [`BootstrapError::Curve`] if the final curve construction rejects
    ///   the bootstrapped knots (should not happen given the validation
    ///   above).
    /// - [`BootstrapError::Type`] if a day-count query fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::bootstrap::Bootstrap;
    /// use regit_curves::instruments::{Deposit, Instrument};
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let dep = Deposit::new(
    ///     reference,
    ///     Date::from_ymd(2024, 4, 2).unwrap(),
    ///     0.05,
    ///     Daycount::Act360,
    /// )
    /// .unwrap();
    /// let curve = Bootstrap::new(reference, Daycount::Act360)
    ///     .build(&[Instrument::Deposit(dep)], Interpolation::LogLinear)
    ///     .unwrap();
    /// assert!(curve.discounts().len() == 2);
    /// ```
    pub fn build(
        &self,
        instruments: &[Instrument],
        method: Interpolation,
    ) -> Result<DiscountCurve, BootstrapError> {
        // Step 0 — validation.
        if instruments.is_empty() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "no instruments supplied",
            });
        }
        for (i, inst) in instruments.iter().enumerate() {
            let pillar = inst.pillar();
            if pillar.days_between(self.reference_date) >= 0 {
                // pillar <= reference_date
                return Err(BootstrapError::InvalidInstrument {
                    at_index: i,
                    reason: "instrument pillar must be after reference_date",
                });
            }
            if i > 0 {
                let prev_pillar = instruments[i - 1].pillar();
                if pillar.days_between(prev_pillar) >= 0 {
                    // pillar <= prev_pillar
                    return Err(BootstrapError::NonIncreasingAnchor { at_index: i });
                }
            }
        }

        // Step 1 — initial nodes (anchor + scaffold).
        let n = instruments.len();
        let mut times: Vec<f64> = Vec::with_capacity(n + 1);
        let mut discounts: Vec<f64> = Vec::with_capacity(n + 1);
        times.push(0.0);
        discounts.push(1.0);
        for inst in instruments {
            let t = self
                .daycount
                .year_fraction(self.reference_date, inst.pillar())?;
            times.push(t);
            discounts.push(1.0); // placeholder; overwritten in the sweep
        }

        // Step 2 — single-pass sequential bootstrap.
        self.sweep(instruments, method, &times, &mut discounts, true)?;

        // Step 3 — outer iteration for non-local interpolators.
        if self.config.iterative && method_is_nonlocal(method) {
            let mut converged = false;
            let mut last_change = 0.0_f64;
            for _ in 0..self.config.iter_max {
                let previous = discounts.clone();
                self.sweep(instruments, method, &times, &mut discounts, false)?;
                let mut max_change = 0.0_f64;
                for (a, b) in discounts.iter().zip(previous.iter()) {
                    let delta = (a - b).abs();
                    if delta > max_change {
                        max_change = delta;
                    }
                }
                last_change = max_change;
                if max_change < self.config.iter_tol {
                    converged = true;
                    break;
                }
            }
            if !converged {
                return Err(BootstrapError::LegDidNotConverge {
                    at_index: usize::MAX,
                    residual: last_change,
                });
            }
        }

        // Step 4 — final curve.
        let curve = DiscountCurve::from_times_and_discounts(
            self.reference_date,
            self.daycount,
            &times,
            &discounts,
            method,
        )?;
        Ok(curve)
    }

    /// Re-solves the discount factor at each instrument's pillar against the
    /// current `discounts` vector. When `is_initial` is `true`, the previous-
    /// anchor forward extrapolation is used as the initial guess for the
    /// segment; when `false`, the current value at that pillar is used as the
    /// warm start (outer iteration).
    fn sweep(
        &self,
        instruments: &[Instrument],
        _method: Interpolation,
        times: &[f64],
        discounts: &mut [f64],
        is_initial: bool,
    ) -> Result<(), BootstrapError> {
        for (k, inst) in instruments.iter().enumerate() {
            let idx = k + 1; // discounts index (skip anchor)
            let t_k = times[idx];
            let t_prev = times[idx - 1];
            let d_prev = discounts[idx - 1];

            let d_guess = if is_initial {
                let r_prev = if k == 0 {
                    0.05_f64
                } else {
                    // Previous-segment forward rate (continuous).
                    let t_p2 = times[idx - 2];
                    let d_p2 = discounts[idx - 2];
                    // r = ln(d_p2 / d_prev) / (t_prev - t_p2)
                    let dt = t_prev - t_p2;
                    if dt > 0.0 && d_prev > 0.0 && d_p2 > 0.0 {
                        (d_p2 / d_prev).ln() / dt
                    } else {
                        0.05_f64
                    }
                };
                let dt = t_k - t_prev;
                d_prev * (-r_prev * dt).exp()
            } else {
                // Warm start at the existing value, but guard against zero or
                // non-positive values from a corrupted previous pass.
                let warm = discounts[idx];
                if warm.is_finite() && warm > 0.0 {
                    warm
                } else {
                    d_prev
                }
            };

            // Bracket: [d_guess * exp(-bracket), d_guess * exp(+bracket)].
            // Expand by doubling up to 5 times if no sign change found.
            let ctx = LegContext {
                index: k,
                instrument: inst,
                pillar_idx: idx,
                d_guess,
                times,
                discounts,
            };
            let solved = self.solve_leg(&ctx)?;
            discounts[idx] = solved;
        }
        Ok(())
    }

    /// Solves for the discount factor at `ctx.pillar_idx` so that the
    /// instrument's residual is zero against the curve whose `(times,
    /// discounts)` are the supplied slices with the candidate discount factor
    /// substituted at `pillar_idx`.
    fn solve_leg(&self, ctx: &LegContext<'_>) -> Result<f64, BootstrapError> {
        let mut bracket = self.config.bracket;
        // Tracks the most recently seen endpoint residual, so that an
        // unexpected `Err(_)` from Brent has something concrete to surface.
        // The initial assignment is overwritten on the first iteration
        // before any read; the `let mut` plus seed value lets us avoid an
        // `Option<f64>` round-trip.
        #[allow(unused_assignments)]
        let mut last_residual: f64 = 0.0;
        for attempt in 0..=5_u32 {
            let lo = (ctx.d_guess * (-bracket).exp()).max(f64::MIN_POSITIVE);
            let hi = ctx.d_guess * bracket.exp();

            let residual_fn = |d: f64| -> f64 {
                let mut probe = ctx.discounts.to_vec();
                probe[ctx.pillar_idx] = d;
                let snapshot = CurveSnapshot {
                    reference_date: self.reference_date,
                    daycount: self.daycount,
                    times: ctx.times,
                    discounts: &probe,
                };
                // A `Result::Err` from the instrument is reported as a large
                // positive sentinel so Brent does not see a NaN; the surrounding
                // sweep loop catches the real error on the final settled call.
                instrument_residual(ctx.instrument, self.reference_date, &snapshot)
                    .unwrap_or(f64::INFINITY)
            };

            // Probe endpoints once to detect a same-sign bracket cheaply.
            let f_lo = residual_fn(lo);
            let f_hi = residual_fn(hi);
            last_residual = f_lo;
            if !f_lo.is_finite() || !f_hi.is_finite() || f_lo * f_hi > 0.0 {
                if attempt == 5 {
                    return Err(BootstrapError::NoBracket {
                        at_index: ctx.index,
                    });
                }
                bracket *= 2.0;
                continue;
            }

            let brent_cfg = BrentConfig {
                xtol: 1e-15,
                ftol: self.config.tolerance,
                max_iter: self.config.max_iter,
            };
            match brent_root(residual_fn, lo, hi, brent_cfg) {
                Ok(root) => {
                    // Confirm the residual is within tolerance — Brent may
                    // return on xtol convergence without ftol being met for
                    // pathological functions.
                    let mut probe = ctx.discounts.to_vec();
                    probe[ctx.pillar_idx] = root;
                    let snapshot = CurveSnapshot {
                        reference_date: self.reference_date,
                        daycount: self.daycount,
                        times: ctx.times,
                        discounts: &probe,
                    };
                    let final_residual =
                        instrument_residual(ctx.instrument, self.reference_date, &snapshot)?;
                    if final_residual.abs() > self.config.tolerance {
                        return Err(BootstrapError::LegDidNotConverge {
                            at_index: ctx.index,
                            residual: final_residual,
                        });
                    }
                    return Ok(root);
                }
                Err(MathError::BracketNotStraddling) => {
                    if attempt == 5 {
                        return Err(BootstrapError::NoBracket {
                            at_index: ctx.index,
                        });
                    }
                    bracket *= 2.0;
                }
                Err(_) => {
                    return Err(BootstrapError::LegDidNotConverge {
                        at_index: ctx.index,
                        residual: last_residual,
                    });
                }
            }
        }
        Err(BootstrapError::NoBracket {
            at_index: ctx.index,
        })
    }
}

/// Bundled inputs for a single leg's solve. Keeps `solve_leg`'s signature
/// short and clippy-clean.
struct LegContext<'a> {
    /// Instrument index in the input list (used to populate `BootstrapError`).
    index: usize,
    /// The instrument being re-priced.
    instrument: &'a Instrument,
    /// Index of the candidate discount factor in the running `discounts`
    /// vector.
    pillar_idx: usize,
    /// Initial guess for the discount factor at the pillar.
    d_guess: f64,
    /// Full running `times` vector (curve `t`-axis).
    times: &'a [f64],
    /// Full running `discounts` vector with the previous best estimate at
    /// `pillar_idx`.
    discounts: &'a [f64],
}

/// Dispatches the instrument variant to its `InstrumentLike::residual`
/// implementation. Internal helper used by the residual function passed to
/// Brent.
fn instrument_residual(
    inst: &Instrument,
    reference_date: Date,
    snapshot: &CurveSnapshot<'_>,
) -> Result<f64, BootstrapError> {
    match inst {
        Instrument::Bond(b) => b.residual(reference_date, snapshot),
        Instrument::Deposit(d) => d.residual(reference_date, snapshot),
        Instrument::Fra(f) => f.residual(reference_date, snapshot),
        Instrument::Future(f) => f.residual(reference_date, snapshot),
        Instrument::SwapFixedFloat(s) => s.residual(reference_date, snapshot),
        Instrument::OisSwap(s) => s.residual(reference_date, snapshot),
        Instrument::BasisSwap(b) => b.residual(reference_date, snapshot),
    }
}

/// Returns `true` if the interpolation method's value at one pillar depends
/// on the value at later pillars (i.e. it is a globally-coupled interpolant
/// and the bootstrap requires an outer iteration to reach a self-consistent
/// fixed point).
fn method_is_nonlocal(method: Interpolation) -> bool {
    match method {
        Interpolation::CubicSpline(_)
        | Interpolation::HermiteBessel
        | Interpolation::MonotoneHyman => true,
        // ConvexMonotone is local — each segment depends only on its four
        // adjacent knots — so the sequential bootstrap converges without an
        // outer iteration.
        Interpolation::ConvexMonotone
        | Interpolation::Linear
        | Interpolation::LogLinear
        | Interpolation::LinearInZero
        | Interpolation::PiecewiseConstantForward
        | Interpolation::MonotoneCubic
        | Interpolation::MonotoneSteffen => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruments::{Bond, Deposit, Fra, OisSwap, SwapFixedFloat, SwapSchedule};
    use crate::interpolation::SplineBoundary;
    use crate::types::Frequency;

    fn d(y: i32, m: u32, day: u32) -> Date {
        Date::from_ymd(y, m, day).unwrap()
    }

    // ─── Default configuration ────────────────────────────────────────────

    #[test]
    fn bootstrap_config_default_values_match_spec() {
        let cfg = BootstrapConfig::default();
        assert!((cfg.tolerance - 1e-12).abs() < 1e-18);
        assert_eq!(cfg.max_iter, 100);
        assert!((cfg.bracket - 0.5).abs() < 1e-15);
        assert!(cfg.iterative);
        assert_eq!(cfg.iter_max, 8);
        assert!((cfg.iter_tol - 1e-14).abs() < 1e-20);
    }

    #[test]
    fn bootstrap_with_config_round_trip() {
        let reference = d(2024, 1, 2);
        let cfg = BootstrapConfig {
            tolerance: 1e-10,
            max_iter: 50,
            bracket: 0.25,
            iterative: false,
            iter_max: 4,
            iter_tol: 1e-12,
        };
        let bs = Bootstrap::new(reference, Daycount::Act360).with_config(cfg);
        assert_eq!(bs.config, cfg);
    }

    // ─── Validation errors ────────────────────────────────────────────────

    #[test]
    fn build_rejects_empty_instrument_list() {
        let reference = d(2024, 1, 2);
        let bs = Bootstrap::new(reference, Daycount::Act360);
        let err = bs.build(&[], Interpolation::LogLinear).unwrap_err();
        assert!(matches!(
            err,
            BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "no instruments supplied",
            }
        ));
    }

    #[test]
    fn build_rejects_pillar_on_or_before_reference_date() {
        let reference = d(2024, 1, 2);
        let dep = Deposit::new(reference, reference, 0.05, Daycount::Act360).unwrap();
        let bs = Bootstrap::new(reference, Daycount::Act360);
        let err = bs
            .build(&[Instrument::Deposit(dep)], Interpolation::LogLinear)
            .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn build_rejects_non_increasing_pillars() {
        let reference = d(2024, 1, 2);
        let dep_late = Deposit::new(reference, d(2024, 7, 2), 0.05, Daycount::Act360).unwrap();
        let dep_early = Deposit::new(reference, d(2024, 4, 2), 0.05, Daycount::Act360).unwrap();
        let bs = Bootstrap::new(reference, Daycount::Act360);
        let err = bs
            .build(
                &[
                    Instrument::Deposit(dep_late),
                    Instrument::Deposit(dep_early),
                ],
                Interpolation::LogLinear,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            BootstrapError::NonIncreasingAnchor { at_index: 1 }
        ));
    }

    // ─── Deposit-only bootstrap on a flat 5% curve ────────────────────────

    #[test]
    fn deposit_only_bootstrap_flat_5pct_log_linear() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let rate = 0.05_f64;
        let tenors = [
            (d(2024, 2, 2)), // ~1M
            (d(2024, 3, 2)), // ~2M
            (d(2024, 4, 2)), // ~3M
            (d(2024, 7, 2)), // ~6M
        ];
        let instruments: Vec<Instrument> = tenors
            .iter()
            .map(|&payment| {
                Instrument::Deposit(Deposit::new(reference, payment, rate, dc).unwrap())
            })
            .collect();

        let bs = Bootstrap::new(reference, dc);
        let curve = bs.build(&instruments, Interpolation::LogLinear).unwrap();

        // Re-pricing certificate: every deposit's residual against the final
        // curve must be < tolerance.
        let times: Vec<f64> = curve.times().to_vec();
        let discounts: Vec<f64> = curve.discounts().to_vec();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discounts,
        };
        for inst in &instruments {
            let residual = instrument_residual(inst, reference, &snapshot).unwrap();
            assert!(
                residual.abs() < 1e-12,
                "deposit residual must be < 1e-12, got {residual}",
            );
        }

        // Also confirm D(payment) ~ 1 / (1 + rate * tau).
        for &payment in &tenors {
            let tau = dc.year_fraction(reference, payment).unwrap();
            let expected = 1.0 / (1.0 + rate * tau);
            let got = curve.discount_at(payment).unwrap();
            assert!(
                (got - expected).abs() < 1e-12,
                "D({payment:?}) -> {got}, expected {expected}",
            );
        }
    }

    // ─── Mixed instruments: deposits + FRAs + swap ────────────────────────

    /// Builds a flat continuously-compounded curve at rate `r_c` evaluated on
    /// a daily grid. Used to compute consistent par quotes for synthetic
    /// instruments.
    fn flat_curve_quotes_for_dep(reference: Date, dc: Daycount, r_c: f64, payment: Date) -> f64 {
        // Simply-compounded deposit rate consistent with D(t) = exp(-r_c * t).
        let tau = dc.year_fraction(reference, payment).unwrap();
        let d_pay = (-r_c * tau).exp();
        (1.0 / d_pay - 1.0) / tau
    }

    fn flat_curve_quotes_for_fra(dc: Daycount, r_c: f64, start: Date, end: Date) -> f64 {
        let tau = dc.year_fraction(start, end).unwrap();
        ((r_c * tau).exp() - 1.0) / tau
    }

    fn flat_par_swap_rate(
        reference: Date,
        start: Date,
        maturity: Date,
        freq: Frequency,
        fixed_dc: Daycount,
        curve_dc: Daycount,
        r_c: f64,
    ) -> f64 {
        let schedule = SwapSchedule::from_regular(start, maturity, freq).unwrap();
        let mut annuity = 0.0_f64;
        for i in 0..schedule.len() {
            let p_end = schedule.period_end(i);
            let tau_i = fixed_dc
                .year_fraction(schedule.period_start(i), p_end)
                .unwrap();
            let t_pay = curve_dc.year_fraction(reference, p_end).unwrap();
            annuity += tau_i * (-r_c * t_pay).exp();
        }
        let t_start = curve_dc.year_fraction(reference, start).unwrap();
        let t_mat = curve_dc.year_fraction(reference, maturity).unwrap();
        ((-r_c * t_start).exp() - (-r_c * t_mat).exp()) / annuity
    }

    #[test]
    fn mixed_deposits_fras_swap_bootstrap_log_linear() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.04_f64;

        let dep1_pay = d(2024, 4, 2);
        let dep2_pay = d(2024, 7, 2);
        let fra1_start = d(2024, 7, 2);
        let fra1_end = d(2024, 10, 2);
        let fra2_start = d(2024, 10, 2);
        let fra2_end = d(2025, 1, 2);
        let swap_start = reference;
        let swap_maturity = d(2026, 1, 2);

        let dep1 = Deposit::new(
            reference,
            dep1_pay,
            flat_curve_quotes_for_dep(reference, dc, r_c, dep1_pay),
            dc,
        )
        .unwrap();
        let dep2 = Deposit::new(
            reference,
            dep2_pay,
            flat_curve_quotes_for_dep(reference, dc, r_c, dep2_pay),
            dc,
        )
        .unwrap();
        let fra1 = Fra::new(
            fra1_start,
            fra1_end,
            flat_curve_quotes_for_fra(dc, r_c, fra1_start, fra1_end),
            dc,
        )
        .unwrap();
        let fra2 = Fra::new(
            fra2_start,
            fra2_end,
            flat_curve_quotes_for_fra(dc, r_c, fra2_start, fra2_end),
            dc,
        )
        .unwrap();
        let par = flat_par_swap_rate(
            reference,
            swap_start,
            swap_maturity,
            Frequency::SemiAnnual,
            Daycount::Act360,
            dc,
            r_c,
        );
        let swap = SwapFixedFloat::new(
            swap_start,
            swap_maturity,
            par,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();

        let instruments = [
            Instrument::Deposit(dep1),
            Instrument::Deposit(dep2),
            Instrument::Fra(fra1),
            Instrument::Fra(fra2),
            Instrument::SwapFixedFloat(swap),
        ];
        let bs = Bootstrap::new(reference, dc);
        let curve = bs.build(&instruments, Interpolation::LogLinear).unwrap();

        let times: Vec<f64> = curve.times().to_vec();
        let discounts: Vec<f64> = curve.discounts().to_vec();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discounts,
        };
        for inst in &instruments {
            let residual = instrument_residual(inst, reference, &snapshot).unwrap();
            assert!(
                residual.abs() < 1e-10,
                "mixed residual must be < 1e-10, got {residual}",
            );
        }
    }

    // ─── OIS-only bootstrap ───────────────────────────────────────────────

    #[test]
    fn ois_only_bootstrap_log_linear() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.03_f64;

        let m_1y = d(2025, 1, 2);
        let m_2y = d(2026, 1, 2);
        let m_5y = d(2029, 1, 2);

        let par_1y = flat_par_swap_rate(reference, reference, m_1y, Frequency::Annual, dc, dc, r_c);
        let par_2y = flat_par_swap_rate(reference, reference, m_2y, Frequency::Annual, dc, dc, r_c);
        let par_5y = flat_par_swap_rate(reference, reference, m_5y, Frequency::Annual, dc, dc, r_c);

        let ois_1y = OisSwap::new(reference, m_1y, par_1y, Frequency::Annual, dc).unwrap();
        let ois_2y = OisSwap::new(reference, m_2y, par_2y, Frequency::Annual, dc).unwrap();
        let ois_5y = OisSwap::new(reference, m_5y, par_5y, Frequency::Annual, dc).unwrap();
        let instruments = [
            Instrument::OisSwap(ois_1y),
            Instrument::OisSwap(ois_2y),
            Instrument::OisSwap(ois_5y),
        ];

        let bs = Bootstrap::new(reference, dc);
        let curve = bs.build(&instruments, Interpolation::LogLinear).unwrap();
        let times: Vec<f64> = curve.times().to_vec();
        let discounts: Vec<f64> = curve.discounts().to_vec();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discounts,
        };
        for inst in &instruments {
            let residual = instrument_residual(inst, reference, &snapshot).unwrap();
            assert!(
                residual.abs() < 1e-10,
                "OIS residual must be < 1e-10, got {residual}",
            );
        }
    }

    // ─── Cubic spline bootstrap with outer iteration ──────────────────────

    #[test]
    fn cubic_spline_bootstrap_converges_via_outer_iteration() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.04_f64;

        let dep1_pay = d(2024, 4, 2);
        let dep2_pay = d(2024, 7, 2);
        let fra1_start = d(2024, 7, 2);
        let fra1_end = d(2024, 10, 2);
        let fra2_start = d(2024, 10, 2);
        let fra2_end = d(2025, 1, 2);
        let swap_start = reference;
        let swap_maturity = d(2026, 1, 2);

        let dep1 = Deposit::new(
            reference,
            dep1_pay,
            flat_curve_quotes_for_dep(reference, dc, r_c, dep1_pay),
            dc,
        )
        .unwrap();
        let dep2 = Deposit::new(
            reference,
            dep2_pay,
            flat_curve_quotes_for_dep(reference, dc, r_c, dep2_pay),
            dc,
        )
        .unwrap();
        let fra1 = Fra::new(
            fra1_start,
            fra1_end,
            flat_curve_quotes_for_fra(dc, r_c, fra1_start, fra1_end),
            dc,
        )
        .unwrap();
        let fra2 = Fra::new(
            fra2_start,
            fra2_end,
            flat_curve_quotes_for_fra(dc, r_c, fra2_start, fra2_end),
            dc,
        )
        .unwrap();
        let par = flat_par_swap_rate(
            reference,
            swap_start,
            swap_maturity,
            Frequency::SemiAnnual,
            Daycount::Act360,
            dc,
            r_c,
        );
        let swap = SwapFixedFloat::new(
            swap_start,
            swap_maturity,
            par,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();

        let instruments = [
            Instrument::Deposit(dep1),
            Instrument::Deposit(dep2),
            Instrument::Fra(fra1),
            Instrument::Fra(fra2),
            Instrument::SwapFixedFloat(swap),
        ];
        let bs = Bootstrap::new(reference, dc);
        let curve = bs
            .build(
                &instruments,
                Interpolation::CubicSpline(SplineBoundary::NotAKnot),
            )
            .unwrap();

        let times: Vec<f64> = curve.times().to_vec();
        let discounts: Vec<f64> = curve.discounts().to_vec();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discounts,
        };
        for inst in &instruments {
            let residual = instrument_residual(inst, reference, &snapshot).unwrap();
            assert!(
                residual.abs() < 1e-10,
                "cubic spline residual must be < 1e-10, got {residual}",
            );
        }
    }

    // ─── Every interpolation method works on a small bootstrap ────────────

    #[test]
    fn every_interpolation_method_converges_on_three_deposits() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.04_f64;
        let payments = [d(2024, 4, 2), d(2024, 7, 2), d(2024, 10, 2)];
        let instruments: Vec<Instrument> = payments
            .iter()
            .map(|&p| {
                Instrument::Deposit(
                    Deposit::new(
                        reference,
                        p,
                        flat_curve_quotes_for_dep(reference, dc, r_c, p),
                        dc,
                    )
                    .unwrap(),
                )
            })
            .collect();

        let methods = [
            Interpolation::Linear,
            Interpolation::LogLinear,
            Interpolation::LinearInZero,
            Interpolation::PiecewiseConstantForward,
            Interpolation::CubicSpline(SplineBoundary::NotAKnot),
            Interpolation::ConvexMonotone,
            Interpolation::HermiteBessel,
            Interpolation::MonotoneCubic,
            Interpolation::MonotoneHyman,
            Interpolation::MonotoneSteffen,
        ];
        let bs = Bootstrap::new(reference, dc);
        for method in methods {
            let curve = bs.build(&instruments, method).unwrap_or_else(|e| {
                panic!("method {method:?} failed: {e:?}");
            });
            let times: Vec<f64> = curve.times().to_vec();
            let discounts: Vec<f64> = curve.discounts().to_vec();
            let snapshot = CurveSnapshot {
                reference_date: reference,
                daycount: dc,
                times: &times,
                discounts: &discounts,
            };
            for inst in &instruments {
                let r = instrument_residual(inst, reference, &snapshot).unwrap();
                assert!(r.abs() < 1e-10, "method {method:?}: residual {r}");
            }
        }
    }

    // ─── No-bracket failure path ──────────────────────────────────────────

    #[test]
    fn no_bracket_failure_on_pathological_deposit_rate() {
        // A deposit with rate = -100 over ~91 days has growth factor
        // 1 + r * tau ≈ 1 - 25.28 = -24.28. The residual
        // `D(fix)/D(pay) - (1 + r*tau)` is `positive/positive + 24.28`,
        // which is always strictly positive — no sign change exists. The
        // bracket-expanding search should give up with NoBracket.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let dep = Deposit::new(reference, d(2024, 4, 2), -100.0, dc).unwrap();
        let bs = Bootstrap::new(reference, dc);
        let res = bs.build(&[Instrument::Deposit(dep)], Interpolation::LogLinear);
        match res {
            Err(BootstrapError::NoBracket { at_index: 0 }) => {}
            Err(other) => panic!("expected NoBracket, got {other:?}"),
            Ok(_) => panic!("expected NoBracket error, got Ok"),
        }
    }

    // ─── method_is_nonlocal helper coverage ───────────────────────────────

    #[test]
    fn method_is_nonlocal_distinguishes_local_and_global_methods() {
        assert!(method_is_nonlocal(Interpolation::CubicSpline(
            SplineBoundary::NotAKnot
        )));
        assert!(method_is_nonlocal(Interpolation::HermiteBessel));
        assert!(method_is_nonlocal(Interpolation::MonotoneHyman));
        assert!(!method_is_nonlocal(Interpolation::ConvexMonotone));
        assert!(!method_is_nonlocal(Interpolation::Linear));
        assert!(!method_is_nonlocal(Interpolation::LogLinear));
        assert!(!method_is_nonlocal(Interpolation::LinearInZero));
        assert!(!method_is_nonlocal(Interpolation::PiecewiseConstantForward));
        assert!(!method_is_nonlocal(Interpolation::MonotoneCubic));
        assert!(!method_is_nonlocal(Interpolation::MonotoneSteffen));
    }

    // ─── Debug formatting ─────────────────────────────────────────────────

    #[test]
    fn bootstrap_debug_includes_reference_date() {
        let bs = Bootstrap::new(d(2024, 1, 2), Daycount::Act360);
        let dbg = format!("{bs:?}");
        assert!(dbg.contains("Bootstrap"));
    }

    #[test]
    fn bootstrap_config_debug_includes_struct_name() {
        let cfg = BootstrapConfig::default();
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("BootstrapConfig"));
    }

    // ─── Iterative flag disables outer iteration ──────────────────────────

    #[test]
    fn iterative_off_skips_outer_iteration_local_methods_still_work() {
        // For local methods, the outer iteration is a no-op anyway. Verify
        // the engine produces the same curve regardless of `iterative`.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.04_f64;
        let p1 = d(2024, 4, 2);
        let p2 = d(2024, 7, 2);
        let dep1 = Deposit::new(
            reference,
            p1,
            flat_curve_quotes_for_dep(reference, dc, r_c, p1),
            dc,
        )
        .unwrap();
        let dep2 = Deposit::new(
            reference,
            p2,
            flat_curve_quotes_for_dep(reference, dc, r_c, p2),
            dc,
        )
        .unwrap();
        let instruments = [Instrument::Deposit(dep1), Instrument::Deposit(dep2)];

        let bs_on = Bootstrap::new(reference, dc);
        let bs_off = Bootstrap::new(reference, dc).with_config(BootstrapConfig {
            iterative: false,
            ..BootstrapConfig::default()
        });
        let curve_on = bs_on.build(&instruments, Interpolation::LogLinear).unwrap();
        let curve_off = bs_off
            .build(&instruments, Interpolation::LogLinear)
            .unwrap();
        for (a, b) in curve_on
            .discounts()
            .iter()
            .zip(curve_off.discounts().iter())
        {
            assert!((a - b).abs() < 1e-14);
        }
    }

    // ─── Anchor on bootstrapped curve ─────────────────────────────────────

    #[test]
    fn bootstrapped_curve_has_canonical_anchor() {
        // The returned curve must satisfy D(reference_date) = 1 to f64
        // round-off. The anchor is appended internally before any leg is
        // solved.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let dep = Deposit::new(reference, d(2024, 7, 2), 0.05, Daycount::Act360).unwrap();
        let bs = Bootstrap::new(reference, dc);
        let curve = bs
            .build(&[Instrument::Deposit(dep)], Interpolation::LogLinear)
            .unwrap();
        assert!((curve.times()[0] - 0.0).abs() < 1e-15);
        assert!((curve.discounts()[0] - 1.0).abs() < 1e-15);
        assert!((curve.discount(0.0).unwrap() - 1.0).abs() < 1e-15);
    }

    // ─── Bootstrapped curve has the expected pillar count ─────────────────

    #[test]
    fn bootstrapped_curve_pillar_count_equals_anchor_plus_n_instruments() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.04_f64;
        let payments = [d(2024, 4, 2), d(2024, 7, 2), d(2024, 10, 2), d(2025, 1, 2)];
        let instruments: Vec<Instrument> = payments
            .iter()
            .map(|&p| {
                Instrument::Deposit(
                    Deposit::new(
                        reference,
                        p,
                        flat_curve_quotes_for_dep(reference, dc, r_c, p),
                        dc,
                    )
                    .unwrap(),
                )
            })
            .collect();
        let bs = Bootstrap::new(reference, dc);
        let curve = bs.build(&instruments, Interpolation::LogLinear).unwrap();
        // Anchor + four deposits = five knots.
        assert_eq!(curve.times().len(), 5);
        assert_eq!(curve.discounts().len(), 5);
    }

    // ─── Outer iteration finishes within the spec's 3-5 iteration budget ──

    #[test]
    fn cubic_spline_outer_iteration_succeeds_with_small_iter_max() {
        // The spec claims that the consistent-input cubic spline test
        // converges within 3-5 outer iterations. Set iter_max = 5 and
        // verify it succeeds.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.04_f64;
        let payments = [d(2024, 4, 2), d(2024, 7, 2), d(2024, 10, 2), d(2025, 1, 2)];
        let instruments: Vec<Instrument> = payments
            .iter()
            .map(|&p| {
                Instrument::Deposit(
                    Deposit::new(
                        reference,
                        p,
                        flat_curve_quotes_for_dep(reference, dc, r_c, p),
                        dc,
                    )
                    .unwrap(),
                )
            })
            .collect();
        let bs = Bootstrap::new(reference, dc).with_config(BootstrapConfig {
            iter_max: 5,
            ..BootstrapConfig::default()
        });
        let curve = bs
            .build(
                &instruments,
                Interpolation::CubicSpline(SplineBoundary::NotAKnot),
            )
            .unwrap();
        // Re-pricing certificate still holds.
        let times: Vec<f64> = curve.times().to_vec();
        let discounts: Vec<f64> = curve.discounts().to_vec();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discounts,
        };
        for inst in &instruments {
            let residual = instrument_residual(inst, reference, &snapshot).unwrap();
            assert!(residual.abs() < 1e-10);
        }
    }

    // ─── Outer iteration cap respected ────────────────────────────────────

    #[test]
    fn cubic_spline_outer_iteration_caps_at_zero_returns_did_not_converge_or_ok() {
        // With iter_max = 0 the engine performs only the initial sweep and
        // skips the outer iteration entirely (the `for 0..0` loop body
        // never runs). On consistent inputs the initial sweep is already
        // very close to the fixed point, but the convergence check sees
        // zero changes (since no outer pass ran) — the function returns
        // `LegDidNotConverge` because `iter_max` ran out before a self-
        // consistent pass was observed.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.04_f64;
        let payments = [d(2024, 4, 2), d(2024, 7, 2), d(2024, 10, 2)];
        let instruments: Vec<Instrument> = payments
            .iter()
            .map(|&p| {
                Instrument::Deposit(
                    Deposit::new(
                        reference,
                        p,
                        flat_curve_quotes_for_dep(reference, dc, r_c, p),
                        dc,
                    )
                    .unwrap(),
                )
            })
            .collect();
        let bs = Bootstrap::new(reference, dc).with_config(BootstrapConfig {
            iter_max: 0,
            ..BootstrapConfig::default()
        });
        let res = bs.build(
            &instruments,
            Interpolation::CubicSpline(SplineBoundary::NotAKnot),
        );
        // With iter_max = 0 the outer loop never executes; the cap-exceeded
        // branch fires.
        assert!(matches!(
            res,
            Err(BootstrapError::LegDidNotConverge {
                at_index: usize::MAX,
                ..
            })
        ));
    }

    // ─── Bond bootstrap integration ───────────────────────────────────────

    /// Closed-form par-bond coupon consistent with `D(t) = exp(-r_c * t)`:
    /// solves `coupon * SUM_i tau_i * D(t_i) + D(t_N) = 1` for `coupon`.
    fn flat_par_bond_coupon(
        reference: Date,
        issue: Date,
        maturity: Date,
        freq: Frequency,
        coupon_dc: Daycount,
        curve_dc: Daycount,
        r_c: f64,
    ) -> f64 {
        let schedule = SwapSchedule::from_regular(issue, maturity, freq).unwrap();
        let mut annuity = 0.0_f64;
        for i in 0..schedule.len() {
            let s = schedule.period_start(i);
            let e = schedule.period_end(i);
            let tau = coupon_dc.year_fraction(s, e).unwrap();
            let t_pay = curve_dc.year_fraction(reference, e).unwrap();
            annuity += tau * (-r_c * t_pay).exp();
        }
        let t_n = curve_dc.year_fraction(reference, maturity).unwrap();
        (1.0 - (-r_c * t_n).exp()) / annuity
    }

    #[test]
    fn bond_bootstrap_deposits_plus_three_bonds_reprices_all() {
        // 4 deposits + 3 par bonds (1y, 3y, 5y) at flat-curve-consistent
        // par coupons. Verify every instrument re-prices with residual
        // < 1e-10.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.04_f64;

        // Four deposits to pin the short end.
        let dep_pays = [d(2024, 4, 2), d(2024, 7, 2), d(2024, 10, 2), d(2025, 1, 2)];
        let mut instruments: Vec<Instrument> = dep_pays
            .iter()
            .map(|&p| {
                Instrument::Deposit(
                    Deposit::new(
                        reference,
                        p,
                        flat_curve_quotes_for_dep(reference, dc, r_c, p),
                        dc,
                    )
                    .unwrap(),
                )
            })
            .collect();

        // Three bonds: 1y is the last deposit pillar, so use 2y/3y/5y to
        // keep pillars strictly increasing past the deposits.
        let bond_mats = [d(2026, 1, 2), d(2027, 1, 2), d(2029, 1, 2)];
        for &m in &bond_mats {
            let coupon =
                flat_par_bond_coupon(reference, reference, m, Frequency::Annual, dc, dc, r_c);
            let bond =
                Bond::new(reference, m, coupon, Frequency::Annual, dc, 1.0, 1.0, 0.0).unwrap();
            instruments.push(Instrument::Bond(bond));
        }

        let bs = Bootstrap::new(reference, dc);
        let curve = bs.build(&instruments, Interpolation::LogLinear).unwrap();
        let times: Vec<f64> = curve.times().to_vec();
        let discounts: Vec<f64> = curve.discounts().to_vec();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discounts,
        };
        for inst in &instruments {
            let residual = instrument_residual(inst, reference, &snapshot).unwrap();
            assert!(
                residual.abs() < 1e-10,
                "bond bootstrap residual must be < 1e-10, got {residual}",
            );
        }
    }

    #[test]
    fn bond_bootstrap_mixed_with_swap_reprices_all() {
        // 2 deposits + 1 bond (3y) + 1 swap (5y). Verify all re-price.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.035_f64;

        let dep1_pay = d(2024, 4, 2);
        let dep2_pay = d(2024, 7, 2);
        let bond_mat = d(2027, 1, 2);
        let swap_mat = d(2029, 1, 2);

        let dep1 = Deposit::new(
            reference,
            dep1_pay,
            flat_curve_quotes_for_dep(reference, dc, r_c, dep1_pay),
            dc,
        )
        .unwrap();
        let dep2 = Deposit::new(
            reference,
            dep2_pay,
            flat_curve_quotes_for_dep(reference, dc, r_c, dep2_pay),
            dc,
        )
        .unwrap();
        let bond_coupon = flat_par_bond_coupon(
            reference,
            reference,
            bond_mat,
            Frequency::Annual,
            dc,
            dc,
            r_c,
        );
        let bond = Bond::new(
            reference,
            bond_mat,
            bond_coupon,
            Frequency::Annual,
            dc,
            1.0,
            1.0,
            0.0,
        )
        .unwrap();
        let par = flat_par_swap_rate(
            reference,
            reference,
            swap_mat,
            Frequency::SemiAnnual,
            Daycount::Act360,
            dc,
            r_c,
        );
        let swap = SwapFixedFloat::new(
            reference,
            swap_mat,
            par,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();

        let instruments = [
            Instrument::Deposit(dep1),
            Instrument::Deposit(dep2),
            Instrument::Bond(bond),
            Instrument::SwapFixedFloat(swap),
        ];
        let bs = Bootstrap::new(reference, dc);
        let curve = bs.build(&instruments, Interpolation::LogLinear).unwrap();
        let times: Vec<f64> = curve.times().to_vec();
        let discounts: Vec<f64> = curve.discounts().to_vec();
        let snapshot = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &times,
            discounts: &discounts,
        };
        for inst in &instruments {
            let residual = instrument_residual(inst, reference, &snapshot).unwrap();
            assert!(
                residual.abs() < 1e-10,
                "mixed bond+swap residual must be < 1e-10, got {residual}",
            );
        }
    }
}
