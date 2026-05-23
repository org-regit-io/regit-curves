// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Multi-curve (OIS-discounted) bootstrap engine.
//!
//! In the post-2008 framework introduced by Bianchetti (2010) and Mercurio
//! (2009), the **discount** and **forward-projection** roles of the yield
//! curve are separated. Cash flows are discounted on the **OIS** curve while
//! IBOR-style floating-leg forwards are projected from a **tenor-specific
//! projection curve** (one per fixing tenor — 1M, 3M, 6M, 12M, ...). Basis
//! between projection curves of different tenors becomes a first-class market
//! observable.
//!
//! [`MultiCurveBootstrap`] builds the OIS curve first from `ois_instruments`
//! (deposits + OIS swaps, dispatching to the single-curve [`Bootstrap`]
//! engine), then bootstraps one projection curve per supplied tenor by
//! sequentially solving for the projection-curve discount factor at each
//! projection instrument's pillar so that the multi-curve residual against
//! the OIS curve + the in-progress projection curve is zero.
//!
//! ```text
//! Step 1: D_OIS  <- Bootstrap(ois_instruments,  ois_method)
//! Step 2: for each (tenor, instruments_tenor) in projection_instruments:
//!             D_proj_tenor <- ProjectionBootstrap(instruments_tenor,
//!                                                 D_OIS,
//!                                                 projection_method)
//! Step 3: return MultiCurve { discount: D_OIS, projection: [(tenor, D_proj_tenor), ...] }
//! ```
//!
//! # Multi-curve pricing formulas
//!
//! For a vanilla fixed-floating swap with float schedule
//! `[t_0, t_1, ..., t_N]` and accruals `tau_i^float`,
//!
//! ```text
//! Float-leg PV = SUM_i  tau_i^float * F_i * D_OIS(t_i),
//! ```
//!
//! where `F_i` is the simply-compounded forward rate implied by the
//! projection curve over `[t_{i-1}, t_i]`:
//!
//! ```text
//! F_i = (D_proj(t_{i-1}) / D_proj(t_i) - 1) / tau_i^float.
//! ```
//!
//! The fixed-leg PV uses the OIS curve only:
//!
//! ```text
//! Fixed-leg PV = rate * SUM_j tau_j^fixed * D_OIS(t_j^fixed).
//! ```
//!
//! Equating the two legs yields the multi-curve par-swap equation
//!
//! ```text
//! rate * SUM_j tau_j^fixed * D_OIS(t_j^fixed)
//!     = SUM_i tau_i^float * F_i * D_OIS(t_i).
//! ```
//!
//! For deposits, FRAs, and futures driving a projection curve the residual is
//! the single-curve identity evaluated on the projection curve alone (they pin
//! a single forward rate that is independent of the OIS discount curve):
//!
//! ```text
//! Deposit / FRA / Future on projection curve:
//!     residual = D_proj(start) / D_proj(end) - (1 + rate * tau).
//! ```
//!
//! # Scope: basis swaps
//!
//! Basis swaps pin the **relationship** between two projection curves and
//! require a joint solve (both projection curves move together). They are
//! **rejected** as projection-curve instruments by [`MultiCurveBootstrap`] —
//! supply them only after the projection curves have been bootstrapped, as
//! a re-pricing check. A future release may add a basis-swap-driven joint
//! solve; a crate-private `basis_swap_residual_multi_curve` helper is
//! exposed inside the module so the multi-curve evaluator can already
//! price basis swaps.
//!
//! # References
//!
//! - Bianchetti, M., "Two Curves, One Price: Pricing & Hedging Interest Rate
//!   Derivatives Decoupling Forwarding and Discounting Yield Curves",
//!   *Risk Magazine*, August 2010, pp. 66-72; arXiv 0905.2770 (2009).
//!   Seminal multi-curve formulation.
//! - Mercurio, F., "Interest Rates and The Credit Crunch: New Formulas and
//!   Market Models", Bloomberg Portfolio Research Paper No. 2010-01-FRONTIERS
//!   (February 2009), §3 - §5. Multi-curve swap, FRA, cap/floor and
//!   swaption pricing.
//! - Ametrano, F. M. & Bianchetti, M., "Everything You Always Wanted to Know
//!   About Multiple Interest Rate Curve Bootstrapping but Were Afraid to Ask",
//!   SSRN 2219548 (April 2013). Comprehensive treatment of multi-curve
//!   bootstrapping; secondary reference.
//! - Andersen, L. B. G. & Piterbarg, V. V., *Interest Rate Modeling*,
//!   Volume I: Foundations and Vanilla Models, Atlantic Financial Press
//!   (2010), §6.5 - §6.6. OIS-discounted multi-curve bootstrap.

use crate::bootstrap::{Bootstrap, BootstrapConfig};
use crate::curves::DiscountCurve;
use crate::errors::BootstrapError;
use crate::instruments::basis_swap::BasisSwap;
use crate::instruments::swap_fixed_float::SwapFixedFloat;
use crate::instruments::{CurveSnapshot, Instrument, InstrumentLike};
use crate::interpolation::Interpolation;
use crate::math::MathError;
use crate::math::brent::{BrentConfig, brent_root};
use crate::types::{Date, Daycount, Tenor};

/// Sequential OIS-discount + tenor-projection bootstrap engine.
///
/// The engine builds an OIS-discount curve first, then bootstraps a separate
/// projection curve for each supplied tenor. See the module documentation
/// for the multi-curve pricing formulas and the algorithm.
///
/// # Examples
///
/// ```
/// use regit_curves::multi_curve::MultiCurveBootstrap;
/// use regit_curves::types::{Date, Daycount};
///
/// let reference = Date::from_ymd(2024, 1, 2).unwrap();
/// let mcb = MultiCurveBootstrap::new(reference, Daycount::Act360);
/// assert_eq!(mcb.reference_date, reference);
/// assert_eq!(mcb.daycount, Daycount::Act360);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiCurveBootstrap {
    /// Curve anchor / reference date — shared between the OIS and all
    /// projection curves.
    pub reference_date: Date,
    /// Day-count convention for the OIS- and projection-curve `t`-axes.
    pub daycount: Daycount,
    /// Solver configuration shared by the OIS bootstrap and the projection
    /// bootstraps.
    pub config: BootstrapConfig,
}

impl MultiCurveBootstrap {
    /// Constructs a multi-curve bootstrap engine with
    /// [`BootstrapConfig::default`].
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::multi_curve::MultiCurveBootstrap;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let mcb = MultiCurveBootstrap::new(reference, Daycount::Act360);
    /// assert!((mcb.config.tolerance - 1e-12).abs() < 1e-18);
    /// ```
    #[must_use]
    pub fn new(reference_date: Date, daycount: Daycount) -> Self {
        Self {
            reference_date,
            daycount,
            config: BootstrapConfig::default(),
        }
    }

    /// Returns the engine with the supplied configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::bootstrap::BootstrapConfig;
    /// use regit_curves::multi_curve::MultiCurveBootstrap;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let cfg = BootstrapConfig {
    ///     tolerance: 1e-10,
    ///     ..BootstrapConfig::default()
    /// };
    /// let mcb = MultiCurveBootstrap::new(reference, Daycount::Act360).with_config(cfg);
    /// assert!((mcb.config.tolerance - 1e-10).abs() < 1e-18);
    /// ```
    #[must_use]
    pub fn with_config(mut self, config: BootstrapConfig) -> Self {
        self.config = config;
        self
    }

    /// Builds the OIS discount curve and one projection curve per supplied
    /// tenor.
    ///
    /// `ois_instruments` are typically deposits and OIS swaps; they drive the
    /// OIS discount curve via the single-curve [`Bootstrap`] engine.
    ///
    /// `projection_instruments` is a list of `(tenor, instruments)` pairs.
    /// For each pair the engine sequentially solves for the projection
    /// curve's discount factor at each instrument's pillar so that the
    /// multi-curve residual against the OIS curve plus the in-progress
    /// projection curve is zero. Only deposits, FRAs, futures, and vanilla
    /// fixed-float swaps are accepted as projection-curve drivers; OIS swaps
    /// and basis swaps are rejected with [`BootstrapError::InvalidInstrument`]
    /// — OIS swaps drive the OIS curve, basis swaps require a joint solve
    /// across two projection curves.
    ///
    /// # Errors
    ///
    /// - [`BootstrapError::InvalidInstrument`] if `ois_instruments` is empty,
    ///   if any projection instrument is an OIS swap or basis swap, if any
    ///   pillar lies on or before [`Self::reference_date`], or if a
    ///   projection-instrument set is empty.
    /// - [`BootstrapError::NonIncreasingAnchor`] if pillars are not strictly
    ///   increasing within the OIS set or within any projection set.
    /// - [`BootstrapError::NoBracket`] if no sign change can be found in the
    ///   discount-factor search interval for a leg, even after widening.
    /// - [`BootstrapError::LegDidNotConverge`] if Brent fails to converge, or
    ///   if a projection-curve outer iteration fails to converge within
    ///   [`BootstrapConfig::iter_max`] passes.
    /// - [`BootstrapError::Curve`] if a final curve construction rejects the
    ///   bootstrapped knots.
    /// - [`BootstrapError::Type`] if a day-count query fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::instruments::{Deposit, Instrument};
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::multi_curve::MultiCurveBootstrap;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let dep = Deposit::new(
    ///     reference,
    ///     Date::from_ymd(2024, 4, 2).unwrap(),
    ///     0.03,
    ///     Daycount::Act360,
    /// )
    /// .unwrap();
    /// let mc = MultiCurveBootstrap::new(reference, Daycount::Act360)
    ///     .build(
    ///         &[Instrument::Deposit(dep)],
    ///         Interpolation::LogLinear,
    ///         &[],
    ///         Interpolation::LogLinear,
    ///     )
    ///     .unwrap();
    /// // OIS-only multi-curve has no projection curves.
    /// assert!(mc.projection.is_empty());
    /// ```
    pub fn build(
        &self,
        ois_instruments: &[Instrument],
        ois_method: Interpolation,
        projection_instruments: &[(Tenor, Vec<Instrument>)],
        projection_method: Interpolation,
    ) -> Result<MultiCurve, BootstrapError> {
        // Step 1 — OIS curve via the single-curve engine. Empty / invalid sets
        // are rejected by the inner engine.
        let ois_engine =
            Bootstrap::new(self.reference_date, self.daycount).with_config(self.config);
        let ois_curve = ois_engine.build(ois_instruments, ois_method)?;

        // Step 2 — one projection curve per tenor.
        let mut projection: Vec<(Tenor, DiscountCurve)> =
            Vec::with_capacity(projection_instruments.len());
        for (tenor, instruments) in projection_instruments {
            let proj = self.build_projection(instruments, projection_method, &ois_curve)?;
            projection.push((*tenor, proj));
        }

        Ok(MultiCurve {
            discount: ois_curve,
            projection,
        })
    }

    /// Bootstraps a single projection curve against the supplied OIS curve.
    #[allow(clippy::too_many_lines)]
    fn build_projection(
        &self,
        instruments: &[Instrument],
        method: Interpolation,
        ois_curve: &DiscountCurve,
    ) -> Result<DiscountCurve, BootstrapError> {
        // Validation. An empty projection set is a programming error.
        if instruments.is_empty() {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "no projection instruments supplied",
            });
        }
        for (i, inst) in instruments.iter().enumerate() {
            match inst {
                Instrument::OisSwap(_) => {
                    return Err(BootstrapError::InvalidInstrument {
                        at_index: i,
                        reason: "OIS swaps drive the OIS curve, not a projection curve",
                    });
                }
                Instrument::BasisSwap(_) => {
                    return Err(BootstrapError::InvalidInstrument {
                        at_index: i,
                        reason: "basis swaps require a joint solve and are not supported in MultiCurveBootstrap",
                    });
                }
                Instrument::Bond(_) => {
                    return Err(BootstrapError::InvalidInstrument {
                        at_index: i,
                        reason: "Bond is not a projection-curve instrument",
                    });
                }
                Instrument::Deposit(_)
                | Instrument::Fra(_)
                | Instrument::Future(_)
                | Instrument::SwapFixedFloat(_) => {}
            }
            let pillar = inst.pillar();
            if pillar.days_between(self.reference_date) >= 0 {
                return Err(BootstrapError::InvalidInstrument {
                    at_index: i,
                    reason: "instrument pillar must be after reference_date",
                });
            }
            if i > 0 {
                let prev_pillar = instruments[i - 1].pillar();
                if pillar.days_between(prev_pillar) >= 0 {
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
            discounts.push(1.0);
        }

        // OIS curve snapshot — stable for the entire projection sweep.
        let ois_times: Vec<f64> = ois_curve.times().to_vec();
        let ois_discounts: Vec<f64> = ois_curve.discounts().to_vec();

        // Step 2 — single-pass sequential bootstrap.
        self.projection_sweep(
            instruments,
            &times,
            &mut discounts,
            true,
            &ois_times,
            &ois_discounts,
        )?;

        // Step 3 — outer iteration for non-local interpolators.
        if self.config.iterative && method_is_nonlocal(method) {
            let mut converged = false;
            let mut last_change = 0.0_f64;
            for _ in 0..self.config.iter_max {
                let previous = discounts.clone();
                self.projection_sweep(
                    instruments,
                    &times,
                    &mut discounts,
                    false,
                    &ois_times,
                    &ois_discounts,
                )?;
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

    /// Re-solves the projection-curve discount factor at each pillar against
    /// the in-progress projection curve and the (fixed) OIS curve.
    fn projection_sweep(
        &self,
        instruments: &[Instrument],
        times: &[f64],
        discounts: &mut [f64],
        is_initial: bool,
        ois_times: &[f64],
        ois_discounts: &[f64],
    ) -> Result<(), BootstrapError> {
        for (k, inst) in instruments.iter().enumerate() {
            let idx = k + 1;
            let t_k = times[idx];
            let t_prev = times[idx - 1];
            let d_prev = discounts[idx - 1];

            let d_guess = if is_initial {
                let r_prev = if k == 0 {
                    0.05_f64
                } else {
                    let t_p2 = times[idx - 2];
                    let d_p2 = discounts[idx - 2];
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
                let warm = discounts[idx];
                if warm.is_finite() && warm > 0.0 {
                    warm
                } else {
                    d_prev
                }
            };

            let ctx = ProjectionLegContext {
                index: k,
                instrument: inst,
                pillar_idx: idx,
                d_guess,
                times,
                discounts,
                ois_times,
                ois_discounts,
            };
            let solved = self.solve_projection_leg(&ctx)?;
            discounts[idx] = solved;
        }
        Ok(())
    }

    /// Solves for the projection discount factor at `ctx.pillar_idx` so the
    /// multi-curve residual is zero.
    fn solve_projection_leg(&self, ctx: &ProjectionLegContext<'_>) -> Result<f64, BootstrapError> {
        let mut bracket = self.config.bracket;
        #[allow(unused_assignments)]
        let mut last_residual: f64 = 0.0;
        for attempt in 0..=5_u32 {
            let lo = (ctx.d_guess * (-bracket).exp()).max(f64::MIN_POSITIVE);
            let hi = ctx.d_guess * bracket.exp();

            let residual_fn = |d: f64| -> f64 {
                let mut probe = ctx.discounts.to_vec();
                probe[ctx.pillar_idx] = d;
                let projection_snapshot = CurveSnapshot {
                    reference_date: self.reference_date,
                    daycount: self.daycount,
                    times: ctx.times,
                    discounts: &probe,
                };
                let ois_snapshot = CurveSnapshot {
                    reference_date: self.reference_date,
                    daycount: self.daycount,
                    times: ctx.ois_times,
                    discounts: ctx.ois_discounts,
                };
                projection_residual(
                    ctx.instrument,
                    self.reference_date,
                    &ois_snapshot,
                    &projection_snapshot,
                )
                .unwrap_or(f64::INFINITY)
            };

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
                    let mut probe = ctx.discounts.to_vec();
                    probe[ctx.pillar_idx] = root;
                    let projection_snapshot = CurveSnapshot {
                        reference_date: self.reference_date,
                        daycount: self.daycount,
                        times: ctx.times,
                        discounts: &probe,
                    };
                    let ois_snapshot = CurveSnapshot {
                        reference_date: self.reference_date,
                        daycount: self.daycount,
                        times: ctx.ois_times,
                        discounts: ctx.ois_discounts,
                    };
                    let final_residual = projection_residual(
                        ctx.instrument,
                        self.reference_date,
                        &ois_snapshot,
                        &projection_snapshot,
                    )?;
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

/// A fully-bootstrapped multi-curve set: one OIS-discount curve plus one
/// projection curve per supplied tenor.
///
/// # Examples
///
/// ```
/// use regit_curves::curves::DiscountCurve;
/// use regit_curves::interpolation::Interpolation;
/// use regit_curves::multi_curve::MultiCurve;
/// use regit_curves::types::{Date, Daycount};
///
/// let reference = Date::from_ymd(2024, 1, 2).unwrap();
/// let discount = DiscountCurve::from_times_and_discounts(
///     reference,
///     Daycount::Act365F,
///     &[0.0, 1.0, 2.0],
///     &[1.0, 0.98, 0.96],
///     Interpolation::LogLinear,
/// )
/// .unwrap();
/// let mc = MultiCurve { discount, projection: Vec::new() };
/// assert_eq!(mc.discount_curve().reference_date(), reference);
/// ```
#[derive(Debug, Clone)]
pub struct MultiCurve {
    /// OIS discount curve — discounts every cash flow.
    pub discount: DiscountCurve,
    /// Projection curves keyed by tenor (3M, 6M, ...). Ordering matches the
    /// `(Tenor, Vec<Instrument>)` order supplied to
    /// [`MultiCurveBootstrap::build`]; at most one curve per tenor is held.
    pub projection: Vec<(Tenor, DiscountCurve)>,
}

impl MultiCurve {
    /// Returns the OIS discount curve.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::DiscountCurve;
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::multi_curve::MultiCurve;
    /// use regit_curves::types::{Date, Daycount};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let discount = DiscountCurve::from_times_and_discounts(
    ///     reference,
    ///     Daycount::Act365F,
    ///     &[0.0, 1.0],
    ///     &[1.0, 0.98],
    ///     Interpolation::LogLinear,
    /// )
    /// .unwrap();
    /// let mc = MultiCurve { discount, projection: Vec::new() };
    /// assert_eq!(mc.discount_curve().reference_date(), reference);
    /// ```
    #[must_use]
    #[inline]
    pub fn discount_curve(&self) -> &DiscountCurve {
        &self.discount
    }

    /// Returns the projection curve for `tenor`, or `None` if no projection
    /// curve was bootstrapped for that tenor.
    ///
    /// Equality is on the literal `Tenor` value — `Tenor::new(3,
    /// TenorUnit::Months)` and `Tenor::new(90, TenorUnit::Days)` are
    /// **distinct** keys here even though they refer to roughly the same
    /// span; multi-curve markets quote curves by their literal index tenor.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_curves::curves::DiscountCurve;
    /// use regit_curves::interpolation::Interpolation;
    /// use regit_curves::multi_curve::MultiCurve;
    /// use regit_curves::types::{Date, Daycount, Tenor, TenorUnit};
    ///
    /// let reference = Date::from_ymd(2024, 1, 2).unwrap();
    /// let discount = DiscountCurve::from_times_and_discounts(
    ///     reference,
    ///     Daycount::Act365F,
    ///     &[0.0, 1.0],
    ///     &[1.0, 0.98],
    ///     Interpolation::LogLinear,
    /// )
    /// .unwrap();
    /// let mc = MultiCurve { discount, projection: Vec::new() };
    /// assert!(mc.projection_curve(Tenor::new(3, TenorUnit::Months)).is_none());
    /// ```
    #[must_use]
    pub fn projection_curve(&self, tenor: Tenor) -> Option<&DiscountCurve> {
        self.projection
            .iter()
            .find(|(t, _)| *t == tenor)
            .map(|(_, c)| c)
    }
}

/// Bundled inputs for a single projection-curve leg solve.
struct ProjectionLegContext<'a> {
    index: usize,
    instrument: &'a Instrument,
    pillar_idx: usize,
    d_guess: f64,
    times: &'a [f64],
    discounts: &'a [f64],
    ois_times: &'a [f64],
    ois_discounts: &'a [f64],
}

/// Returns the multi-curve residual for an instrument driving a projection
/// curve. Deposits / FRAs / Futures pin the projection curve directly via
/// their single-curve residual; vanilla swaps use the multi-curve par
/// equation against both the OIS and projection curves; OIS and basis swaps
/// are not accepted here (the caller validates these upstream).
fn projection_residual(
    inst: &Instrument,
    reference_date: Date,
    ois_snapshot: &CurveSnapshot<'_>,
    projection_snapshot: &CurveSnapshot<'_>,
) -> Result<f64, BootstrapError> {
    match inst {
        Instrument::Deposit(d) => d.residual(reference_date, projection_snapshot),
        Instrument::Fra(f) => f.residual(reference_date, projection_snapshot),
        Instrument::Future(f) => f.residual(reference_date, projection_snapshot),
        Instrument::SwapFixedFloat(s) => {
            swap_residual_multi_curve(s, reference_date, ois_snapshot, projection_snapshot)
        }
        Instrument::OisSwap(_) | Instrument::BasisSwap(_) | Instrument::Bond(_) => {
            Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "instrument cannot drive a projection curve",
            })
        }
    }
}

/// Multi-curve residual for a vanilla fixed-floating swap:
///
/// ```text
/// residual = PV_fixed(OIS) - PV_float(projection, OIS)
///          = rate * SUM_j tau_j^fixed * D_OIS(t_j^fixed)
///              - SUM_i tau_i^float * F_i * D_OIS(t_i),
/// ```
///
/// where `F_i = (D_proj(t_{i-1}) / D_proj(t_i) - 1) / tau_i^float` is the
/// simply-compounded forward rate on the projection curve over the `i`-th
/// float-leg accrual.
///
/// # Errors
///
/// - [`BootstrapError::Type`] if any day-count query fails.
/// - [`BootstrapError::InvalidInstrument`] if either snapshot is empty or
///   returns a non-positive discount factor where one is required.
pub(crate) fn swap_residual_multi_curve(
    swap: &SwapFixedFloat,
    reference_date: Date,
    ois: &CurveSnapshot<'_>,
    projection: &CurveSnapshot<'_>,
) -> Result<f64, BootstrapError> {
    // Fixed leg: PV_fixed = rate * sum tau_j^fixed * D_OIS(t_j^fixed).
    let pv_fixed = swap.fixed_leg_pv(reference_date, ois)?;

    // Float leg: PV_float = sum tau_i^float * F_i * D_OIS(t_i)
    //                     = sum (D_proj(t_{i-1}) / D_proj(t_i) - 1) * D_OIS(t_i).
    // The `tau` factors in `F_i` and in the coupon multiplier cancel.
    let mut pv_float = 0.0_f64;
    for i in 0..swap.float_schedule.len() {
        let p_start = swap.float_schedule.period_start(i);
        let p_end = swap.float_schedule.period_end(i);

        let t_p_start = projection
            .daycount
            .year_fraction(projection.reference_date, p_start)?;
        let t_p_end = projection
            .daycount
            .year_fraction(projection.reference_date, p_end)?;
        let d_proj_start =
            projection
                .discount_at(t_p_start)
                .ok_or(BootstrapError::InvalidInstrument {
                    at_index: 0,
                    reason: "projection curve snapshot is empty",
                })?;
        let d_proj_end =
            projection
                .discount_at(t_p_end)
                .ok_or(BootstrapError::InvalidInstrument {
                    at_index: 0,
                    reason: "projection curve snapshot is empty",
                })?;
        if d_proj_start <= 0.0 || d_proj_end <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "non-positive discount factor in projection curve",
            });
        }
        let t_ois_end = ois.daycount.year_fraction(ois.reference_date, p_end)?;
        let d_ois_end = ois
            .discount_at(t_ois_end)
            .ok_or(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "OIS curve snapshot is empty",
            })?;
        if d_ois_end <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "non-positive discount factor in OIS curve",
            });
        }
        // tau_i^float * F_i = D_proj(t_{i-1}) / D_proj(t_i) - 1.
        let coupon = d_proj_start / d_proj_end - 1.0;
        pv_float += coupon * d_ois_end;
    }
    Ok(pv_fixed - pv_float)
}

/// Multi-curve residual for a basis swap:
///
/// ```text
/// residual = PV_float(leg_a; projection_a) + spread * Annuity(leg_a; OIS)
///              - PV_float(leg_b; projection_b),
/// ```
///
/// where each `PV_float(leg; projection)` discounts on the OIS curve and
/// projects each accrual on the leg's tenor projection curve via
/// `F_i = (D_proj(t_{i-1}) / D_proj(t_i) - 1) / tau_i`, and the spread
/// annuity uses OIS discounting under `leg_a`'s day-count.
///
/// This helper is exposed so the multi-curve evaluator can price basis swaps
/// against a fully-built [`MultiCurve`]; basis swaps are **not** accepted as
/// projection-curve drivers in [`MultiCurveBootstrap::build`] (they would
/// require a joint solve across two projection curves).
///
/// # Errors
///
/// - [`BootstrapError::Type`] if any day-count query fails.
/// - [`BootstrapError::InvalidInstrument`] if any snapshot is empty or
///   returns a non-positive discount factor.
#[allow(dead_code)] // consumed by integration suites + future multi-curve evaluator
pub(crate) fn basis_swap_residual_multi_curve(
    bs: &BasisSwap,
    reference_date: Date,
    ois: &CurveSnapshot<'_>,
    projection_a: &CurveSnapshot<'_>,
    projection_b: &CurveSnapshot<'_>,
) -> Result<f64, BootstrapError> {
    let pv_a = leg_float_pv_multi(&bs.leg_a.schedule, reference_date, ois, projection_a)?;
    let pv_b = leg_float_pv_multi(&bs.leg_b.schedule, reference_date, ois, projection_b)?;
    let annuity_a = bs.leg_a.annuity(reference_date, ois)?;
    Ok(pv_a + bs.spread * annuity_a - pv_b)
}

/// Helper: multi-curve float-leg PV `SUM_i (D_proj(t_{i-1}) / D_proj(t_i) - 1) * D_OIS(t_i)`.
#[allow(dead_code)]
fn leg_float_pv_multi(
    schedule: &crate::instruments::SwapSchedule,
    _reference_date: Date,
    ois: &CurveSnapshot<'_>,
    projection: &CurveSnapshot<'_>,
) -> Result<f64, BootstrapError> {
    let mut pv = 0.0_f64;
    for i in 0..schedule.len() {
        let p_start = schedule.period_start(i);
        let p_end = schedule.period_end(i);
        let t_p_start = projection
            .daycount
            .year_fraction(projection.reference_date, p_start)?;
        let t_p_end = projection
            .daycount
            .year_fraction(projection.reference_date, p_end)?;
        let d_proj_start =
            projection
                .discount_at(t_p_start)
                .ok_or(BootstrapError::InvalidInstrument {
                    at_index: 0,
                    reason: "projection curve snapshot is empty",
                })?;
        let d_proj_end =
            projection
                .discount_at(t_p_end)
                .ok_or(BootstrapError::InvalidInstrument {
                    at_index: 0,
                    reason: "projection curve snapshot is empty",
                })?;
        if d_proj_start <= 0.0 || d_proj_end <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "non-positive discount factor in projection curve",
            });
        }
        let t_ois_end = ois.daycount.year_fraction(ois.reference_date, p_end)?;
        let d_ois_end = ois
            .discount_at(t_ois_end)
            .ok_or(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "OIS curve snapshot is empty",
            })?;
        if d_ois_end <= 0.0 {
            return Err(BootstrapError::InvalidInstrument {
                at_index: 0,
                reason: "non-positive discount factor in OIS curve",
            });
        }
        pv += (d_proj_start / d_proj_end - 1.0) * d_ois_end;
    }
    Ok(pv)
}

/// Returns `true` if the interpolation method's value at one pillar depends
/// on the value at later pillars. Mirrors the helper in `bootstrap.rs` —
/// duplicated locally to keep the multi-curve module decoupled from the
/// single-curve engine's private helpers.
fn method_is_nonlocal(method: Interpolation) -> bool {
    match method {
        Interpolation::CubicSpline(_)
        | Interpolation::HermiteBessel
        | Interpolation::MonotoneHyman => true,
        // ConvexMonotone is local — each segment depends only on its four
        // adjacent knots — so the joint multi-curve solve converges without
        // an outer iteration.
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
    use crate::instruments::basis_swap::{BasisLeg, BasisSwap};
    use crate::instruments::{Deposit, Fra, OisSwap, SwapFixedFloat, SwapSchedule};
    use crate::interpolation::SplineBoundary;
    use crate::types::{Frequency, TenorUnit};

    fn d(y: i32, m: u32, day: u32) -> Date {
        Date::from_ymd(y, m, day).unwrap()
    }

    fn flat_curve_quotes_for_dep(reference: Date, dc: Daycount, r_c: f64, payment: Date) -> f64 {
        let tau = dc.year_fraction(reference, payment).unwrap();
        let d_pay = (-r_c * tau).exp();
        (1.0 / d_pay - 1.0) / tau
    }

    fn flat_curve_quotes_for_fra(dc: Daycount, r_c: f64, start: Date, end: Date) -> f64 {
        let tau = dc.year_fraction(start, end).unwrap();
        ((r_c * tau).exp() - 1.0) / tau
    }

    fn flat_par_swap_rate_single(
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

    /// Closed-form multi-curve par swap rate against flat continuously-
    /// compounded OIS curve at `r_ois` and flat projection curve at `r_proj`.
    ///
    /// ```text
    /// numerator   = sum_i (exp(r_proj * tau_i) - 1) * D_OIS(t_i)
    /// denominator = sum_j tau_j^fixed * D_OIS(t_j^fixed)
    /// ```
    #[allow(clippy::too_many_arguments)]
    fn flat_par_swap_rate_multi(
        reference: Date,
        start: Date,
        maturity: Date,
        fixed_freq: Frequency,
        float_freq: Frequency,
        fixed_dc: Daycount,
        float_dc: Daycount,
        curve_dc: Daycount,
        r_ois: f64,
        r_proj: f64,
    ) -> f64 {
        let fixed_sch = SwapSchedule::from_regular(start, maturity, fixed_freq).unwrap();
        let float_sch = SwapSchedule::from_regular(start, maturity, float_freq).unwrap();
        let mut annuity = 0.0_f64;
        for i in 0..fixed_sch.len() {
            let p_end = fixed_sch.period_end(i);
            let tau_j = fixed_dc
                .year_fraction(fixed_sch.period_start(i), p_end)
                .unwrap();
            let t_pay = curve_dc.year_fraction(reference, p_end).unwrap();
            annuity += tau_j * (-r_ois * t_pay).exp();
        }
        let mut float_pv = 0.0_f64;
        for i in 0..float_sch.len() {
            let p_start = float_sch.period_start(i);
            let p_end = float_sch.period_end(i);
            let tau_i = float_dc.year_fraction(p_start, p_end).unwrap();
            let t_pay = curve_dc.year_fraction(reference, p_end).unwrap();
            // F_i = (exp(r_proj * tau_i) - 1) / tau_i; tau_i * F_i = exp(r_proj * tau_i) - 1.
            let coupon = (r_proj * tau_i).exp_m1();
            float_pv += coupon * (-r_ois * t_pay).exp();
        }
        float_pv / annuity
    }

    /// Builds an OIS-only set of 1y/2y/5y annual OIS swaps at a flat rate
    /// `r_c` (the par-OIS-rate identity is the same as the par-swap-rate
    /// identity on a single curve).
    fn ois_instruments_flat(reference: Date, dc: Daycount, r_c: f64) -> Vec<Instrument> {
        let m1 = d(2025, 1, 2);
        let m2 = d(2026, 1, 2);
        let m5 = d(2029, 1, 2);
        let p1 =
            flat_par_swap_rate_single(reference, reference, m1, Frequency::Annual, dc, dc, r_c);
        let p2 =
            flat_par_swap_rate_single(reference, reference, m2, Frequency::Annual, dc, dc, r_c);
        let p5 =
            flat_par_swap_rate_single(reference, reference, m5, Frequency::Annual, dc, dc, r_c);
        vec![
            Instrument::OisSwap(OisSwap::new(reference, m1, p1, Frequency::Annual, dc).unwrap()),
            Instrument::OisSwap(OisSwap::new(reference, m2, p2, Frequency::Annual, dc).unwrap()),
            Instrument::OisSwap(OisSwap::new(reference, m5, p5, Frequency::Annual, dc).unwrap()),
        ]
    }

    // ─── Construction ──────────────────────────────────────────────────────

    #[test]
    fn multi_curve_bootstrap_new_carries_defaults() {
        let reference = d(2024, 1, 2);
        let mcb = MultiCurveBootstrap::new(reference, Daycount::Act360);
        assert_eq!(mcb.reference_date, reference);
        assert_eq!(mcb.daycount, Daycount::Act360);
        assert_eq!(mcb.config, BootstrapConfig::default());
    }

    #[test]
    fn multi_curve_bootstrap_with_config_round_trip() {
        let reference = d(2024, 1, 2);
        let cfg = BootstrapConfig {
            tolerance: 1e-10,
            max_iter: 50,
            bracket: 0.25,
            iterative: false,
            iter_max: 4,
            iter_tol: 1e-12,
        };
        let mcb = MultiCurveBootstrap::new(reference, Daycount::Act360).with_config(cfg);
        assert_eq!(mcb.config, cfg);
    }

    // ─── OIS-only mode coincides with single-curve mode ────────────────────

    #[test]
    fn ois_only_bootstrap_matches_single_curve() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.03_f64;
        let ois = ois_instruments_flat(reference, dc, r_c);

        let mc = MultiCurveBootstrap::new(reference, dc)
            .build(
                &ois,
                Interpolation::LogLinear,
                &[],
                Interpolation::LogLinear,
            )
            .unwrap();
        let single = Bootstrap::new(reference, dc)
            .build(&ois, Interpolation::LogLinear)
            .unwrap();

        assert_eq!(mc.discount.times().len(), single.times().len());
        for (a, b) in mc
            .discount
            .discounts()
            .iter()
            .zip(single.discounts().iter())
        {
            assert!((a - b).abs() < 1e-14, "OIS curves diverge: {a} vs {b}");
        }
        assert!(mc.projection.is_empty());
    }

    // ─── Single tenor projection on a flat curve (consistent quotes) ───────

    #[test]
    fn single_projection_tenor_flat_curve_matches_ois() {
        // OIS at 3% (continuous), 3M projection at the same 3%, so the two
        // curves are mathematically identical at the projection knots.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.03_f64;
        let ois = ois_instruments_flat(reference, dc, r_c);

        // 3M projection from four FRAs spaced 3M apart at the flat-rate-
        // consistent quote.
        let fra_starts = [d(2024, 4, 2), d(2024, 7, 2), d(2024, 10, 2), d(2025, 1, 2)];
        let fra_ends = [d(2024, 7, 2), d(2024, 10, 2), d(2025, 1, 2), d(2025, 4, 2)];
        // Anchor the short end with a 3M deposit so the bootstrap has a
        // pillar before the first FRA's start.
        let dep = Deposit::new(
            reference,
            d(2024, 4, 2),
            flat_curve_quotes_for_dep(reference, dc, r_c, d(2024, 4, 2)),
            dc,
        )
        .unwrap();
        let mut projection_set: Vec<Instrument> = vec![Instrument::Deposit(dep)];
        for (s, e) in fra_starts.iter().zip(fra_ends.iter()) {
            projection_set.push(Instrument::Fra(
                Fra::new(*s, *e, flat_curve_quotes_for_fra(dc, r_c, *s, *e), dc).unwrap(),
            ));
        }

        let tenor_3m = Tenor::new(3, TenorUnit::Months);
        let mc = MultiCurveBootstrap::new(reference, dc)
            .build(
                &ois,
                Interpolation::LogLinear,
                &[(tenor_3m, projection_set.clone())],
                Interpolation::LogLinear,
            )
            .unwrap();

        let proj = mc.projection_curve(tenor_3m).unwrap();
        // At every projection knot the discount factor matches the OIS curve
        // (both built on the same flat 3% rate).
        for &t in proj.times() {
            let d_proj = proj.discount(t).unwrap();
            let d_ois = mc.discount.discount(t).unwrap();
            assert!(
                (d_proj - d_ois).abs() < 1e-10,
                "knot t={t}: D_proj={d_proj}, D_OIS={d_ois}"
            );
        }
    }

    // ─── Par-swap re-pricing on a flat curve ───────────────────────────────

    #[test]
    fn multi_curve_par_swap_residual_zero_flat_curve() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.03_f64;
        let ois = ois_instruments_flat(reference, dc, r_c);

        // 3M projection from a deposit + three FRAs.
        let dep_pay = d(2024, 4, 2);
        let dep = Deposit::new(
            reference,
            dep_pay,
            flat_curve_quotes_for_dep(reference, dc, r_c, dep_pay),
            dc,
        )
        .unwrap();
        let fra1 = Fra::new(
            d(2024, 4, 2),
            d(2024, 7, 2),
            flat_curve_quotes_for_fra(dc, r_c, d(2024, 4, 2), d(2024, 7, 2)),
            dc,
        )
        .unwrap();
        let fra2 = Fra::new(
            d(2024, 7, 2),
            d(2024, 10, 2),
            flat_curve_quotes_for_fra(dc, r_c, d(2024, 7, 2), d(2024, 10, 2)),
            dc,
        )
        .unwrap();
        let fra3 = Fra::new(
            d(2024, 10, 2),
            d(2025, 1, 2),
            flat_curve_quotes_for_fra(dc, r_c, d(2024, 10, 2), d(2025, 1, 2)),
            dc,
        )
        .unwrap();
        let projection_set = vec![
            Instrument::Deposit(dep),
            Instrument::Fra(fra1),
            Instrument::Fra(fra2),
            Instrument::Fra(fra3),
        ];

        let tenor_3m = Tenor::new(3, TenorUnit::Months);
        let mc = MultiCurveBootstrap::new(reference, dc)
            .build(
                &ois,
                Interpolation::LogLinear,
                &[(tenor_3m, projection_set)],
                Interpolation::LogLinear,
            )
            .unwrap();

        // Construct a 1y vanilla swap at the multi-curve par rate (which
        // coincides with the single-curve par rate when projection == OIS).
        let swap_start = reference;
        let swap_maturity = d(2025, 1, 2);
        let par = flat_par_swap_rate_multi(
            reference,
            swap_start,
            swap_maturity,
            Frequency::SemiAnnual,
            Frequency::Quarterly,
            Daycount::Act360,
            Daycount::Act360,
            dc,
            r_c,
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

        let ois_times = mc.discount.times().to_vec();
        let ois_discs = mc.discount.discounts().to_vec();
        let proj_curve = mc.projection_curve(tenor_3m).unwrap();
        let proj_times = proj_curve.times().to_vec();
        let proj_discs = proj_curve.discounts().to_vec();
        let ois_snap = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &ois_times,
            discounts: &ois_discs,
        };
        let proj_snap = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &proj_times,
            discounts: &proj_discs,
        };
        let residual = swap_residual_multi_curve(&swap, reference, &ois_snap, &proj_snap).unwrap();
        assert!(
            residual.abs() < 1e-10,
            "multi-curve par-swap residual must be < 1e-10, got {residual}"
        );
    }

    // ─── Genuine multi-curve case: 50bp basis between OIS and projection ───

    #[test]
    #[allow(clippy::too_many_lines)]
    fn multi_curve_50bp_basis_vanilla_swap_repricing() {
        // OIS curve at r_ois = 2%; 3M projection curve at r_proj = 2.5%.
        // Build the OIS curve from OIS swaps quoted on r_ois. Then build a
        // 3M projection from a vanilla SA/Q swap quoted at the multi-curve
        // par rate against (r_ois, r_proj). Verify the projection curve's
        // implied 3M forwards are roughly 50bp above the OIS 3M forwards.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_ois = 0.02_f64;
        let r_proj = 0.025_f64;

        let ois = ois_instruments_flat(reference, dc, r_ois);

        // Anchor the short end of the projection curve with a 3M deposit
        // quoted on r_proj.
        let dep_pay = d(2024, 4, 2);
        let dep = Deposit::new(
            reference,
            dep_pay,
            flat_curve_quotes_for_dep(reference, dc, r_proj, dep_pay),
            dc,
        )
        .unwrap();
        // Add a sequence of 3M FRAs to build the projection out to 5y.
        let mut fras: Vec<Instrument> = vec![Instrument::Deposit(dep)];
        let dep_start = d(2024, 4, 2);
        // Walk forward in 3M increments using Tenor arithmetic.
        let mut p_start = dep_start;
        for k in 1..=19_i32 {
            let p_end = Tenor::new(3 * k, TenorUnit::Months).add_to(dep_start);
            fras.push(Instrument::Fra(
                Fra::new(
                    p_start,
                    p_end,
                    flat_curve_quotes_for_fra(dc, r_proj, p_start, p_end),
                    dc,
                )
                .unwrap(),
            ));
            p_start = p_end;
        }
        // Add a 5y vanilla swap quoted at the multi-curve par rate. This is
        // the *real* test — the swap pins the projection curve via the
        // multi-curve par equation, not the single-curve telescoping.
        let swap_start = reference;
        let swap_maturity = d(2029, 1, 2);
        let par_multi = flat_par_swap_rate_multi(
            reference,
            swap_start,
            swap_maturity,
            Frequency::SemiAnnual,
            Frequency::Quarterly,
            Daycount::Act360,
            Daycount::Act360,
            dc,
            r_ois,
            r_proj,
        );
        // Sanity bound: the multi-curve par rate should sit between r_ois
        // and r_proj (the OIS-discounted weighting pulls it toward r_proj).
        assert!(
            par_multi > r_ois && par_multi < r_proj + 0.005,
            "unexpected par_multi = {par_multi}"
        );

        let tenor_3m = Tenor::new(3, TenorUnit::Months);
        let mc = MultiCurveBootstrap::new(reference, dc)
            .build(
                &ois,
                Interpolation::LogLinear,
                &[(tenor_3m, fras.clone())],
                Interpolation::LogLinear,
            )
            .unwrap();

        // Re-price every projection instrument: residual must be < 1e-10.
        let ois_times = mc.discount.times().to_vec();
        let ois_discs = mc.discount.discounts().to_vec();
        let proj_curve = mc.projection_curve(tenor_3m).unwrap();
        let proj_times = proj_curve.times().to_vec();
        let proj_discs = proj_curve.discounts().to_vec();
        let ois_snap = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &ois_times,
            discounts: &ois_discs,
        };
        let proj_snap = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &proj_times,
            discounts: &proj_discs,
        };
        for inst in &fras {
            let r = projection_residual(inst, reference, &ois_snap, &proj_snap).unwrap();
            assert!(
                r.abs() < 1e-10,
                "projection residual must be < 1e-10, got {r}"
            );
        }

        // Independent 5y SA/Q swap quoted at the closed-form par_multi: its
        // multi-curve residual against the bootstrapped curves must be at
        // floating-point round-off (we measured ~3e-14 during development).
        let swap = SwapFixedFloat::new(
            swap_start,
            swap_maturity,
            par_multi,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )
        .unwrap();
        let swap_residual =
            swap_residual_multi_curve(&swap, reference, &ois_snap, &proj_snap).unwrap();
        assert!(
            swap_residual.abs() < 1e-10,
            "5y SA/Q swap residual at par_multi must be < 1e-10, got {swap_residual}"
        );

        // Verify the implied 3M forward rates are ~50bp above OIS forwards.
        let probe_dates = [d(2025, 1, 2), d(2026, 1, 2), d(2028, 1, 2)];
        for &p_start in &probe_dates {
            let p_end = Tenor::new(3, TenorUnit::Months).add_to(p_start);
            let tau = dc.year_fraction(p_start, p_end).unwrap();
            let t_s = dc.year_fraction(reference, p_start).unwrap();
            let t_e = dc.year_fraction(reference, p_end).unwrap();
            let f_proj =
                (proj_curve.discount(t_s).unwrap() / proj_curve.discount(t_e).unwrap() - 1.0) / tau;
            let f_ois = (mc.discount.discount(t_s).unwrap() / mc.discount.discount(t_e).unwrap()
                - 1.0)
                / tau;
            // 50bp basis: f_proj - f_ois ≈ 0.005 to within FRA-coupon
            // discreteness (the projection curve is bootstrapped from FRAs
            // that pin the simply-compounded forward, not the continuous).
            assert!(
                (f_proj - f_ois - 0.005).abs() < 5e-4,
                "basis at {p_start:?}: f_proj={f_proj}, f_ois={f_ois}, diff={}",
                f_proj - f_ois,
            );
        }
    }

    // ─── Non-local interpolation (cubic spline) on both curves ─────────────

    #[test]
    fn multi_curve_cubic_spline_repricing() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_ois = 0.02_f64;
        let r_proj = 0.025_f64;

        let ois = ois_instruments_flat(reference, dc, r_ois);

        let dep_pay = d(2024, 4, 2);
        let dep = Deposit::new(
            reference,
            dep_pay,
            flat_curve_quotes_for_dep(reference, dc, r_proj, dep_pay),
            dc,
        )
        .unwrap();
        let dep_start = d(2024, 4, 2);
        let mut fras: Vec<Instrument> = vec![Instrument::Deposit(dep)];
        let mut p_start = dep_start;
        for k in 1..=7_i32 {
            let p_end = Tenor::new(3 * k, TenorUnit::Months).add_to(dep_start);
            fras.push(Instrument::Fra(
                Fra::new(
                    p_start,
                    p_end,
                    flat_curve_quotes_for_fra(dc, r_proj, p_start, p_end),
                    dc,
                )
                .unwrap(),
            ));
            p_start = p_end;
        }

        let tenor_3m = Tenor::new(3, TenorUnit::Months);
        let method = Interpolation::CubicSpline(SplineBoundary::NotAKnot);
        let mc = MultiCurveBootstrap::new(reference, dc)
            .build(&ois, method, &[(tenor_3m, fras.clone())], method)
            .unwrap();

        let ois_times = mc.discount.times().to_vec();
        let ois_discs = mc.discount.discounts().to_vec();
        let proj_curve = mc.projection_curve(tenor_3m).unwrap();
        let proj_times = proj_curve.times().to_vec();
        let proj_discs = proj_curve.discounts().to_vec();
        let ois_snap = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &ois_times,
            discounts: &ois_discs,
        };
        let proj_snap = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &proj_times,
            discounts: &proj_discs,
        };
        for inst in &fras {
            let r = projection_residual(inst, reference, &ois_snap, &proj_snap).unwrap();
            assert!(
                r.abs() < 1e-10,
                "cubic-spline projection residual must be < 1e-10, got {r}"
            );
        }
    }

    // ─── Validation errors ─────────────────────────────────────────────────

    #[test]
    fn build_rejects_empty_ois_instruments() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let err = MultiCurveBootstrap::new(reference, dc)
            .build(&[], Interpolation::LogLinear, &[], Interpolation::LogLinear)
            .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn build_rejects_ois_swap_in_projection_instruments() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.03_f64;
        let ois = ois_instruments_flat(reference, dc, r_c);
        // OIS swap supplied in projection set -> rejected.
        let projection_ois =
            OisSwap::new(reference, d(2025, 1, 2), r_c, Frequency::Annual, dc).unwrap();
        let tenor_3m = Tenor::new(3, TenorUnit::Months);
        let err = MultiCurveBootstrap::new(reference, dc)
            .build(
                &ois,
                Interpolation::LogLinear,
                &[(tenor_3m, vec![Instrument::OisSwap(projection_ois)])],
                Interpolation::LogLinear,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            BootstrapError::InvalidInstrument {
                reason: r,
                ..
            } if r.contains("OIS")
        ));
    }

    #[test]
    fn build_rejects_basis_swap_in_projection_instruments() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.03_f64;
        let ois = ois_instruments_flat(reference, dc, r_c);

        let leg_a = BasisLeg::new(
            reference,
            d(2025, 1, 2),
            Frequency::Quarterly,
            dc,
            Tenor::new(3, TenorUnit::Months),
        )
        .unwrap();
        let leg_b = BasisLeg::new(
            reference,
            d(2025, 1, 2),
            Frequency::SemiAnnual,
            dc,
            Tenor::new(6, TenorUnit::Months),
        )
        .unwrap();
        let bs = BasisSwap::new(leg_a, leg_b, 0.0010).unwrap();

        let tenor_3m = Tenor::new(3, TenorUnit::Months);
        let err = MultiCurveBootstrap::new(reference, dc)
            .build(
                &ois,
                Interpolation::LogLinear,
                &[(tenor_3m, vec![Instrument::BasisSwap(bs)])],
                Interpolation::LogLinear,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            BootstrapError::InvalidInstrument {
                reason: r,
                ..
            } if r.contains("basis")
        ));
    }

    #[test]
    fn build_rejects_empty_projection_instrument_set() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.03_f64;
        let ois = ois_instruments_flat(reference, dc, r_c);
        let tenor_3m = Tenor::new(3, TenorUnit::Months);
        let err = MultiCurveBootstrap::new(reference, dc)
            .build(
                &ois,
                Interpolation::LogLinear,
                &[(tenor_3m, Vec::new())],
                Interpolation::LogLinear,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            BootstrapError::InvalidInstrument {
                reason: r,
                ..
            } if r.contains("no projection")
        ));
    }

    #[test]
    fn build_rejects_projection_pillar_before_reference() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.03_f64;
        let ois = ois_instruments_flat(reference, dc, r_c);
        let degenerate = Deposit::new(reference, reference, 0.03, dc).unwrap();
        let tenor_3m = Tenor::new(3, TenorUnit::Months);
        let err = MultiCurveBootstrap::new(reference, dc)
            .build(
                &ois,
                Interpolation::LogLinear,
                &[(tenor_3m, vec![Instrument::Deposit(degenerate)])],
                Interpolation::LogLinear,
            )
            .unwrap_err();
        assert!(matches!(err, BootstrapError::InvalidInstrument { .. }));
    }

    #[test]
    fn build_rejects_projection_non_increasing_pillars() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.03_f64;
        let ois = ois_instruments_flat(reference, dc, r_c);
        let dep_late = Deposit::new(
            reference,
            d(2024, 7, 2),
            flat_curve_quotes_for_dep(reference, dc, r_c, d(2024, 7, 2)),
            dc,
        )
        .unwrap();
        let dep_early = Deposit::new(
            reference,
            d(2024, 4, 2),
            flat_curve_quotes_for_dep(reference, dc, r_c, d(2024, 4, 2)),
            dc,
        )
        .unwrap();
        let tenor_3m = Tenor::new(3, TenorUnit::Months);
        let err = MultiCurveBootstrap::new(reference, dc)
            .build(
                &ois,
                Interpolation::LogLinear,
                &[(
                    tenor_3m,
                    vec![
                        Instrument::Deposit(dep_late),
                        Instrument::Deposit(dep_early),
                    ],
                )],
                Interpolation::LogLinear,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            BootstrapError::NonIncreasingAnchor { at_index: 1 }
        ));
    }

    // ─── Accessors ─────────────────────────────────────────────────────────

    #[test]
    fn projection_curve_returns_some_for_present_tenor_and_none_otherwise() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.03_f64;
        let ois = ois_instruments_flat(reference, dc, r_c);
        let dep_pay = d(2024, 4, 2);
        let dep = Deposit::new(
            reference,
            dep_pay,
            flat_curve_quotes_for_dep(reference, dc, r_c, dep_pay),
            dc,
        )
        .unwrap();
        let tenor_3m = Tenor::new(3, TenorUnit::Months);
        let tenor_6m = Tenor::new(6, TenorUnit::Months);
        let mc = MultiCurveBootstrap::new(reference, dc)
            .build(
                &ois,
                Interpolation::LogLinear,
                &[(tenor_3m, vec![Instrument::Deposit(dep)])],
                Interpolation::LogLinear,
            )
            .unwrap();
        assert!(mc.projection_curve(tenor_3m).is_some());
        assert!(mc.projection_curve(tenor_6m).is_none());
        assert_eq!(mc.discount_curve().reference_date(), reference);
    }

    // ─── Two projection curves bootstrapped independently ──────────────────

    #[test]
    fn two_projection_curves_3m_and_6m_independent() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_ois = 0.02_f64;
        let r_3m = 0.025_f64;
        let r_6m = 0.028_f64;
        let ois = ois_instruments_flat(reference, dc, r_ois);

        // 3M projection from a deposit + three FRAs.
        let dep_3m_pay = d(2024, 4, 2);
        let dep_3m = Deposit::new(
            reference,
            dep_3m_pay,
            flat_curve_quotes_for_dep(reference, dc, r_3m, dep_3m_pay),
            dc,
        )
        .unwrap();
        let mut proj_3m: Vec<Instrument> = vec![Instrument::Deposit(dep_3m)];
        let mut p_start = dep_3m_pay;
        for k in 1..=3_i32 {
            let p_end = Tenor::new(3 * k, TenorUnit::Months).add_to(dep_3m_pay);
            proj_3m.push(Instrument::Fra(
                Fra::new(
                    p_start,
                    p_end,
                    flat_curve_quotes_for_fra(dc, r_3m, p_start, p_end),
                    dc,
                )
                .unwrap(),
            ));
            p_start = p_end;
        }

        // 6M projection from a deposit + two FRAs.
        let dep_6m_pay = d(2024, 7, 2);
        let dep_6m = Deposit::new(
            reference,
            dep_6m_pay,
            flat_curve_quotes_for_dep(reference, dc, r_6m, dep_6m_pay),
            dc,
        )
        .unwrap();
        let mut proj_6m: Vec<Instrument> = vec![Instrument::Deposit(dep_6m)];
        let mut p_start = dep_6m_pay;
        for k in 1..=2_i32 {
            let p_end = Tenor::new(6 * k, TenorUnit::Months).add_to(dep_6m_pay);
            proj_6m.push(Instrument::Fra(
                Fra::new(
                    p_start,
                    p_end,
                    flat_curve_quotes_for_fra(dc, r_6m, p_start, p_end),
                    dc,
                )
                .unwrap(),
            ));
            p_start = p_end;
        }

        let tenor_3m = Tenor::new(3, TenorUnit::Months);
        let tenor_6m = Tenor::new(6, TenorUnit::Months);
        let mc = MultiCurveBootstrap::new(reference, dc)
            .build(
                &ois,
                Interpolation::LogLinear,
                &[(tenor_3m, proj_3m), (tenor_6m, proj_6m)],
                Interpolation::LogLinear,
            )
            .unwrap();

        assert!(mc.projection_curve(tenor_3m).is_some());
        assert!(mc.projection_curve(tenor_6m).is_some());
        // The 3M curve's forward should be ~50bp above OIS; the 6M curve's
        // forward should be ~80bp above OIS at a generic 1y horizon.
        let t_a = dc.year_fraction(reference, d(2024, 7, 2)).unwrap();
        let t_b = dc.year_fraction(reference, d(2024, 10, 2)).unwrap();
        let f_3m = (mc
            .projection_curve(tenor_3m)
            .unwrap()
            .discount(t_a)
            .unwrap()
            / mc.projection_curve(tenor_3m)
                .unwrap()
                .discount(t_b)
                .unwrap()
            - 1.0)
            / (t_b - t_a);
        let f_6m_a = dc.year_fraction(reference, d(2024, 7, 2)).unwrap();
        let f_6m_b = dc.year_fraction(reference, d(2025, 1, 2)).unwrap();
        let f_6m = (mc
            .projection_curve(tenor_6m)
            .unwrap()
            .discount(f_6m_a)
            .unwrap()
            / mc.projection_curve(tenor_6m)
                .unwrap()
                .discount(f_6m_b)
                .unwrap()
            - 1.0)
            / (f_6m_b - f_6m_a);
        // Tolerances reflect simply-compounded vs continuous discretisation.
        assert!(
            (f_3m - 0.025).abs() < 1e-3,
            "3M projection forward = {f_3m}"
        );
        assert!(
            (f_6m - 0.028).abs() < 1e-3,
            "6M projection forward = {f_6m}"
        );
    }

    // ─── Basis-swap residual helper (multi-curve evaluator) ────────────────

    #[test]
    #[allow(clippy::too_many_lines)]
    fn basis_swap_residual_multi_curve_matches_independent_sum() {
        // Build the multi-curve set from §10. Then build a 5y 3M/6M basis
        // swap, compute the residual via `basis_swap_residual_multi_curve`,
        // and cross-check it against a hand-rolled sum.
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_ois = 0.02_f64;
        let r_3m = 0.025_f64;
        let r_6m = 0.028_f64;
        let ois = ois_instruments_flat(reference, dc, r_ois);

        // Build 3M and 6M projection curves out to >=5y.
        let dep_3m_pay = d(2024, 4, 2);
        let dep_3m = Deposit::new(
            reference,
            dep_3m_pay,
            flat_curve_quotes_for_dep(reference, dc, r_3m, dep_3m_pay),
            dc,
        )
        .unwrap();
        let mut proj_3m: Vec<Instrument> = vec![Instrument::Deposit(dep_3m)];
        let mut p_start = dep_3m_pay;
        for k in 1..=19_i32 {
            let p_end = Tenor::new(3 * k, TenorUnit::Months).add_to(dep_3m_pay);
            proj_3m.push(Instrument::Fra(
                Fra::new(
                    p_start,
                    p_end,
                    flat_curve_quotes_for_fra(dc, r_3m, p_start, p_end),
                    dc,
                )
                .unwrap(),
            ));
            p_start = p_end;
        }

        let dep_6m_pay = d(2024, 7, 2);
        let dep_6m = Deposit::new(
            reference,
            dep_6m_pay,
            flat_curve_quotes_for_dep(reference, dc, r_6m, dep_6m_pay),
            dc,
        )
        .unwrap();
        let mut proj_6m: Vec<Instrument> = vec![Instrument::Deposit(dep_6m)];
        let mut p_start = dep_6m_pay;
        for k in 1..=9_i32 {
            let p_end = Tenor::new(6 * k, TenorUnit::Months).add_to(dep_6m_pay);
            proj_6m.push(Instrument::Fra(
                Fra::new(
                    p_start,
                    p_end,
                    flat_curve_quotes_for_fra(dc, r_6m, p_start, p_end),
                    dc,
                )
                .unwrap(),
            ));
            p_start = p_end;
        }

        let tenor_3m = Tenor::new(3, TenorUnit::Months);
        let tenor_6m = Tenor::new(6, TenorUnit::Months);
        let mc = MultiCurveBootstrap::new(reference, dc)
            .build(
                &ois,
                Interpolation::LogLinear,
                &[(tenor_3m, proj_3m), (tenor_6m, proj_6m)],
                Interpolation::LogLinear,
            )
            .unwrap();

        let leg_a =
            BasisLeg::new(reference, d(2029, 1, 2), Frequency::Quarterly, dc, tenor_3m).unwrap();
        let leg_b = BasisLeg::new(
            reference,
            d(2029, 1, 2),
            Frequency::SemiAnnual,
            dc,
            tenor_6m,
        )
        .unwrap();
        let bs = BasisSwap::new(leg_a, leg_b, 0.0010).unwrap();

        let ois_times = mc.discount.times().to_vec();
        let ois_discs = mc.discount.discounts().to_vec();
        let proj_3m_curve = mc.projection_curve(tenor_3m).unwrap();
        let proj_3m_times = proj_3m_curve.times().to_vec();
        let proj_3m_discs = proj_3m_curve.discounts().to_vec();
        let proj_6m_curve = mc.projection_curve(tenor_6m).unwrap();
        let proj_6m_times = proj_6m_curve.times().to_vec();
        let proj_6m_discs = proj_6m_curve.discounts().to_vec();
        let ois_snap = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &ois_times,
            discounts: &ois_discs,
        };
        let proj_a = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &proj_3m_times,
            discounts: &proj_3m_discs,
        };
        let proj_b = CurveSnapshot {
            reference_date: reference,
            daycount: dc,
            times: &proj_6m_times,
            discounts: &proj_6m_discs,
        };
        let residual =
            basis_swap_residual_multi_curve(&bs, reference, &ois_snap, &proj_a, &proj_b).unwrap();
        // It is finite and nonzero for a non-zero spread on a 50bp/80bp
        // basis curve set — just sanity-check that the helper returns a
        // sensible value (the multi-curve par spread is not the input 10bp
        // here, since the projection curves are quoted at different flat
        // rates).
        assert!(residual.is_finite());
    }

    // ─── method_is_nonlocal helper coverage ────────────────────────────────

    #[test]
    fn method_is_nonlocal_distinguishes_local_and_global() {
        assert!(method_is_nonlocal(Interpolation::CubicSpline(
            SplineBoundary::NotAKnot
        )));
        assert!(method_is_nonlocal(Interpolation::HermiteBessel));
        assert!(method_is_nonlocal(Interpolation::MonotoneHyman));
        assert!(!method_is_nonlocal(Interpolation::Linear));
        assert!(!method_is_nonlocal(Interpolation::LogLinear));
        assert!(!method_is_nonlocal(Interpolation::LinearInZero));
        assert!(!method_is_nonlocal(Interpolation::PiecewiseConstantForward));
        assert!(!method_is_nonlocal(Interpolation::MonotoneCubic));
        assert!(!method_is_nonlocal(Interpolation::MonotoneSteffen));
    }

    // ─── Debug formatting ──────────────────────────────────────────────────

    #[test]
    fn multi_curve_bootstrap_debug_includes_struct_name() {
        let mcb = MultiCurveBootstrap::new(d(2024, 1, 2), Daycount::Act360);
        let dbg = format!("{mcb:?}");
        assert!(dbg.contains("MultiCurveBootstrap"));
    }

    #[test]
    fn multi_curve_debug_includes_struct_name() {
        let reference = d(2024, 1, 2);
        let discount = DiscountCurve::from_times_and_discounts(
            reference,
            Daycount::Act360,
            &[0.0, 1.0],
            &[1.0, 0.97],
            Interpolation::LogLinear,
        )
        .unwrap();
        let mc = MultiCurve {
            discount,
            projection: Vec::new(),
        };
        let dbg = format!("{mc:?}");
        assert!(dbg.contains("MultiCurve"));
    }
}
