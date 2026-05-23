// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Brent's root-finding method.
//!
//! Brent (1973) combines the guaranteed convergence of bisection with the
//! super-linear speed of inverse-quadratic interpolation and the secant
//! rule, falling back to bisection whenever the interpolant would step
//! outside the current bracket or fail a progress test. On any smooth
//! function with `f(a)` and `f(b)` of opposite sign, the method converges
//! to a root in `O(log((b-a)/xtol))` evaluations in the worst case.
//!
//! The method maintains four scalars `(a, b, c, d)` and their function
//! values:
//!
//! - `b` — the best estimate of the root (with `|f(b)| <= |f(a)|`).
//! - `a` — the other end of the current bracket (`f(a) * f(b) < 0`).
//! - `c` — the previous value of `b`.
//! - `d` — the value of `b` two iterations back (used for the
//!   "step-shrink" progress test of Brent §4.4).
//!
//! The single-letter names are kept because every primary source uses them
//! verbatim.
//!
//! # References
//!
//! - Brent, R. P., *Algorithms for Minimization Without Derivatives*,
//!   Prentice-Hall (1973), Chapter 4.

use super::MathError;

/// Configuration for [`brent_root`].
///
/// Defaults are `xtol = 1e-12`, `ftol = 1e-14`, `max_iter = 100`.
///
/// # Examples
///
/// ```
/// use regit_curves::math::brent::BrentConfig;
///
/// let cfg = BrentConfig::default();
/// assert!((cfg.xtol - 1e-12).abs() < 1e-18);
/// assert!((cfg.ftol - 1e-14).abs() < 1e-20);
/// assert_eq!(cfg.max_iter, 100);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrentConfig {
    /// Convergence tolerance on the bracket width `|b - a|`.
    pub xtol: f64,
    /// Convergence tolerance on the residual magnitude `|f(b)|`.
    pub ftol: f64,
    /// Iteration cap.
    pub max_iter: u32,
}

impl Default for BrentConfig {
    fn default() -> Self {
        Self {
            xtol: 1e-12,
            ftol: 1e-14,
            max_iter: 100,
        }
    }
}

/// Finds a root of `f` inside the bracket `[a, b]` by Brent's method.
///
/// `f(a)` and `f(b)` must straddle zero (their signs must differ — an exact
/// zero at either endpoint is also accepted). The method returns when the
/// bracket width drops below `config.xtol`, the residual magnitude drops
/// below `config.ftol`, or `f` evaluates exactly to zero.
///
/// # Errors
///
/// - [`MathError::BracketNotStraddling`] if `f(a)` and `f(b)` have the same
///   non-zero sign.
/// - [`MathError::NoConvergence`] if neither convergence test is met within
///   `config.max_iter` iterations.
///
/// # Examples
///
/// ```
/// use regit_curves::math::brent::{brent_root, BrentConfig};
///
/// // cos(x) - x has its fixed point at the Dottie number, ~0.7390851...
/// let root = brent_root(|x| x.cos() - x, 0.0, 1.0, BrentConfig::default()).unwrap();
/// assert!((root - 0.739_085_133_215_160_6).abs() < 1e-10);
/// ```
// Brent's method names its bracket and history points a, b, c, d, s as in
// the primary source (Brent 1973, Chapter 4); the single-char lint is noise
// for this canonical algorithm.
#[allow(clippy::many_single_char_names)]
pub fn brent_root<F>(mut f: F, a: f64, b: f64, config: BrentConfig) -> Result<f64, MathError>
where
    F: FnMut(f64) -> f64,
{
    let mut a = a;
    let mut b = b;
    let mut fa = f(a);
    let mut fb = f(b);

    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if fa * fb > 0.0 {
        return Err(MathError::BracketNotStraddling);
    }

    // Maintain |f(b)| <= |f(a)| so b is the better current estimate.
    if fa.abs() < fb.abs() {
        core::mem::swap(&mut a, &mut b);
        core::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = a;
    let mut mflag = true;

    for _ in 0..config.max_iter {
        if (b - a).abs() <= config.xtol || fb.abs() <= config.ftol || fb == 0.0 {
            return Ok(b);
        }

        // Trial step.
        let mut s = if (fa - fc).abs() > f64::EPSILON && (fb - fc).abs() > f64::EPSILON {
            // Inverse quadratic interpolation.
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            // Secant rule.
            b - fb * (b - a) / (fb - fa)
        };

        // Brent's progress test: bisect if the trial step is out of range
        // or stalls.
        let lo = (3.0 * a + b) / 4.0;
        let bound_lo = lo.min(b);
        let bound_hi = lo.max(b);
        let use_bisection = !(bound_lo..=bound_hi).contains(&s)
            || (mflag && (s - b).abs() >= (b - c).abs() / 2.0)
            || (!mflag && (s - b).abs() >= (c - d).abs() / 2.0)
            || (mflag && (b - c).abs() < config.xtol)
            || (!mflag && (c - d).abs() < config.xtol);

        if use_bisection {
            s = f64::midpoint(a, b);
            mflag = true;
        } else {
            mflag = false;
        }

        let fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if fa.abs() < fb.abs() {
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut fa, &mut fb);
        }
    }

    // Final tolerance check before declaring non-convergence.
    if (b - a).abs() <= config.xtol || fb.abs() <= config.ftol {
        return Ok(b);
    }
    Err(MathError::NoConvergence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brent_config_default_values() {
        let cfg = BrentConfig::default();
        assert!((cfg.xtol - 1e-12).abs() < 1e-18);
        assert!((cfg.ftol - 1e-14).abs() < 1e-20);
        assert_eq!(cfg.max_iter, 100);
    }

    #[test]
    fn brent_config_copy_eq() {
        let cfg = BrentConfig::default();
        let copy = cfg;
        assert_eq!(cfg, copy);
    }

    #[test]
    fn brent_finds_sqrt_two() {
        // Polynomial root: x^2 - 2 on [0, 2].
        let root = brent_root(|x| x * x - 2.0, 0.0, 2.0, BrentConfig::default()).unwrap();
        assert!((root - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn brent_finds_cubic_root() {
        // x^3 - x - 2 has a single real root near 1.5213797068.
        let root = brent_root(|x| x * x * x - x - 2.0, 1.0, 2.0, BrentConfig::default()).unwrap();
        assert!((root - 1.521_379_706_804_567_6).abs() < 1e-10);
    }

    #[test]
    fn brent_finds_dottie_number() {
        // The Dottie number is the unique real root of cos(x) - x.
        let root = brent_root(|x| x.cos() - x, 0.0, 1.0, BrentConfig::default()).unwrap();
        assert!((root - 0.739_085_133_215_160_6).abs() < 1e-10);
    }

    #[test]
    fn brent_endpoint_root_accepted() {
        // f(a) = 0 exactly: must return a.
        let root = brent_root(|x| x - 3.0, 3.0, 5.0, BrentConfig::default()).unwrap();
        assert!((root - 3.0).abs() < 1e-15);
    }

    #[test]
    fn brent_endpoint_root_b() {
        let root = brent_root(|x| x - 5.0, 3.0, 5.0, BrentConfig::default()).unwrap();
        assert!((root - 5.0).abs() < 1e-15);
    }

    #[test]
    fn brent_rejects_no_bracket() {
        // f(x) = x^2 + 1 is always positive; no bracket exists on [-1, 1].
        let err = brent_root(|x| x * x + 1.0, -1.0, 1.0, BrentConfig::default()).unwrap_err();
        assert!(matches!(err, MathError::BracketNotStraddling));
    }

    #[test]
    fn brent_finds_transcendental_root() {
        // exp(-x) - x has a root near 0.5671432904 (Omega constant).
        let root = brent_root(|x| (-x).exp() - x, 0.0, 1.0, BrentConfig::default()).unwrap();
        assert!((root - 0.567_143_290_409_783_8).abs() < 1e-10);
    }

    #[test]
    fn brent_respects_max_iter() {
        // Use a tiny iteration cap on a function that demands more iterations
        // to reach the default tolerance. We can't easily force this on a
        // smooth target — use a non-convex narrow root.
        let cfg = BrentConfig {
            xtol: 1e-30,
            ftol: 1e-30,
            max_iter: 1,
        };
        // One iteration of bisection won't reach xtol = 1e-30; either we
        // converge by accident or hit NoConvergence. Use a smooth target
        // where we definitely won't hit 1e-30 in 1 iteration.
        let res = brent_root(|x| x * x * x - 0.123, 0.0, 1.0, cfg);
        assert!(matches!(res, Err(MathError::NoConvergence)));
    }

    #[test]
    fn brent_uses_secant_when_three_points_coincide() {
        // Linear function: f(x) = x - 0.5; bracketed on [0, 1]. The
        // inverse-quadratic branch divides by (fa - fc) which is non-zero
        // initially, but after a few iterations the three points may align.
        // Either way, the algorithm must converge.
        let root = brent_root(|x| x - 0.5, 0.0, 1.0, BrentConfig::default()).unwrap();
        assert!((root - 0.5).abs() < 1e-12);
    }

    #[test]
    fn brent_swapped_endpoints() {
        // [b, a] with f(b) > 0, f(a) < 0 -> opposite ordering must still work.
        let root = brent_root(|x| x - 2.0, 5.0, 0.0, BrentConfig::default()).unwrap();
        assert!((root - 2.0).abs() < 1e-12);
    }

    #[test]
    fn brent_tight_ftol() {
        // Demand a very small residual; verify the returned point achieves it.
        let cfg = BrentConfig {
            xtol: 1e-15,
            ftol: 1e-15,
            max_iter: 200,
        };
        let f = |x: f64| (x - 3.7).powi(3);
        let root = brent_root(f, 0.0, 10.0, cfg).unwrap();
        assert!((root - 3.7).abs() < 1e-5);
    }

    #[test]
    fn brent_strictly_monotonic_function() {
        // exp(x) - 5 = 0 -> x = ln(5).
        let root = brent_root(|x| x.exp() - 5.0, 0.0, 5.0, BrentConfig::default()).unwrap();
        assert!((root - 5.0_f64.ln()).abs() < 1e-12);
    }
}
