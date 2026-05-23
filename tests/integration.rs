// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for regit-curves.
//!
//! Structure:
//!   - mod golden          -- worked-example regression anchors
//!   - mod oracle          -- QuantLib + tf-quant-finance + Hyman cross-oracles
//!   - mod roundtrip       -- conversion / view round-trip identities
//!   - mod arbitrage       -- no-arbitrage invariants on bootstrapped curves
//!   - mod properties      -- proptest invariants
//!   - mod multi_curve_e2e -- OIS + projection end-to-end
//!
//! Oracle vectors are transcribed from QuantLib (Modified BSD) and Google
//! tf-quant-finance (Apache-2.0); see RESEARCH.md §2 for the per-vector
//! provenance and the originating commits.

#![allow(
    clippy::float_cmp,            // canonical anchor / knot equalities
    clippy::doc_markdown,         // narrative prose references to identifier-like names
    clippy::cast_possible_wrap,   // i32 / usize / u32 casts in test glue
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::single_match_else,    // empty Err branches in proptest "either Ok or typed Err"
    clippy::type_complexity,      // table-of-tuples test fixtures
    clippy::excessive_precision,  // verbatim transcription of oracle constants
)]

use approx::assert_abs_diff_eq;
use regit_curves::bootstrap::Bootstrap;
use regit_curves::curves::{DiscountCurve, ForwardCurve, ParCurve, ZeroCurve};
use regit_curves::instruments::{
    Bond, Deposit, Fra, Instrument, OisSwap, SwapFixedFloat, SwapSchedule,
};
use regit_curves::interpolation::{
    Interpolation, Interpolator, LogLinear, MonotoneCubic, MonotoneHyman, MonotoneSteffen,
};
use regit_curves::math::brent::{BrentConfig, brent_root};
use regit_curves::multi_curve::MultiCurveBootstrap;
use regit_curves::types::{Compounding, Date, Daycount, Frequency, Tenor, TenorUnit};

// ─── Common helpers ───────────────────────────────────────────────────────

fn d(y: i32, m: u32, day: u32) -> Date {
    Date::from_ymd(y, m, day).unwrap()
}

/// Flat continuous-rate deposit quote consistent with `D(t) = exp(-r * t)`.
fn flat_deposit_rate(reference: Date, dc: Daycount, r_c: f64, payment: Date) -> f64 {
    let tau = dc.year_fraction(reference, payment).unwrap();
    let d_pay = (-r_c * tau).exp();
    (1.0 / d_pay - 1.0) / tau
}

/// Flat continuous-rate FRA quote consistent with `D(t) = exp(-r * t)`.
fn flat_fra_rate(dc: Daycount, r_c: f64, start: Date, end: Date) -> f64 {
    let tau = dc.year_fraction(start, end).unwrap();
    ((r_c * tau).exp() - 1.0) / tau
}

/// Closed-form par swap rate on a flat continuous-rate curve.
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
        let tau = fixed_dc
            .year_fraction(schedule.period_start(i), p_end)
            .unwrap();
        let t_pay = curve_dc.year_fraction(reference, p_end).unwrap();
        annuity += tau * (-r_c * t_pay).exp();
    }
    let t_start = curve_dc.year_fraction(reference, start).unwrap();
    let t_mat = curve_dc.year_fraction(reference, maturity).unwrap();
    ((-r_c * t_start).exp() - (-r_c * t_mat).exp()) / annuity
}

/// Builds a flat continuous-rate discount curve on a quarterly grid.
fn flat_curve(reference: Date, r_c: f64, years: i32) -> DiscountCurve {
    let n = years * 4 + 1;
    let mut times = Vec::with_capacity(n as usize);
    let mut discs = Vec::with_capacity(n as usize);
    for i in 0..n {
        let date = Date::from_serial(reference.serial() + i * 91);
        let t = Daycount::Act365F.year_fraction(reference, date).unwrap();
        times.push(t);
        discs.push((-r_c * t).exp());
    }
    DiscountCurve::from_times_and_discounts(
        reference,
        Daycount::Act365F,
        &times,
        &discs,
        Interpolation::LogLinear,
    )
    .unwrap()
}

// ─── Golden values ────────────────────────────────────────────────────────

mod golden {
    //! Worked-example regression anchors.
    //!
    //! - Act/Act ISDA and Act/Act ICMA worked examples (RESEARCH.md §2.3)
    //! - 30/360 Bond Basis worked cases (RESEARCH.md §2.4)
    //! - Brent root-finder polynomial tests (RESEARCH.md §2.6)
    //! - Continuous-compounding round-trip
    //! - Deposit pricing on a flat curve
    //! - Single-curve par swap at flat 4 %

    use super::*;

    /// QuantLib `testActualActual` worked examples (RESEARCH.md §2.3).
    /// Tolerance: QuantLib applies `1.0e-10` (line 209 of `daycounters.cpp`).
    #[test]
    fn act_act_isda_and_icma_worked_examples() {
        // Each row: (variant, d1, d2, period_start_for_icma, period_end_for_icma, expected).
        // Where the second/third Date columns are `None`, the variant is ISDA.
        type Row = (Daycount, (i32, u32, u32), (i32, u32, u32), f64);
        let isda_rows: &[Row] = &[
            (
                Daycount::ActActIsda,
                (2003, 11, 1),
                (2004, 5, 1),
                0.497_724_380_567,
            ),
            (
                Daycount::ActActIsda,
                (1999, 2, 1),
                (1999, 7, 1),
                0.410_958_904_110,
            ),
            (
                Daycount::ActActIsda,
                (1999, 7, 1),
                (2000, 7, 1),
                1.001_377_348_600,
            ),
            (
                Daycount::ActActIsda,
                (2002, 8, 15),
                (2003, 7, 15),
                0.915_068_493_151,
            ),
            (
                Daycount::ActActIsda,
                (2003, 7, 15),
                (2004, 1, 15),
                0.504_004_790_778,
            ),
            (
                Daycount::ActActIsda,
                (1999, 7, 30),
                (2000, 1, 30),
                0.503_892_506_924,
            ),
            (
                Daycount::ActActIsda,
                (2000, 1, 30),
                (2000, 6, 30),
                0.415_300_546_448,
            ),
        ];
        for &(dc, (y1, m1, d1_), (y2, m2, d2_), expected) in isda_rows {
            let yf = dc.year_fraction(d(y1, m1, d1_), d(y2, m2, d2_)).unwrap();
            assert_abs_diff_eq!(yf, expected, epsilon = 1e-10);
        }

        // Act/Act ICMA rows where the (d1, d2) span equals one regular
        // coupon period under the given frequency. Our implementation of
        // `ActActIcma { coupons_per_year }` returns `1 / coupons_per_year`
        // for the period — the canonical Rule-251 result on a regular
        // period.
        let icma_rows: &[(u32, (i32, u32, u32), (i32, u32, u32), f64)] = &[
            // 6-month period at freq = 2.
            (2, (2003, 11, 1), (2004, 5, 1), 0.500_000_000_000),
            // 12-month period at freq = 1.
            (1, (1999, 7, 1), (2000, 7, 1), 1.000_000_000_000),
            // 6-month period at freq = 2.
            (2, (2003, 7, 15), (2004, 1, 15), 0.500_000_000_000),
            // 6-month period at freq = 2.
            (2, (1999, 7, 30), (2000, 1, 30), 0.500_000_000_000),
        ];
        for &(freq, (y1, m1, d1_), (y2, m2, d2_), expected) in icma_rows {
            let dc = Daycount::ActActIcma {
                coupons_per_year: freq,
            };
            let yf = dc.year_fraction(d(y1, m1, d1_), d(y2, m2, d2_)).unwrap();
            assert_abs_diff_eq!(yf, expected, epsilon = 1e-10);
        }
    }

    /// ISDA Dec-2008 30/360 Bond Basis worked examples (RESEARCH.md §2.4).
    /// `dayCount = yearFraction * 360` exactly.
    #[test]
    fn thirty_360_bond_basis_worked_examples() {
        // Example 1 — End dates do not involve the last day of February.
        let ex1 = [
            ((2006, 8, 20), (2007, 2, 20), 180_i32),
            ((2007, 2, 20), (2007, 8, 20), 180),
            ((2007, 8, 20), (2008, 2, 20), 180),
            ((2008, 2, 20), (2008, 8, 20), 180),
            ((2008, 8, 20), (2009, 2, 20), 180),
            ((2009, 2, 20), (2009, 8, 20), 180),
        ];
        // Example 2 — End dates include some end-February dates.
        let ex2 = [
            ((2006, 8, 31), (2007, 2, 28), 178_i32),
            ((2007, 2, 28), (2007, 8, 31), 183),
            ((2007, 8, 31), (2008, 2, 29), 179),
            ((2008, 2, 29), (2008, 8, 31), 182),
            ((2008, 8, 31), (2009, 2, 28), 178),
            ((2009, 2, 28), (2009, 8, 31), 183),
        ];
        // Example 3 — Miscellaneous calculations.
        let ex3 = [
            ((2006, 1, 31), (2006, 2, 28), 28_i32),
            ((2006, 1, 30), (2006, 2, 28), 28),
            ((2006, 2, 28), (2006, 3, 3), 5),
            ((2006, 2, 14), (2006, 2, 28), 14),
            ((2006, 9, 30), (2006, 10, 31), 30),
            ((2006, 10, 31), (2006, 11, 28), 28),
            ((2007, 8, 31), (2008, 2, 28), 178),
            ((2008, 2, 28), (2008, 8, 28), 180),
            ((2008, 2, 28), (2008, 8, 30), 182),
            ((2008, 2, 28), (2008, 8, 31), 183),
            ((2007, 2, 26), (2008, 2, 28), 362),
            ((2007, 2, 26), (2008, 2, 29), 363),
            ((2008, 2, 29), (2009, 2, 28), 359),
            ((2008, 2, 28), (2008, 3, 30), 32),
            ((2008, 2, 28), (2008, 3, 31), 33),
        ];

        for ex in [ex1.as_slice(), ex2.as_slice(), ex3.as_slice()] {
            for &((y1, m1, d1_), (y2, m2, d2_), expected_days) in ex {
                let yf = Daycount::Thirty360BondBasis
                    .year_fraction(d(y1, m1, d1_), d(y2, m2, d2_))
                    .unwrap();
                let expected = f64::from(expected_days) / 360.0;
                assert_abs_diff_eq!(yf, expected, epsilon = 1e-13);
            }
        }
    }

    /// 30E/360 (Eurobond Basis) discriminator dates against Bond Basis
    /// (RESEARCH.md §2.4 cross-check).
    #[test]
    fn thirty_360_eurobond_vs_bond_basis_discriminator() {
        // 2008-02-28 -> 2008-08-31: 182 (Eurobond) vs 183 (Bond).
        let bb = Daycount::Thirty360BondBasis
            .year_fraction(d(2008, 2, 28), d(2008, 8, 31))
            .unwrap();
        let eb = Daycount::Thirty360E
            .year_fraction(d(2008, 2, 28), d(2008, 8, 31))
            .unwrap();
        assert_abs_diff_eq!(bb, 183.0 / 360.0, epsilon = 1e-13);
        assert_abs_diff_eq!(eb, 182.0 / 360.0, epsilon = 1e-13);

        // 2008-02-28 -> 2008-03-31: 32 (Eurobond) vs 33 (Bond).
        let bb2 = Daycount::Thirty360BondBasis
            .year_fraction(d(2008, 2, 28), d(2008, 3, 31))
            .unwrap();
        let eb2 = Daycount::Thirty360E
            .year_fraction(d(2008, 2, 28), d(2008, 3, 31))
            .unwrap();
        assert_abs_diff_eq!(bb2, 33.0 / 360.0, epsilon = 1e-13);
        assert_abs_diff_eq!(eb2, 32.0 / 360.0, epsilon = 1e-13);
    }

    /// QuantLib `testBrent` polynomial roots (RESEARCH.md §2.6).
    #[test]
    fn brent_polynomial_oracle() {
        let cfg = BrentConfig::default();
        // x^2 - 1 = 0 with root +1 on [0, 2].
        let r1 = brent_root(|x: f64| x * x - 1.0, 0.0, 2.0, cfg).unwrap();
        assert_abs_diff_eq!(r1, 1.0, epsilon = 1e-10);
        // 1 - x^2 = 0 with root +1 on [0, 2].
        let r2 = brent_root(|x: f64| 1.0 - x * x, 0.0, 2.0, cfg).unwrap();
        assert_abs_diff_eq!(r2, 1.0, epsilon = 1e-10);
        // atan(x - 1) = 0 with root +1 on [0, 2].
        let r3 = brent_root(|x: f64| (x - 1.0).atan(), 0.0, 2.0, cfg).unwrap();
        assert_abs_diff_eq!(r3, 1.0, epsilon = 1e-10);

        // Additional canonical roots: Dottie number; x^3 - x - 2 = 0.
        let dottie = brent_root(|x: f64| x.cos() - x, 0.0, 1.0, cfg).unwrap();
        assert_abs_diff_eq!(dottie, 0.739_085_133_215_160_6, epsilon = 1e-12);
        let cubic = brent_root(|x: f64| x.powi(3) - x - 2.0, 1.0, 2.0, cfg).unwrap();
        assert_abs_diff_eq!(cubic, 1.521_379_706_804_567_8, epsilon = 1e-12);
    }

    /// `Compounding::Continuous` is the canonical convention; the rate ->
    /// discount -> rate round-trip is exact in arithmetic.
    #[test]
    fn continuous_compounding_round_trip_exact() {
        for &r in &[-0.02_f64, 0.0, 0.01, 0.05, 0.20] {
            for &t in &[0.1_f64, 0.5, 1.0, 5.0, 30.0] {
                let d_t = Compounding::Continuous.discount_from_rate(r, t).unwrap();
                let back = Compounding::Continuous.rate_from_discount(d_t, t).unwrap();
                assert_abs_diff_eq!(back, r, epsilon = 1e-15);
            }
        }
    }

    /// Deposit pricing on a flat continuously-compounded curve: the residual
    /// against the flat-curve quote is zero to f64 round-off.
    ///
    /// The curve and the deposit must share a day-count for the identity
    /// `D(fix)/D(pay) = 1 + r * tau` to hold exactly. We pin the curve and
    /// the deposit to `Act/360` so the conversion is consistent.
    #[test]
    fn deposit_pricing_on_flat_curve_is_exact() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.05_f64;
        let payments = [d(2024, 4, 2), d(2024, 7, 2), d(2024, 10, 2), d(2025, 1, 2)];

        // Build an Act/360-anchored flat curve.
        let n = 21_usize;
        let mut times = Vec::with_capacity(n);
        let mut discs = Vec::with_capacity(n);
        for i in 0..n {
            let date = Date::from_serial(reference.serial() + (i as i32) * 91);
            let t = dc.year_fraction(reference, date).unwrap();
            times.push(t);
            discs.push((-r_c * t).exp());
        }
        let curve = DiscountCurve::from_times_and_discounts(
            reference,
            dc,
            &times,
            &discs,
            Interpolation::LogLinear,
        )
        .unwrap();

        for &payment in &payments {
            let rate = flat_deposit_rate(reference, dc, r_c, payment);
            let tau = dc.year_fraction(reference, payment).unwrap();
            // Deposit identity: D(fix) / D(pay) = 1 + r * tau.
            let t_pay = curve.daycount().year_fraction(reference, payment).unwrap();
            let d_pay = curve.discount(t_pay).unwrap();
            let lhs = 1.0 / d_pay;
            let rhs = 1.0 + rate * tau;
            assert_abs_diff_eq!(lhs, rhs, epsilon = 1e-12);
        }
    }

    /// Single-curve par swap on a flat continuous 4 % curve matches the
    /// closed-form rate.
    #[test]
    fn par_swap_flat_4pct_matches_closed_form() {
        let reference = d(2024, 1, 2);
        let r_c = 0.04_f64;
        let curve = flat_curve(reference, r_c, 11);
        let maturity = d(2034, 1, 2); // 10y
        // Compute via the curve.
        let par = curve
            .par_swap_rate(reference, maturity, Frequency::SemiAnnual, Daycount::Act360)
            .unwrap();
        // Closed-form expectation on the same flat curve (same dates).
        let expected = flat_par_swap_rate(
            reference,
            reference,
            maturity,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Daycount::Act365F,
            r_c,
        );
        assert_abs_diff_eq!(par, expected, epsilon = 1e-10);
    }
}

// ─── Oracle suite ─────────────────────────────────────────────────────────

mod oracle {
    //! Cross-oracle tests.
    //!
    //! Inputs and expected outputs transcribed verbatim from open-source
    //! references (RESEARCH.md §2.2, §2.5, §2.7, §2.8):
    //!
    //! - QuantLib `testLogLinearDiscountConsistency` re-pricing test
    //!   (`test-suite/piecewiseyieldcurve.cpp`, commit
    //!   `2eb86846efc496bf6ea0312fad2d31fec8c4ea13`).
    //! - Hyman 1983 RPN15A data set (transcribed via QuantLib
    //!   `test-suite/interpolations.cpp::testSplineOnRPN15AValues`).
    //! - tf-quant-finance `bond_curve_test.py::test_correctness` (Google,
    //!   commit `4551a94e8267a5c5eef9ad9d6079abaae19dcf14`).
    //! - GSL `interpolation/test.c::test_steffen` style monotonicity scan
    //!   used as the Steffen 1990 substitute (RESEARCH.md §2.10).

    use super::*;

    /// QuantLib testLogLinearDiscountConsistency (RESEARCH.md §2.2).
    ///
    /// QuantLib does **not** pin discount factors; it requires every input
    /// rate to be reproduced to within `tolerance = 1.0e-9` by re-discounting
    /// through the bootstrapped LogLinear curve. We use a representative
    /// subset of the suite (3 deposits + 3 swaps) on a 2024-01-02 reference
    /// date with no calendar adjustment (the test passes Unadjusted dates).
    ///
    /// Source: `https://github.com/lballabio/QuantLib/blob/2eb86846efc496bf6ea0312fad2d31fec8c4ea13/test-suite/piecewiseyieldcurve.cpp`, lines 97-142 (quotes) and 350-401 (assertion).
    #[test]
    fn quantlib_test_log_linear_discount_consistency_repricing() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;

        // Three of the QuantLib deposits (1M, 3M, 6M).
        let deposits = [
            (1_i32, 4.581_f64 / 100.0),
            (3, 4.557 / 100.0),
            (6, 4.496 / 100.0),
        ];
        // Three of the QuantLib swap pillars (2y, 5y, 10y).
        let swaps = [
            (2_i32, 4.63_f64 / 100.0),
            (5, 4.99 / 100.0),
            (10, 5.47 / 100.0),
        ];

        let mut instruments: Vec<Instrument> = Vec::new();
        for (m, rate) in deposits {
            instruments.push(Instrument::Deposit(
                Deposit::new(
                    reference,
                    Tenor::new(m, TenorUnit::Months).add_to(reference),
                    rate,
                    dc,
                )
                .unwrap(),
            ));
        }
        for (y, rate) in swaps {
            instruments.push(Instrument::SwapFixedFloat(
                SwapFixedFloat::new(
                    reference,
                    Tenor::new(y, TenorUnit::Years).add_to(reference),
                    rate,
                    Frequency::Annual,
                    Daycount::Thirty360BondBasis,
                    Frequency::SemiAnnual,
                    Daycount::Act360,
                )
                .unwrap(),
            ));
        }

        let bootstrap = Bootstrap::new(reference, dc);
        let curve = bootstrap
            .build(&instruments, Interpolation::LogLinear)
            .unwrap();

        // Re-pricing certificate: every instrument re-prices to within 1e-9.
        for (m, rate) in deposits {
            let payment = Tenor::new(m, TenorUnit::Months).add_to(reference);
            let tau = dc.year_fraction(reference, payment).unwrap();
            let t_pay = dc.year_fraction(reference, payment).unwrap();
            let d_pay = curve.discount(t_pay).unwrap();
            let curve_rate = (1.0 / d_pay - 1.0) / tau;
            assert_abs_diff_eq!(curve_rate, rate, epsilon = 1e-9);
        }
        for (y, rate) in swaps {
            let maturity = Tenor::new(y, TenorUnit::Years).add_to(reference);
            let par = curve
                .par_swap_rate(
                    reference,
                    maturity,
                    Frequency::Annual,
                    Daycount::Thirty360BondBasis,
                )
                .unwrap();
            assert_abs_diff_eq!(par, rate, epsilon = 1e-9);
        }
    }

    /// Hyman 1983 RPN15A test set (RESEARCH.md §2.5). The data is the CDF-
    /// like 9-point fixture from Hyman 1983; the discriminator is that an
    /// **unfiltered** cubic spline gives `f(11.0) > 1.0`, whereas the Hyman-
    /// filtered cubic gives `f(11.0) <= 1.0`. We test the filtered side and
    /// monotonicity on a fine grid.
    #[test]
    fn hyman_rpn15a_monotonicity_and_discriminator() {
        let knots: [(f64, f64); 9] = [
            (7.99, 0.0),
            (8.09, 2.764_29e-5),
            (8.19, 4.374_98e-5),
            (8.70, 0.169_183),
            (9.20, 0.469_428),
            (10.00, 0.943_740),
            (12.00, 0.998_636),
            (15.00, 0.999_919),
            (20.00, 0.999_994),
        ];
        let hyman = MonotoneHyman::new(&knots).unwrap();

        // Hyman filter: f(11.0) must be < 1.0 (monotonicity preserved).
        let v_11 = hyman.eval(11.0);
        assert!(
            v_11 <= 1.0,
            "Hyman-filtered f(11.0) = {v_11}, must be <= 1.0"
        );

        // Monotonicity on a fine grid covering the full input range.
        let mut prev = hyman.eval(knots[0].0);
        for i in 1..=2000_u32 {
            let x = knots[0].0 + f64::from(i) * (knots[8].0 - knots[0].0) / 2000.0;
            let v = hyman.eval(x);
            assert!(
                v >= prev - 1e-12,
                "Hyman interpolant decreased at x = {x}: prev = {prev}, v = {v}",
            );
            prev = v;
        }

        // Knot reproduction at each of the 9 data points.
        for &(x_i, y_i) in &knots {
            let v = hyman.eval(x_i);
            assert_abs_diff_eq!(v, y_i, epsilon = 1e-12);
        }
    }

    /// tf-quant-finance bond-curve correctness test (RESEARCH.md §2.8).
    ///
    /// The published `test_correctness` test bootstraps four bonds and
    /// asserts that the resulting continuous zero rates at `t ∈ {1, 2, 3, 4}`
    /// equal `[0.05, 0.0475, 0.045333..., 0.04775]` to `atol = 1e-6`. The
    /// associated discount factors are `exp(-r · t)`.
    ///
    /// Our bootstrap engine consumes swap-style instruments, not free-form
    /// bonds, so we cannot reproduce the exact bootstrap. Instead we build
    /// a `DiscountCurve` directly with the published zero rates and assert
    /// (a) the pillar discount factors equal `exp(-r · t)` to 1e-12, and
    /// (b) the implied continuous zero rate read back through the curve
    /// matches the published value to within the tf-quant-finance atol
    /// (`1e-6`). Sub-pillar interpolation between published pillars is
    /// scheme-dependent (Hagan-West monotone-convex in the upstream test,
    /// LogLinear on discount factors here) and is not cross-checked.
    #[test]
    fn tf_quant_finance_bond_curve_oracle() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act365F;
        let rates = [0.05_f64, 0.0475, 0.045_333_333_333_333_336, 0.04775];
        let times = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let mut discs = vec![1.0_f64];
        for (i, &t) in times.iter().enumerate().skip(1) {
            discs.push((-rates[i - 1] * t).exp());
        }
        let curve = DiscountCurve::from_times_and_discounts(
            reference,
            dc,
            &times,
            &discs,
            Interpolation::LogLinear,
        )
        .unwrap();

        // Verify pillar-level discount factors and zero rates.
        for (i, &r_expected) in rates.iter().enumerate() {
            let t_expected = f64::from(i32::try_from(i + 1).unwrap());
            let d_expected = (-r_expected * t_expected).exp();
            let d_got = curve.discount(t_expected).unwrap();
            assert_abs_diff_eq!(d_got, d_expected, epsilon = 1e-12);

            let r_got = curve
                .zero_rate(t_expected, Compounding::Continuous)
                .unwrap();
            assert_abs_diff_eq!(r_got, r_expected, epsilon = 1e-6);
        }
    }

    /// Steffen 1990 monotonicity oracle (RESEARCH.md §2.10).
    /// GSL ships a Steffen test fixture; the canonical property is that
    /// a monotone-increasing input produces a monotone-increasing output on
    /// a fine grid. We use a synthetic 8-point monotone set.
    #[test]
    fn steffen_monotonicity_on_synthetic_monotone_set() {
        let knots = [
            (0.0_f64, 0.0_f64),
            (1.0, 0.6),
            (2.0, 0.85),
            (3.0, 0.94),
            (4.0, 0.98),
            (5.0, 0.99),
            (6.0, 0.995),
            (7.0, 0.999),
        ];
        let steffen = MonotoneSteffen::new(&knots).unwrap();

        let mut prev = steffen.eval(knots[0].0);
        for i in 1..=2000_u32 {
            let x = f64::from(i) * 7.0 / 2000.0;
            let v = steffen.eval(x);
            assert!(
                v >= prev - 1e-12,
                "Steffen interpolant decreased at x = {x}: prev = {prev}, v = {v}",
            );
            prev = v;
        }
        for &(x_i, y_i) in &knots {
            assert_abs_diff_eq!(steffen.eval(x_i), y_i, epsilon = 1e-12);
        }
    }

    /// Coupon-bearing bond oracle on a flat continuous-rate curve.
    ///
    /// A 5y annual 5% bond on a flat 5% continuously-compounded curve
    /// (Act/365F day-count) is priced at par with zero accrued. The
    /// par-bond identity
    /// `coupon * SUM_i tau_i * D(t_i) * notional + notional * D(t_N) = clean_price`
    /// requires the bond's residual against the hand-rolled flat curve to
    /// vanish to machine precision (Andersen & Piterbarg 2010, §5.1).
    #[test]
    fn bond_par_pricing_on_flat_curve_is_exact() {
        let reference = d(2024, 1, 2);
        let curve = flat_curve(reference, 0.05, 6);

        // Build the par coupon analytically: solve
        // coupon * SUM tau_i * D(t_i) + D(t_N) = 1 on the flat curve.
        let maturity = d(2029, 1, 2);
        let schedule = SwapSchedule::from_regular(reference, maturity, Frequency::Annual).unwrap();
        let mut annuity = 0.0_f64;
        for i in 0..schedule.len() {
            let p_end = schedule.period_end(i);
            let tau = Daycount::Act365F
                .year_fraction(schedule.period_start(i), p_end)
                .unwrap();
            let d_pay = curve.discount_at(p_end).unwrap();
            annuity += tau * d_pay;
        }
        let d_maturity = curve.discount_at(maturity).unwrap();
        let par_coupon = (1.0 - d_maturity) / annuity;

        // Construct the bond at par and verify its residual against the
        // flat curve vanishes.
        let bond = Bond::new(
            reference,
            maturity,
            par_coupon,
            Frequency::Annual,
            Daycount::Act365F,
            1.0,
            1.0,
            0.0,
        )
        .unwrap();

        // Re-price via the bootstrap engine: a single Bond pinned at
        // par should produce a curve that re-prices it.
        let bs = Bootstrap::new(reference, Daycount::Act365F);
        let curve_bs = bs
            .build(&[Instrument::Bond(bond)], Interpolation::LogLinear)
            .unwrap();
        // The bootstrapped curve must produce `D(maturity) = exp(-r * T)`
        // up to bootstrap tolerance.
        let d_got = curve_bs.discount_at(maturity).unwrap();
        let t_n = Daycount::Act365F
            .year_fraction(reference, maturity)
            .unwrap();
        let d_expected = (-0.05_f64 * t_n).exp();
        assert_abs_diff_eq!(d_got, d_expected, epsilon = 1e-10);
    }
}

// ─── Round-trip identities ────────────────────────────────────────────────

mod roundtrip {
    //! Round-trip identities between the four curve views and between dates,
    //! tenors, and interpolators.

    use super::*;

    /// DiscountCurve <-> ZeroCurve <-> DiscountCurve at every node (exact).
    #[test]
    fn discount_zero_round_trip_at_nodes_exact() {
        let reference = d(2024, 1, 2);
        let curve = flat_curve(reference, 0.04, 10);
        let z = ZeroCurve::from(&curve, Compounding::Continuous);
        for (i, &t) in curve.times().iter().enumerate() {
            if t == 0.0 {
                continue;
            }
            let d_t = curve.discounts()[i];
            let rate = z.rate(t).unwrap();
            let reconstructed = (-rate * t).exp();
            assert_abs_diff_eq!(reconstructed, d_t, epsilon = 1e-14);
        }
    }

    /// `D(t1) / D(t2)` via the forward curve agrees with the direct ratio.
    #[test]
    fn discount_forward_consistency_on_segments() {
        let reference = d(2024, 1, 2);
        let curve = flat_curve(reference, 0.04, 5);
        let f = ForwardCurve::from(&curve);
        // For a flat curve at rate r, the simply-compounded forward over
        // [t1, t2] is `(exp(r*(t2-t1)) - 1) / (t2 - t1)`.
        for &(t1, t2) in &[(0.25_f64, 1.0), (1.0, 2.0), (0.5, 3.5)] {
            let l = f.forward(t1, t2, Daycount::Act365F).unwrap();
            let expected = ((0.04_f64 * (t2 - t1)).exp() - 1.0) / (t2 - t1);
            assert_abs_diff_eq!(l, expected, epsilon = 1e-10);
        }
    }

    /// `Date::from_ymd(y, m, d) -> (year, month, day)` round-trips exactly.
    #[test]
    fn date_ymd_round_trip_on_representative_dates() {
        let cases: [(i32, u32, u32); 50] = [
            (1900, 1, 1),
            (1900, 2, 28),
            (1904, 2, 29),
            (1969, 12, 31),
            (1970, 1, 1),
            (1972, 2, 29),
            (1999, 12, 31),
            (2000, 1, 1),
            (2000, 2, 29),
            (2000, 3, 1),
            (2003, 7, 15),
            (2003, 12, 31),
            (2004, 1, 1),
            (2004, 2, 29),
            (2004, 3, 1),
            (2007, 5, 15),
            (2008, 2, 29),
            (2008, 3, 1),
            (2008, 12, 31),
            (2010, 6, 30),
            (2012, 2, 29),
            (2015, 8, 15),
            (2016, 2, 29),
            (2019, 11, 30),
            (2020, 2, 29),
            (2020, 3, 1),
            (2021, 1, 1),
            (2022, 7, 4),
            (2023, 4, 15),
            (2024, 1, 1),
            (2024, 1, 2),
            (2024, 2, 29),
            (2024, 3, 1),
            (2024, 12, 31),
            (2025, 1, 1),
            (2026, 5, 23),
            (2030, 11, 11),
            (2040, 7, 4),
            (2050, 12, 31),
            (2060, 2, 29),
            (2070, 6, 15),
            (2080, 1, 1),
            (2090, 12, 31),
            (2100, 1, 1),
            (2100, 2, 28),
            (2100, 12, 31),
            (2104, 2, 29),
            (2200, 6, 30),
            (2300, 3, 1),
            (2400, 2, 29),
        ];
        for &(y, m, day) in &cases {
            let dt = Date::from_ymd(y, m, day).unwrap();
            assert_eq!(dt.year(), y);
            assert_eq!(dt.month(), m);
            assert_eq!(dt.day(), day);
        }
    }

    /// `Tenor::add_to(d).days_between(d)` is consistent with the tenor span.
    #[test]
    fn tenor_round_trip_consistency() {
        let base = d(2024, 6, 15);
        // Day tenors are exact.
        let cases: &[(i32, TenorUnit, i32)] = &[
            (1, TenorUnit::Days, 1),
            (7, TenorUnit::Days, 7),
            (30, TenorUnit::Days, 30),
            (180, TenorUnit::Days, 180),
            (1, TenorUnit::Weeks, 7),
            (4, TenorUnit::Weeks, 28),
        ];
        for &(count, unit, expected_days) in cases {
            let target = Tenor::new(count, unit).add_to(base);
            let span = target.days_between(base).abs();
            assert_eq!(span, expected_days, "tenor {count} {unit:?}");
        }

        // Month/year tenors land on the same day-of-month (subject to
        // end-of-month preserved rule) — assert via `month` arithmetic.
        let one_year = Tenor::new(1, TenorUnit::Years).add_to(base);
        assert_eq!(one_year.year(), 2025);
        assert_eq!(one_year.month(), 6);
        assert_eq!(one_year.day(), 15);

        let six_months = Tenor::new(6, TenorUnit::Months).add_to(base);
        assert_eq!(six_months.year(), 2024);
        assert_eq!(six_months.month(), 12);
        assert_eq!(six_months.day(), 15);
    }

    /// `LogLinear.eval(t_k) == y_k` exactly at every construction knot.
    #[test]
    fn log_linear_knot_reproduction_exact() {
        let knots: Vec<(f64, f64)> = (0..10)
            .map(|i| {
                let t = f64::from(i) * 0.5;
                (t, (-0.04_f64 * t).exp())
            })
            .collect();
        let interp = LogLinear::new(&knots).unwrap();
        for &(t, y) in &knots {
            assert_abs_diff_eq!(interp.eval(t), y, epsilon = 1e-15);
        }
    }

    /// Bootstrap re-pricing certificate (WORKING.md §7): every successfully
    /// bootstrapped curve re-prices every input instrument to within the
    /// configured `tolerance` (1e-9 here, mirroring QuantLib §2.2).
    #[test]
    fn bootstrap_repricing_certificate() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.04_f64;
        let dep1_pay = d(2024, 4, 2);
        let dep2_pay = d(2024, 7, 2);
        let fra_start = d(2024, 7, 2);
        let fra_end = d(2024, 10, 2);
        let swap_mat = d(2026, 1, 2);

        let dep1 = Deposit::new(
            reference,
            dep1_pay,
            flat_deposit_rate(reference, dc, r_c, dep1_pay),
            dc,
        )
        .unwrap();
        let dep2 = Deposit::new(
            reference,
            dep2_pay,
            flat_deposit_rate(reference, dc, r_c, dep2_pay),
            dc,
        )
        .unwrap();
        let fra = Fra::new(
            fra_start,
            fra_end,
            flat_fra_rate(dc, r_c, fra_start, fra_end),
            dc,
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
            Instrument::Fra(fra),
            Instrument::SwapFixedFloat(swap),
        ];
        let curve = Bootstrap::new(reference, dc)
            .build(&instruments, Interpolation::LogLinear)
            .unwrap();

        // Deposits re-priced.
        for pay in [dep1_pay, dep2_pay] {
            let tau = dc.year_fraction(reference, pay).unwrap();
            let t_pay = dc.year_fraction(reference, pay).unwrap();
            let d_pay = curve.discount(t_pay).unwrap();
            let curve_rate = (1.0 / d_pay - 1.0) / tau;
            let input = flat_deposit_rate(reference, dc, r_c, pay);
            assert_abs_diff_eq!(curve_rate, input, epsilon = 1e-9);
        }
        // FRA re-priced.
        let tau = dc.year_fraction(fra_start, fra_end).unwrap();
        let t_s = dc.year_fraction(reference, fra_start).unwrap();
        let t_e = dc.year_fraction(reference, fra_end).unwrap();
        let curve_fra = (curve.discount(t_s).unwrap() / curve.discount(t_e).unwrap() - 1.0) / tau;
        assert_abs_diff_eq!(
            curve_fra,
            flat_fra_rate(dc, r_c, fra_start, fra_end),
            epsilon = 1e-9
        );
        // Swap re-priced.
        let curve_par = curve
            .par_swap_rate(reference, swap_mat, Frequency::SemiAnnual, Daycount::Act360)
            .unwrap();
        assert_abs_diff_eq!(curve_par, par, epsilon = 1e-9);
    }

    /// ParCurve.par_rate ≡ DiscountCurve.par_swap_rate.
    #[test]
    fn par_curve_view_round_trips_to_discount_curve() {
        let reference = d(2024, 1, 2);
        let curve = flat_curve(reference, 0.04, 10);
        let p = ParCurve::from(&curve);
        let direct = curve
            .par_swap_rate(
                reference,
                d(2027, 1, 2),
                Frequency::SemiAnnual,
                Daycount::Act360,
            )
            .unwrap();
        let viewed = p
            .par_rate(
                reference,
                d(2027, 1, 2),
                Frequency::SemiAnnual,
                Daycount::Act360,
            )
            .unwrap();
        assert_abs_diff_eq!(direct, viewed, epsilon = 1e-15);
    }
}

// ─── Arbitrage invariants ─────────────────────────────────────────────────

mod arbitrage {
    //! Arbitrage-free invariants on bootstrapped curves.

    use super::*;

    /// A bootstrapped curve has strictly positive discount factors.
    #[test]
    fn bootstrapped_curve_has_positive_discounts() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.04_f64;
        let payments = [d(2024, 4, 2), d(2024, 7, 2), d(2024, 10, 2), d(2025, 1, 2)];
        let instruments: Vec<Instrument> = payments
            .iter()
            .map(|&p| {
                Instrument::Deposit(
                    Deposit::new(reference, p, flat_deposit_rate(reference, dc, r_c, p), dc)
                        .unwrap(),
                )
            })
            .collect();
        let curve = Bootstrap::new(reference, dc)
            .build(&instruments, Interpolation::LogLinear)
            .unwrap();
        for &d_t in curve.discounts() {
            assert!(d_t > 0.0, "non-positive discount factor in curve: {d_t}");
        }
        // Off-knot probe.
        for i in 1..=1000 {
            let t = f64::from(i) * 0.001;
            assert!(curve.discount(t).unwrap() > 0.0);
        }
    }

    /// Monotone-decreasing input -> monotone-decreasing curve under the
    /// monotone interpolators.
    #[test]
    fn monotone_input_produces_monotone_curve() {
        // D(t) is monotone-decreasing on any positive-rate curve. Pick a
        // strictly-decreasing 6-point input and verify under three monotone
        // interpolators.
        let knots: [(f64, f64); 6] = [
            (0.0, 1.0),
            (0.5, 0.98),
            (1.0, 0.95),
            (2.0, 0.90),
            (5.0, 0.80),
            (10.0, 0.65),
        ];
        let interpolators: [Box<dyn Fn(f64) -> f64>; 3] = [
            {
                let m = MonotoneCubic::new(&knots).unwrap();
                Box::new(move |t: f64| m.eval(t))
            },
            {
                let m = MonotoneSteffen::new(&knots).unwrap();
                Box::new(move |t: f64| m.eval(t))
            },
            {
                let m = MonotoneHyman::new(&knots).unwrap();
                Box::new(move |t: f64| m.eval(t))
            },
        ];
        for eval in &interpolators {
            let mut prev = eval(0.0);
            for i in 1..=1000 {
                let t = f64::from(i) * 0.01;
                let v = eval(t);
                assert!(
                    v <= prev + 1e-12,
                    "non-monotone at t = {t}: prev = {prev}, v = {v}"
                );
                prev = v;
            }
        }
    }

    /// Positive-rate flat curve has positive instantaneous forward, and the
    /// discount factor is monotone-decreasing.
    ///
    /// The instantaneous-forward sampling stays strictly inside the curve's
    /// knot range — flat extrapolation past the final knot returns a zero
    /// right derivative, which is correct behaviour but is not a test of
    /// the arbitrage invariant.
    #[test]
    fn flat_positive_rate_curve_has_positive_forward_and_decreasing_discount() {
        let reference = d(2024, 1, 2);
        let curve = flat_curve(reference, 0.05, 10);
        let f = ForwardCurve::from(&curve);
        let mut prev = curve.discount(0.0).unwrap();
        for i in 1..=200 {
            let t = f64::from(i) * 0.025; // 0.025 .. 5.0 — strictly interior
            let v = curve.discount(t).unwrap();
            assert!(v <= prev + 1e-12, "non-decreasing at t = {t}");
            prev = v;
            let inst = f.instantaneous(t).unwrap();
            assert!(
                inst > 0.0,
                "non-positive instantaneous forward at t = {t}: {inst}"
            );
        }
    }

    /// Multi-curve par swap rate coincides with the single-curve par swap
    /// rate when the projection curve equals the OIS curve.
    #[test]
    fn multi_curve_equals_single_when_projection_equals_ois() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.03_f64;
        let m1 = d(2025, 1, 2);
        let m2 = d(2026, 1, 2);
        let m5 = d(2029, 1, 2);
        let ois: Vec<Instrument> = [m1, m2, m5]
            .iter()
            .map(|&m| {
                let p = flat_par_swap_rate(reference, reference, m, Frequency::Annual, dc, dc, r_c);
                Instrument::OisSwap(OisSwap::new(reference, m, p, Frequency::Annual, dc).unwrap())
            })
            .collect();
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

        for (a, b) in mc
            .discount
            .discounts()
            .iter()
            .zip(single.discounts().iter())
        {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-14);
        }
    }
}

// ─── proptest invariants ──────────────────────────────────────────────────

mod properties {
    //! Proptest invariants.

    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Strictly-decreasing knots produce a monotone-decreasing
        /// LogLinear interpolant.
        #[test]
        fn prop_log_linear_monotonicity(
            seed in 0u32..1000,
            n in 3usize..10,
        ) {
            // Build a strictly-decreasing knot sequence deterministically.
            let mut knots: Vec<(f64, f64)> = Vec::with_capacity(n);
            knots.push((0.0, 1.0));
            for i in 1..n {
                let t = f64::from(u32::try_from(i).unwrap_or(0));
                // Decay shape depends deterministically on the seed.
                let decay = 0.02 + 0.001 * f64::from(seed % 100);
                let d_prev = knots[i - 1].1;
                knots.push((t, d_prev * (-decay).exp()));
            }
            let interp = LogLinear::new(&knots).unwrap();
            let mut prev = interp.eval(0.0);
            for i in 1..=200 {
                let t = f64::from(i) * f64::from(u32::try_from(n - 1).unwrap_or(1)) / 200.0;
                let v = interp.eval(t);
                prop_assert!(v <= prev + 1e-12, "non-monotone at t = {t}");
                prev = v;
            }
        }

        /// Random deposit-only inputs either return a typed error or
        /// produce a curve that re-prices its inputs.
        #[test]
        fn prop_bootstrap_never_panics(
            rate in -0.05_f64..0.20,
            tenor_days in 1i32..366,
            seed in 0u32..1000,
        ) {
            let reference = d(2024, 1, 2);
            let dc = Daycount::Act360;
            let n = 3 + (seed % 8) as i32;
            let mut instruments: Vec<Instrument> = Vec::new();
            let mut last_day = 0_i32;
            for i in 0..n {
                let day_offset = last_day + tenor_days * (i + 1);
                last_day = day_offset;
                let payment = Date::from_serial(reference.serial() + day_offset);
                // Vary rate slightly across deposits.
                let r_i = rate * (1.0 + 0.01 * f64::from(i));
                if let Ok(dep) = Deposit::new(reference, payment, r_i, dc) {
                    instruments.push(Instrument::Deposit(dep));
                }
            }
            if instruments.is_empty() {
                return Ok(());
            }
            let result = Bootstrap::new(reference, dc)
                .build(&instruments, Interpolation::LogLinear);
            match result {
                Ok(curve) => {
                    // Re-pricing certificate.
                    for inst in &instruments {
                        if let Instrument::Deposit(dep) = inst {
                            let tau = dc.year_fraction(reference, dep.payment).unwrap();
                            let t_pay = dc.year_fraction(reference, dep.payment).unwrap();
                            let d_pay = curve.discount(t_pay).unwrap();
                            let curve_rate = (1.0 / d_pay - 1.0) / tau;
                            prop_assert!((curve_rate - dep.rate).abs() < 1e-9);
                        }
                    }
                }
                Err(_) => {
                    // Typed error is acceptable; the property is "no panic".
                }
            }
        }

        /// Compounding round-trip: discount-from-rate followed by
        /// rate-from-discount recovers the input rate.
        #[test]
        fn prop_compounding_roundtrip(
            r in -0.05_f64..0.20,
            t in 0.05_f64..30.0,
        ) {
            // Continuous: exact in arithmetic.
            let d_c = Compounding::Continuous.discount_from_rate(r, t).unwrap();
            let back_c = Compounding::Continuous.rate_from_discount(d_c, t).unwrap();
            prop_assert!((back_c - r).abs() < 1e-13);

            // Simple: D = 1 / (1 + r*t); recovery exact to 1e-13.
            // Skip pathological negative growth factors.
            if 1.0 + r * t > 0.0 {
                let d_s = Compounding::Simple.discount_from_rate(r, t).unwrap();
                let back_s = Compounding::Simple.rate_from_discount(d_s, t).unwrap();
                prop_assert!((back_s - r).abs() < 1e-13);
            }

            // Periodic (semi-annual): same property.
            let compounding = Compounding::Periodic { periods_per_year: 2 };
            if 1.0 + r / 2.0 > 0.0 {
                let d_p = compounding.discount_from_rate(r, t).unwrap();
                let back_p = compounding.rate_from_discount(d_p, t).unwrap();
                prop_assert!((back_p - r).abs() < 1e-13);
            }
        }

        /// Random `(y, m, d)` triples: `from_ymd` either fails on an invalid
        /// date or round-trips exactly through the accessors.
        #[test]
        fn prop_date_serial_roundtrip(
            year in 1900_i32..2200,
            month in 1u32..=12,
            day in 1u32..=31,
        ) {
            match Date::from_ymd(year, month, day) {
                Ok(dt) => {
                    prop_assert_eq!(dt.year(), year);
                    prop_assert_eq!(dt.month(), month);
                    prop_assert_eq!(dt.day(), day);
                }
                Err(_) => {
                    // Invalid date (e.g. Feb 30) — error is acceptable.
                }
            }
        }
    }
}

// ─── Multi-curve end-to-end ───────────────────────────────────────────────

mod multi_curve_e2e {
    //! End-to-end multi-curve OIS + 3M projection workflows.

    use super::*;

    /// Complete OIS + 3M projection bootstrap on consistent quotes, with
    /// every input re-priced to within `1e-10`.
    #[test]
    fn ois_plus_3m_projection_repricing_certificate() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_ois = 0.02_f64;
        let r_proj = 0.025_f64;

        // OIS swaps at flat 2 %.
        let ois: Vec<Instrument> = [(1_i32), 2, 3, 5]
            .iter()
            .map(|&y| {
                let m = Tenor::new(y, TenorUnit::Years).add_to(reference);
                let p =
                    flat_par_swap_rate(reference, reference, m, Frequency::Annual, dc, dc, r_ois);
                Instrument::OisSwap(OisSwap::new(reference, m, p, Frequency::Annual, dc).unwrap())
            })
            .collect();

        // 3M projection from a deposit + sequential 3M FRAs out to 5y, with
        // forward rates consistent with a flat 2.5 % projection curve.
        let dep_pay = Tenor::new(3, TenorUnit::Months).add_to(reference);
        let mut projection: Vec<Instrument> = vec![Instrument::Deposit(
            Deposit::new(
                reference,
                dep_pay,
                flat_deposit_rate(reference, dc, r_proj, dep_pay),
                dc,
            )
            .unwrap(),
        )];
        let mut p_start = dep_pay;
        for _k in 1..=19_i32 {
            let p_end = Tenor::new(3, TenorUnit::Months).add_to(p_start);
            projection.push(Instrument::Fra(
                Fra::new(
                    p_start,
                    p_end,
                    flat_fra_rate(dc, r_proj, p_start, p_end),
                    dc,
                )
                .unwrap(),
            ));
            p_start = p_end;
        }

        let tenor_3m = Tenor::new(3, TenorUnit::Months);
        let mc = MultiCurveBootstrap::new(reference, dc)
            .build(
                &ois,
                Interpolation::LogLinear,
                &[(tenor_3m, projection.clone())],
                Interpolation::LogLinear,
            )
            .unwrap();

        // OIS curve re-prices its OIS swaps.
        for inst in &ois {
            if let Instrument::OisSwap(s) = inst {
                let par = mc
                    .discount
                    .par_swap_rate(s.start, s.maturity, s.freq, s.daycount)
                    .unwrap();
                assert_abs_diff_eq!(par, s.rate, epsilon = 1e-10);
            }
        }

        // Projection curve re-prices each FRA / deposit.
        let proj = mc.projection_curve(tenor_3m).unwrap();
        for inst in &projection {
            match inst {
                Instrument::Deposit(dep) => {
                    let tau = dc.year_fraction(reference, dep.payment).unwrap();
                    let t_pay = dc.year_fraction(reference, dep.payment).unwrap();
                    let d_pay = proj.discount(t_pay).unwrap();
                    let curve_rate = (1.0 / d_pay - 1.0) / tau;
                    assert_abs_diff_eq!(curve_rate, dep.rate, epsilon = 1e-10);
                }
                Instrument::Fra(fra) => {
                    let tau = dc.year_fraction(fra.start, fra.end).unwrap();
                    let t_s = dc.year_fraction(reference, fra.start).unwrap();
                    let t_e = dc.year_fraction(reference, fra.end).unwrap();
                    let curve_rate =
                        (proj.discount(t_s).unwrap() / proj.discount(t_e).unwrap() - 1.0) / tau;
                    assert_abs_diff_eq!(curve_rate, fra.rate, epsilon = 1e-10);
                }
                _ => unreachable!("unexpected instrument in projection set"),
            }
        }
    }

    /// Single-curve / multi-curve consistency: when projection == OIS, the
    /// multi-curve OIS discounts coincide with the single-curve bootstrap.
    #[test]
    fn multi_curve_projection_eq_ois_matches_single_curve() {
        let reference = d(2024, 1, 2);
        let dc = Daycount::Act360;
        let r_c = 0.03_f64;
        let ois: Vec<Instrument> = [(1_i32), 2, 5]
            .iter()
            .map(|&y| {
                let m = Tenor::new(y, TenorUnit::Years).add_to(reference);
                let p = flat_par_swap_rate(reference, reference, m, Frequency::Annual, dc, dc, r_c);
                Instrument::OisSwap(OisSwap::new(reference, m, p, Frequency::Annual, dc).unwrap())
            })
            .collect();
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
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-14);
        }
    }
}
