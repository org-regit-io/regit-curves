// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for regit-curves.
//!
//! Performance targets (indicative, native release on commodity hardware):
//!
//! | Operation                                    | Target   |
//! |----------------------------------------------|----------|
//! | LogLinear `eval` (20 knots)                  | < 25 ns  |
//! | CubicSpline `eval` (20 knots)                | < 50 ns  |
//! | MonotoneCubic `eval` (20 knots)              | < 60 ns  |
//! | DiscountCurve `zero_rate` lookup             | < 60 ns  |
//! | DiscountCurve `par_swap_rate` (5y SA)        | < 5 us   |
//! | Bootstrap LogLinear (10 instruments)         | < 200 us |
//! | Bootstrap CubicSpline (10 instruments)       | < 2 ms   |
//! | Multi-curve OIS + 3M (10 instruments)        | < 500 us |

use criterion::{Criterion, criterion_group, criterion_main};
use regit_curves::bootstrap::Bootstrap;
use regit_curves::curves::DiscountCurve;
use regit_curves::instruments::{Deposit, Fra, Instrument, OisSwap, SwapFixedFloat};
use regit_curves::interpolation::{
    CubicSpline, Interpolation, Interpolator, LogLinear, MonotoneCubic, SplineBoundary,
};
use regit_curves::multi_curve::MultiCurveBootstrap;
use regit_curves::types::{Date, Daycount, Frequency, Tenor, TenorUnit};
use std::hint::black_box;

// ─── Fixtures ────────────────────────────────────────────────────────────────

/// 20-knot synthetic discount curve under continuous compounding at 4 %.
fn twenty_knots() -> Vec<(f64, f64)> {
    let r_c = 0.04_f64;
    (0..20)
        .map(|i| {
            let t = f64::from(i) * 0.5;
            (t, (-r_c * t).exp())
        })
        .collect()
}

fn reference_date() -> Date {
    Date::from_ymd(2024, 1, 2).expect("valid reference date")
}

fn add_months(start: Date, months: i32) -> Date {
    Tenor::new(months, TenorUnit::Months).add_to(start)
}

fn add_years(start: Date, years: i32) -> Date {
    Tenor::new(years, TenorUnit::Years).add_to(start)
}

/// Ten-instrument input set used by the single-curve bootstrap benches.
fn ten_instruments() -> Vec<Instrument> {
    let reference = reference_date();
    let dc = Daycount::Act360;
    let mut out: Vec<Instrument> = Vec::with_capacity(10);
    // Three deposits.
    for &(months, rate) in &[(1_i32, 0.054_f64), (3, 0.0535), (6, 0.0520)] {
        out.push(Instrument::Deposit(
            Deposit::new(reference, add_months(reference, months), rate, dc)
                .expect("valid deposit"),
        ));
    }
    // Two FRAs.
    for &(start_m, end_m, rate) in &[(6_i32, 9_i32, 0.0500_f64), (9, 12, 0.0485)] {
        out.push(Instrument::Fra(
            Fra::new(
                add_months(reference, start_m),
                add_months(reference, end_m),
                rate,
                dc,
            )
            .expect("valid FRA"),
        ));
    }
    // Five swaps.
    for &(years, rate) in &[
        (2_i32, 0.0425_f64),
        (3, 0.0395),
        (5, 0.0380),
        (7, 0.0385),
        (10, 0.0395),
    ] {
        out.push(Instrument::SwapFixedFloat(
            SwapFixedFloat::new(
                reference,
                add_years(reference, years),
                rate,
                Frequency::SemiAnnual,
                Daycount::Act360,
                Frequency::Quarterly,
                Daycount::Act360,
            )
            .expect("valid swap"),
        ));
    }
    out
}

/// Five-OIS-swap input set used by the multi-curve bench.
fn ois_instruments_set() -> Vec<Instrument> {
    let reference = reference_date();
    let dc = Daycount::Act360;
    let quotes = [
        (1_i32, 0.0500_f64),
        (2, 0.0445),
        (3, 0.0420),
        (5, 0.0405),
        (10, 0.0415),
    ];
    quotes
        .iter()
        .map(|&(years, rate)| {
            Instrument::OisSwap(
                OisSwap::new(
                    reference,
                    add_years(reference, years),
                    rate,
                    Frequency::Annual,
                    dc,
                )
                .expect("valid OIS swap"),
            )
        })
        .collect()
}

/// Five-instrument 3M projection set (deposit + four 3M FRAs).
fn projection_set() -> Vec<Instrument> {
    let reference = reference_date();
    let dc = Daycount::Act360;
    let dep_pay = add_months(reference, 3);
    let mut out: Vec<Instrument> = vec![Instrument::Deposit(
        Deposit::new(reference, dep_pay, 0.0535, dc).expect("valid deposit"),
    )];
    let fra_rates = [0.0510_f64, 0.0490, 0.0470, 0.0450];
    let mut p_start = dep_pay;
    for &rate in &fra_rates {
        let p_end = add_months(p_start, 3);
        out.push(Instrument::Fra(
            Fra::new(p_start, p_end, rate, dc).expect("valid FRA"),
        ));
        p_start = p_end;
    }
    out
}

/// 20-knot `DiscountCurve` under `LogLinear` for the lookup benches.
fn twenty_knot_curve() -> DiscountCurve {
    let knots = twenty_knots();
    let times: Vec<f64> = knots.iter().map(|&(t, _)| t).collect();
    let discounts: Vec<f64> = knots.iter().map(|&(_, d)| d).collect();
    DiscountCurve::from_times_and_discounts(
        reference_date(),
        Daycount::Act365F,
        &times,
        &discounts,
        Interpolation::LogLinear,
    )
    .expect("valid curve")
}

// ─── Interpolator evaluation ────────────────────────────────────────────────

fn bench_log_linear_eval(c: &mut Criterion) {
    let knots = twenty_knots();
    let interp = LogLinear::new(&knots).expect("valid knots");
    c.bench_function("log_linear_eval_1M_calls", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for i in 0..1_000_000_u32 {
                // Spread queries across the curve; modulus by knot span.
                let t = f64::from(i % 10_000) * 0.001;
                acc += interp.eval(black_box(t));
            }
            acc
        });
    });
}

fn bench_cubic_spline_eval(c: &mut Criterion) {
    let knots = twenty_knots();
    let interp = CubicSpline::new(&knots, SplineBoundary::NotAKnot).expect("valid knots");
    c.bench_function("cubic_spline_eval_1M_calls", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for i in 0..1_000_000_u32 {
                let t = f64::from(i % 10_000) * 0.001;
                acc += interp.eval(black_box(t));
            }
            acc
        });
    });
}

fn bench_monotone_cubic_eval(c: &mut Criterion) {
    let knots = twenty_knots();
    let interp = MonotoneCubic::new(&knots).expect("valid knots");
    c.bench_function("monotone_cubic_eval_1M_calls", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for i in 0..1_000_000_u32 {
                let t = f64::from(i % 10_000) * 0.001;
                acc += interp.eval(black_box(t));
            }
            acc
        });
    });
}

// ─── Curve lookups ──────────────────────────────────────────────────────────

fn bench_zero_rate_lookup(c: &mut Criterion) {
    let curve = twenty_knot_curve();
    c.bench_function("zero_rate_lookup", |b| {
        b.iter(|| {
            curve
                .zero_rate(black_box(3.5), regit_curves::types::Compounding::Continuous)
                .expect("rate")
        });
    });
}

fn bench_par_swap_rate(c: &mut Criterion) {
    let curve = twenty_knot_curve();
    let reference = reference_date();
    let maturity = add_years(reference, 5);
    c.bench_function("par_swap_rate_5y_SA", |b| {
        b.iter(|| {
            curve
                .par_swap_rate(
                    black_box(reference),
                    black_box(maturity),
                    Frequency::SemiAnnual,
                    Daycount::Act360,
                )
                .expect("par rate")
        });
    });
}

// ─── Bootstrap ──────────────────────────────────────────────────────────────

fn bench_bootstrap_log_linear(c: &mut Criterion) {
    let instruments = ten_instruments();
    let reference = reference_date();
    let bs = Bootstrap::new(reference, Daycount::Act360);
    c.bench_function("bootstrap_log_linear_10_instruments", |b| {
        b.iter(|| {
            bs.build(black_box(&instruments), Interpolation::LogLinear)
                .expect("bootstrap")
        });
    });
}

fn bench_bootstrap_cubic_spline(c: &mut Criterion) {
    let instruments = ten_instruments();
    let reference = reference_date();
    let bs = Bootstrap::new(reference, Daycount::Act360);
    c.bench_function("bootstrap_cubic_spline_10_instruments", |b| {
        b.iter(|| {
            bs.build(
                black_box(&instruments),
                Interpolation::CubicSpline(SplineBoundary::NotAKnot),
            )
            .expect("bootstrap")
        });
    });
}

fn bench_multi_curve_bootstrap(c: &mut Criterion) {
    let ois = ois_instruments_set();
    let projection = projection_set();
    let reference = reference_date();
    let tenor_3m = Tenor::new(3, TenorUnit::Months);
    let engine = MultiCurveBootstrap::new(reference, Daycount::Act360);
    c.bench_function("multi_curve_bootstrap_ois_plus_3m", |b| {
        b.iter(|| {
            engine
                .build(
                    black_box(&ois),
                    Interpolation::LogLinear,
                    black_box(&[(tenor_3m, projection.clone())]),
                    Interpolation::LogLinear,
                )
                .expect("multi-curve bootstrap")
        });
    });
}

// ─── Harness ────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_log_linear_eval,
    bench_cubic_spline_eval,
    bench_monotone_cubic_eval,
    bench_zero_rate_lookup,
    bench_par_swap_rate,
    bench_bootstrap_log_linear,
    bench_bootstrap_cubic_spline,
    bench_multi_curve_bootstrap,
);
criterion_main!(benches);
