// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Quickstart example for regit-curves.
//!
//! Walks through the canonical yield-curve workflow end to end on a realistic
//! USD market snapshot:
//!
//! 1. Define a market quote set (deposits + FRAs + swaps) as of 2024-01-02.
//! 2. Bootstrap a single-curve log-linear discount curve from those quotes.
//! 3. Print the curve knots and a handful of spot rates / par swap rates.
//! 4. Build the OIS + 3M projection multi-curve set on the same reference date.
//! 5. Read the OIS, 3M projection, and a 5y multi-curve par swap rate.
//! 6. Demonstrate the three curve views — `ZeroCurve`, `ForwardCurve`, and
//!    `ParCurve` — over the single-curve discount curve.
//!
//! Run with:
//!
//! ```text
//! cargo run --example quickstart
//! ```

use std::error::Error;

use regit_curves::bootstrap::Bootstrap;
use regit_curves::curves::{DiscountCurve, ForwardCurve, ParCurve, ZeroCurve};
use regit_curves::instruments::{Deposit, Fra, Instrument, OisSwap, SwapFixedFloat};
use regit_curves::interpolation::Interpolation;
use regit_curves::multi_curve::{MultiCurve, MultiCurveBootstrap};
use regit_curves::types::{Compounding, Date, Daycount, Frequency, Tenor, TenorUnit};

type BoxedError = Box<dyn Error>;

/// Helper: add a `count`-month tenor to a `Date`.
fn add_months(start: Date, months: i32) -> Date {
    Tenor::new(months, TenorUnit::Months).add_to(start)
}

/// Helper: add a `count`-year tenor to a `Date`.
fn add_years(start: Date, years: i32) -> Date {
    Tenor::new(years, TenorUnit::Years).add_to(start)
}

/// Builds the single-curve instrument set (deposits + FRAs + swaps).
fn single_curve_instruments(reference: Date, dc: Daycount) -> Result<Vec<Instrument>, BoxedError> {
    let deposits = [
        (Tenor::new(7, TenorUnit::Days).add_to(reference), 0.0542),
        (add_months(reference, 1), 0.0540),
        (add_months(reference, 2), 0.0538),
        (add_months(reference, 3), 0.0535),
        (add_months(reference, 4), 0.0530),
        (add_months(reference, 6), 0.0520),
    ];
    let fras = [
        (add_months(reference, 6), add_months(reference, 9), 0.0510),
        (add_months(reference, 9), add_months(reference, 12), 0.0495),
        (add_months(reference, 12), add_months(reference, 15), 0.0480),
        (add_months(reference, 15), add_months(reference, 18), 0.0465),
    ];
    let swap_quotes = [
        (add_years(reference, 2), 0.0425),
        (add_years(reference, 3), 0.0395),
        (add_years(reference, 5), 0.0380),
        (add_years(reference, 7), 0.0385),
        (add_years(reference, 10), 0.0395),
    ];

    let mut out: Vec<Instrument> = Vec::new();
    for (payment, rate) in deposits {
        out.push(Instrument::Deposit(Deposit::new(
            reference, payment, rate, dc,
        )?));
    }
    for (start, end, rate) in fras {
        out.push(Instrument::Fra(Fra::new(start, end, rate, dc)?));
    }
    for (maturity, rate) in swap_quotes {
        out.push(Instrument::SwapFixedFloat(SwapFixedFloat::new(
            reference,
            maturity,
            rate,
            Frequency::SemiAnnual,
            Daycount::Act360,
            Frequency::Quarterly,
            Daycount::Act360,
        )?));
    }
    Ok(out)
}

/// Prints the single-curve summary: knots, zero rates, par swap rates.
fn print_single_curve_summary(
    reference: Date,
    curve: &DiscountCurve,
    instruments_len: usize,
) -> Result<(), BoxedError> {
    println!("Single-curve LogLinear bootstrap");
    println!("  instruments : {instruments_len}");
    println!(
        "  knots       : {} (anchor + per-instrument pillars)",
        curve.times().len()
    );
    println!();
    println!("  curve knots (t, D(t)):");
    for (t, d_t) in curve.times().iter().zip(curve.discounts().iter()) {
        println!("    t = {t:>8.4}    D = {d_t:.8}");
    }
    println!();

    let probe_tenors_yr = [0.25_f64, 0.5, 1.0, 2.0, 5.0, 10.0];
    println!("  continuous zero rates:");
    for &t in &probe_tenors_yr {
        let z = curve.zero_rate(t, Compounding::Continuous)?;
        println!("    z({t:>5.2}y) = {:.4}%", z * 100.0);
    }
    println!();

    let par_tenors = [2_i32, 5, 10];
    println!("  re-priced par swap rates (semi-annual, Act/360):");
    for &n in &par_tenors {
        let maturity = add_years(reference, n);
        let par =
            curve.par_swap_rate(reference, maturity, Frequency::SemiAnnual, Daycount::Act360)?;
        println!("    par({n}y) = {:.4}%", par * 100.0);
    }
    println!();
    Ok(())
}

/// Builds the OIS instrument strip.
fn ois_instruments(reference: Date, dc: Daycount) -> Result<Vec<Instrument>, BoxedError> {
    let quotes = [
        (add_years(reference, 1), 0.0500),
        (add_years(reference, 2), 0.0445),
        (add_years(reference, 3), 0.0420),
        (add_years(reference, 5), 0.0405),
        (add_years(reference, 10), 0.0415),
    ];
    let mut out: Vec<Instrument> = Vec::new();
    for (maturity, rate) in quotes {
        out.push(Instrument::OisSwap(OisSwap::new(
            reference,
            maturity,
            rate,
            Frequency::Annual,
            dc,
        )?));
    }
    Ok(out)
}

/// Builds the 3M projection-curve instrument strip (deposit + 19 FRAs).
fn projection_instruments(reference: Date, dc: Daycount) -> Result<Vec<Instrument>, BoxedError> {
    let proj_dep_pay = add_months(reference, 3);
    let mut out: Vec<Instrument> = vec![Instrument::Deposit(Deposit::new(
        reference,
        proj_dep_pay,
        0.0535,
        dc,
    )?)];
    let mut p_start = proj_dep_pay;
    let fra_quotes = [
        0.0510, 0.0490, 0.0470, 0.0450, 0.0430, 0.0415, 0.0405, 0.0400, 0.0395, 0.0390, 0.0388,
        0.0386, 0.0385, 0.0386, 0.0388, 0.0390, 0.0392, 0.0395, 0.0398,
    ];
    for &rate in &fra_quotes {
        let p_end = add_months(p_start, 3);
        out.push(Instrument::Fra(Fra::new(p_start, p_end, rate, dc)?));
        p_start = p_end;
    }
    Ok(out)
}

/// Multi-curve par swap rate against `mc` for a swap from `start` to
/// `maturity` paying fixed `fixed_freq` and float `float_freq` (3M projection).
#[allow(clippy::too_many_arguments)]
fn multi_curve_par_swap_rate(
    reference: Date,
    mc: &MultiCurve,
    proj: &DiscountCurve,
    curve_dc: Daycount,
    start: Date,
    maturity: Date,
    fixed_freq: Frequency,
    float_freq: Frequency,
) -> Result<f64, BoxedError> {
    let months_fixed = i32::try_from(12_u32 / fixed_freq.periods_per_year()).unwrap_or(6);
    let mut annuity = 0.0_f64;
    let mut step = months_fixed;
    let mut prev = start;
    loop {
        let next = add_months(start, step);
        let tau = Daycount::Act360.year_fraction(prev, next)?;
        let t_pay = curve_dc.year_fraction(reference, next)?;
        annuity += tau * mc.discount.discount(t_pay)?;
        if next == maturity {
            break;
        }
        prev = next;
        step += months_fixed;
    }
    let months_float = i32::try_from(12_u32 / float_freq.periods_per_year()).unwrap_or(3);
    let mut float_pv = 0.0_f64;
    let mut step = months_float;
    let mut prev = start;
    loop {
        let next = add_months(start, step);
        let t_p_start = curve_dc.year_fraction(reference, prev)?;
        let t_p_end = curve_dc.year_fraction(reference, next)?;
        let d_proj_start = proj.discount(t_p_start)?;
        let d_proj_end = proj.discount(t_p_end)?;
        let d_ois_end = mc.discount.discount(t_p_end)?;
        float_pv += (d_proj_start / d_proj_end - 1.0) * d_ois_end;
        if next == maturity {
            break;
        }
        prev = next;
        step += months_float;
    }
    Ok(float_pv / annuity)
}

/// Prints the multi-curve summary: OIS knots, 3M knots, 5Y multi-curve par.
fn print_multi_curve_summary(
    reference: Date,
    curve_dc: Daycount,
    mc: &MultiCurve,
    tenor_3m: Tenor,
) -> Result<(), BoxedError> {
    println!("Multi-curve OIS + 3M projection bootstrap");
    println!("  OIS curve knots ({}):", mc.discount.times().len());
    for (t, d_t) in mc
        .discount
        .times()
        .iter()
        .zip(mc.discount.discounts().iter())
    {
        println!("    t = {t:>8.4}    D_OIS = {d_t:.8}");
    }
    println!();
    let proj = mc
        .projection_curve(tenor_3m)
        .ok_or("3M projection curve missing")?;
    println!("  3M projection curve knots ({}):", proj.times().len());
    for (t, d_t) in proj.times().iter().zip(proj.discounts().iter()) {
        println!("    t = {t:>8.4}    D_3M  = {d_t:.8}");
    }
    println!();

    let par_5y_multi = multi_curve_par_swap_rate(
        reference,
        mc,
        proj,
        curve_dc,
        reference,
        add_years(reference, 5),
        Frequency::SemiAnnual,
        Frequency::Quarterly,
    )?;
    println!("  5y multi-curve par swap rate (SA fixed vs Q float on 3M):");
    println!("    par = {:.4}%", par_5y_multi * 100.0);
    println!();
    Ok(())
}

/// Prints the three curve-view results on the single-curve discount curve.
fn print_curve_views(reference: Date, curve: &DiscountCurve) -> Result<(), BoxedError> {
    println!("Curve views (single-curve discount curve)");
    let z_view = ZeroCurve::from(curve, Compounding::Continuous);
    let f_view = ForwardCurve::from(curve);
    let p_view = ParCurve::from(curve);
    let t_probe = 2.0_f64;
    println!(
        "  ZeroCurve.rate(2.0)             = {:.4}%  (continuous)",
        z_view.rate(t_probe)? * 100.0,
    );
    println!(
        "  ForwardCurve.instantaneous(2.0) = {:.4}%",
        f_view.instantaneous(t_probe)? * 100.0,
    );
    println!(
        "  ForwardCurve.forward(1, 2, A360)= {:.4}%  (simply compounded)",
        f_view.forward(1.0, 2.0, Daycount::Act360)? * 100.0,
    );
    println!(
        "  ParCurve.par_rate_from_anchor(5y, SA, A360) = {:.4}%",
        p_view.par_rate_from_anchor(
            add_years(reference, 5),
            Frequency::SemiAnnual,
            Daycount::Act360,
        )? * 100.0,
    );
    Ok(())
}

fn main() -> Result<(), BoxedError> {
    let reference = Date::from_ymd(2024, 1, 2)?;
    let curve_dc = Daycount::Act360;

    println!("regit-curves quickstart");
    println!("=========================");
    println!("reference date: 2024-01-02 (USD market)");
    println!();

    // Single-curve bootstrap.
    let instruments = single_curve_instruments(reference, curve_dc)?;
    let curve =
        Bootstrap::new(reference, curve_dc).build(&instruments, Interpolation::LogLinear)?;
    print_single_curve_summary(reference, &curve, instruments.len())?;

    // Multi-curve OIS + 3M projection bootstrap.
    let ois = ois_instruments(reference, curve_dc)?;
    let projection_set = projection_instruments(reference, curve_dc)?;
    let tenor_3m = Tenor::new(3, TenorUnit::Months);
    let mc = MultiCurveBootstrap::new(reference, curve_dc).build(
        &ois,
        Interpolation::LogLinear,
        &[(tenor_3m, projection_set)],
        Interpolation::LogLinear,
    )?;
    print_multi_curve_summary(reference, curve_dc, &mc, tenor_3m)?;

    // Curve views on the single-curve discount curve.
    print_curve_views(reference, &curve)?;
    Ok(())
}
