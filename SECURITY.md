<!-- Copyright 2026 Regit.io — Nicolas Koenig -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in `regit-curves`, please report it
responsibly:

1. **Do not** open a public GitHub issue
2. Email **nicolas.koenig@regit.io** with a description of the vulnerability
3. Include steps to reproduce if possible
4. We will acknowledge receipt within 48 hours and provide a timeline for a fix

## Scope

This crate performs mathematical computation only — it does not handle network
I/O, file I/O, user authentication, or any form of external communication. It
has zero runtime dependencies.

The primary security concern is **numerical correctness**: an error in the
bootstrap, in an interpolation algorithm, or in a curve conversion could lead
to a yield curve that misprices interest-rate instruments or carries
undetected mark-to-market error into a downstream pricing or risk system.

In particular:

- A **silent bootstrap failure** — a curve that reprices its bootstrap
  instruments to within numerical tolerance but where the implementation has a
  subtle bug elsewhere (wrong day-count, wrong compounding convention, wrong
  interpolation domain) — is treated as a correctness defect of the highest
  severity.
- An **interpolation method that disagrees with its primary source** on
  documented worked examples (Hagan & West, QuantLib golden vectors, Andersen
  & Piterbarg numerics) is similarly treated as a correctness defect.

If you find a numerical accuracy issue that falls outside the documented
tolerance bounds, or any case where a curve mis-evaluates against an oracle,
please report it using the process above.

## Dependencies

The crate has no runtime dependencies. License and supply-chain concerns for
development dependencies are policed via `cargo-deny` (`deny.toml` in the
repository root), checked in CI on every push. Dependency changes that
introduce a non-allowed licence or an active advisory are rejected at the gate.
