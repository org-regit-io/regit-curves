// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Tridiagonal solve by the Thomas algorithm.
//!
//! The Thomas algorithm (Thomas 1949) solves the tridiagonal system
//!
//! ```text
//!   b_0  c_0                          x_0       d_0
//!   a_1  b_1  c_1                     x_1       d_1
//!         .    .    .                  .    =    .
//!                a_{n-2} b_{n-2} c_{n-2}  x_{n-2}   d_{n-2}
//!                          a_{n-1} b_{n-1}  x_{n-1}   d_{n-1}
//! ```
//!
//! in `O(n)` operations via a single forward sweep that eliminates the
//! sub-diagonal and a single back substitution. The sub-diagonal `a` and
//! super-diagonal `c` are treated as full-length vectors with `a[0]` and
//! `c[n-1]` unused (callers may set them to anything finite).
//!
//! The algorithm assumes the tridiagonal matrix is non-singular along the
//! elimination path — i.e. no zero pivot is generated. Diagonally dominant
//! matrices, which is the case for natural cubic-spline construction and
//! for Hyman's monotonicity filter, satisfy this condition trivially.
//!
//! # References
//!
//! - Thomas, L. H., *Elliptic Problems in Linear Difference Equations over a
//!   Network*, Watson Sci. Comput. Lab. Report (Columbia 1949).
//! - Conte, S. D. & de Boor, C., *Elementary Numerical Analysis*, 3rd ed.,
//!   McGraw-Hill (1980), §4.3.

use super::MathError;

/// Tolerance below which a pivot is treated as zero.
const PIVOT_TOLERANCE: f64 = 1e-14;

/// Solves the tridiagonal system `T x = d` in `O(n)` time by the Thomas
/// algorithm.
///
/// The matrix `T` is given by its three diagonals:
///
/// - `a` — sub-diagonal, length `n`. `a[0]` is unused.
/// - `b` — main diagonal, length `n`.
/// - `c` — super-diagonal, length `n`. `c[n-1]` is unused.
///
/// `d` is the right-hand side, length `n`. The solution `x` is returned as
/// a fresh `Vec<f64>` of length `n`; the inputs are not modified.
///
/// # Errors
///
/// - [`MathError::DimensionMismatch`] if `a`, `b`, `c`, and `d` do not all
///   have the same length, or if that length is zero.
/// - [`MathError::Singular`] if the forward sweep produces a pivot whose
///   absolute value falls below `1e-14`.
///
/// # Examples
///
/// ```
/// use regit_curves::math::tridiag::thomas;
///
/// // 3x3 system:
/// // [[2, 1, 0], [1, 2, 1], [0, 1, 2]] x = [4, 8, 8] -> x = [1, 2, 3].
/// let a = vec![0.0, 1.0, 1.0];
/// let b = vec![2.0, 2.0, 2.0];
/// let c = vec![1.0, 1.0, 0.0];
/// let d = vec![4.0, 8.0, 8.0];
/// let x = thomas(&a, &b, &c, &d).unwrap();
/// assert!((x[0] - 1.0).abs() < 1e-12);
/// assert!((x[1] - 2.0).abs() < 1e-12);
/// assert!((x[2] - 3.0).abs() < 1e-12);
/// ```
// The textbook tridiagonal vectors `a, b, c, d` (Thomas 1949) are the
// canonical names — kept verbatim from the primary source.
#[allow(clippy::many_single_char_names)]
pub fn thomas(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Result<Vec<f64>, MathError> {
    let n = b.len();
    if n == 0 || a.len() != n || c.len() != n || d.len() != n {
        return Err(MathError::DimensionMismatch);
    }

    // Forward sweep: eliminate the sub-diagonal.
    let mut c_prime = vec![0.0_f64; n];
    let mut d_prime = vec![0.0_f64; n];

    if b[0].abs() < PIVOT_TOLERANCE {
        return Err(MathError::Singular);
    }
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i] * c_prime[i - 1];
        if denom.abs() < PIVOT_TOLERANCE {
            return Err(MathError::Singular);
        }
        if i < n - 1 {
            c_prime[i] = c[i] / denom;
        }
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom;
    }

    // Back substitution.
    let mut x = vec![0.0_f64; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..(n - 1)).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

#[cfg(test)]
// Test cases use the textbook tridiagonal names `a, b, c, d` matching the
// `thomas` signature; that lint is noise here.
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;
    use crate::math::linear_solve::solve;

    #[test]
    fn thomas_3x3_known() {
        // [[2,1,0],[1,2,1],[0,1,2]] x = [4,8,8] -> x = [1,2,3].
        let a = vec![0.0, 1.0, 1.0];
        let b = vec![2.0, 2.0, 2.0];
        let c = vec![1.0, 1.0, 0.0];
        let d = vec![4.0, 8.0, 8.0];
        let x = thomas(&a, &b, &c, &d).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn thomas_4x4_diagonally_dominant() {
        // [[4,-1,0,0],[-1,4,-1,0],[0,-1,4,-1],[0,0,-1,4]] x = [5,5,10,15]
        // Verify by reconstructing.
        let a = vec![0.0, -1.0, -1.0, -1.0];
        let b = vec![4.0, 4.0, 4.0, 4.0];
        let c = vec![-1.0, -1.0, -1.0, 0.0];
        let d = vec![5.0, 5.0, 10.0, 15.0];
        let x = thomas(&a, &b, &c, &d).unwrap();
        // Check the system.
        let r0 = 4.0 * x[0] - x[1] - 5.0;
        let r1 = -x[0] + 4.0 * x[1] - x[2] - 5.0;
        let r2 = -x[1] + 4.0 * x[2] - x[3] - 10.0;
        let r3 = -x[2] + 4.0 * x[3] - 15.0;
        for &r in &[r0, r1, r2, r3] {
            assert!(r.abs() < 1e-12);
        }
    }

    #[test]
    fn thomas_n2_degenerate() {
        // 2x2 tridiagonal is just a 2x2 dense: [[2, 3], [4, 5]] x = [13, 23].
        // Determinant = 10-12 = -2; x = [(13·5 - 3·23)/-2, (2·23 - 13·4)/-2]
        //              = [(65 - 69)/-2, (46 - 52)/-2] = [2, 3].
        let a = vec![0.0, 4.0];
        let b = vec![2.0, 5.0];
        let c = vec![3.0, 0.0];
        let d = vec![13.0, 23.0];
        let x = thomas(&a, &b, &c, &d).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn thomas_n1_trivial() {
        // 1x1 system: b[0] x = d[0] -> x = d[0]/b[0].
        let a = vec![0.0];
        let b = vec![5.0];
        let c = vec![0.0];
        let d = vec![10.0];
        let x = thomas(&a, &b, &c, &d).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn thomas_dimension_mismatch() {
        let a = vec![0.0, 1.0];
        let b = vec![2.0, 2.0, 2.0];
        let c = vec![1.0, 1.0, 0.0];
        let d = vec![4.0, 8.0, 8.0];
        let err = thomas(&a, &b, &c, &d).unwrap_err();
        assert!(matches!(err, MathError::DimensionMismatch));
    }

    #[test]
    fn thomas_empty_rejected() {
        let empty: Vec<f64> = vec![];
        let err = thomas(&empty, &empty, &empty, &empty).unwrap_err();
        assert!(matches!(err, MathError::DimensionMismatch));
    }

    #[test]
    fn thomas_zero_pivot_rejected() {
        let a = vec![0.0, 1.0];
        let b = vec![0.0, 2.0];
        let c = vec![1.0, 0.0];
        let d = vec![1.0, 1.0];
        let err = thomas(&a, &b, &c, &d).unwrap_err();
        assert!(matches!(err, MathError::Singular));
    }

    #[test]
    fn thomas_singular_during_sweep() {
        // Construct so b[1] - a[1] * c_prime[0] = 0:
        // c_prime[0] = c[0]/b[0] = 2/1 = 2; b[1] - a[1] * 2 = 0 requires
        // b[1] = 2*a[1]. With a[1]=1, b[1]=2 -> singular.
        let a = vec![0.0, 1.0];
        let b = vec![1.0, 2.0];
        let c = vec![2.0, 0.0];
        let d = vec![1.0, 1.0];
        let err = thomas(&a, &b, &c, &d).unwrap_err();
        assert!(matches!(err, MathError::Singular));
    }

    #[test]
    fn thomas_cross_check_with_dense_solve_6x6() {
        // 6x6 tridiagonal SPD ("smoothness Laplacian"): construct the dense
        // matrix and check both solvers agree.
        let sub = [-1.0_f64; 6];
        let diag = [2.0_f64; 6];
        let sup = [-1.0_f64; 6];
        let a = sub.to_vec();
        let b = diag.to_vec();
        let c = sup.to_vec();
        let d_rhs = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        // Solve with Thomas.
        let x_thomas = thomas(&a, &b, &c, &d_rhs).unwrap();

        // Build the dense matrix and solve with `solve`.
        let mut dense = vec![vec![0.0_f64; 6]; 6];
        for i in 0..6 {
            dense[i][i] = b[i];
            if i + 1 < 6 {
                dense[i][i + 1] = c[i];
                dense[i + 1][i] = a[i + 1];
            }
        }
        let mut rhs = d_rhs.clone();
        solve(&mut dense, &mut rhs).unwrap();

        for (t, g) in x_thomas.iter().zip(rhs.iter()) {
            assert!((t - g).abs() < 1e-10);
        }
    }
}
