// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Dense linear-system solvers.
//!
//! Two solvers cover the dense linear algebra needed by the rest of the
//! crate:
//!
//! - [`solve`] — Gaussian elimination with partial (row) pivoting for a
//!   general square matrix `A`.
//! - [`solve_spd`] — Cholesky decomposition `A = L Lᵀ` followed by two
//!   triangular solves for a symmetric positive-definite `A`.
//!
//! Both routines are pure functions (no global state), deterministic, and
//! return a typed [`MathError`] on a domain failure
//! (singularity, non-SPD matrix, or dimension mismatch).
//!
//! # References
//!
//! - Golub, G. H. & Van Loan, C. F., *Matrix Computations*, 4th ed.,
//!   Johns Hopkins (2013), Algorithms 3.4.1 (LU with partial pivoting) and
//!   4.2.2 (Cholesky).

use super::MathError;

/// Tolerance below which a pivot is treated as zero in [`solve`].
const PIVOT_TOLERANCE: f64 = 1e-14;

/// Solves `A x = b` for a general square dense matrix via Gaussian
/// elimination with partial (row) pivoting.
///
/// The matrix `a` is modified in place to its row-echelon form, and `b` is
/// overwritten with the solution `x`. Pivot rows are swapped to use the
/// row with the largest absolute pivot at each step — the standard partial
/// pivoting strategy (Golub & Van Loan §3.4).
///
/// # Errors
///
/// - [`MathError::DimensionMismatch`] if `a` is not square or if
///   `b.len() != a.len()`.
/// - [`MathError::Singular`] if a pivot falls below `1e-14` in absolute
///   value (the matrix is singular up to round-off).
///
/// # Examples
///
/// ```
/// use regit_curves::math::linear_solve::solve;
///
/// // 2x2 system: [[2, 1], [5, 7]] x = [11, 13] -> x = [7.111..., -3.222...].
/// // Verified: 2 * 7.111 + 1 * -3.222 ≈ 11, 5 * 7.111 + 7 * -3.222 ≈ 13.
/// let mut a = vec![vec![2.0, 1.0], vec![5.0, 7.0]];
/// let mut b = vec![11.0, 13.0];
/// solve(&mut a, &mut b).unwrap();
/// assert!((b[0] - 64.0 / 9.0).abs() < 1e-12);
/// assert!((b[1] - (-29.0 / 9.0)).abs() < 1e-12);
/// ```
// Gaussian elimination is naturally expressed by row/column indices `i`, `j`,
// `k` indexing both `a` and `b` simultaneously; an iterator transformation
// would obscure the algorithm.
#[allow(clippy::needless_range_loop)]
pub fn solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<(), MathError> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return Err(MathError::DimensionMismatch);
    }
    for row in a.iter() {
        if row.len() != n {
            return Err(MathError::DimensionMismatch);
        }
    }

    // Forward elimination with partial pivoting.
    for k in 0..n {
        // Find the pivot row: the row with the largest |a[i][k]| for i >= k.
        let mut pivot_row = k;
        let mut pivot_val = a[k][k].abs();
        for i in (k + 1)..n {
            let v = a[i][k].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = i;
            }
        }
        if pivot_val < PIVOT_TOLERANCE {
            return Err(MathError::Singular);
        }
        if pivot_row != k {
            a.swap(k, pivot_row);
            b.swap(k, pivot_row);
        }

        // Eliminate column k below the pivot.
        let pivot = a[k][k];
        for i in (k + 1)..n {
            let factor = a[i][k] / pivot;
            if factor == 0.0 {
                continue;
            }
            a[i][k] = 0.0;
            for j in (k + 1)..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution.
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * b[j];
        }
        let pivot = a[i][i];
        if pivot.abs() < PIVOT_TOLERANCE {
            return Err(MathError::Singular);
        }
        b[i] = sum / pivot;
    }

    Ok(())
}

/// Solves `A x = b` for a symmetric positive-definite matrix via Cholesky
/// decomposition.
///
/// `A` is factored as `L Lᵀ` with `L` lower-triangular; then the two
/// triangular solves give `x`. Only the lower triangle of `a` is read.
///
/// # Errors
///
/// - [`MathError::DimensionMismatch`] if `a` is not square or if
///   `b.len() != a.len()`.
/// - [`MathError::NotSpd`] if a diagonal pivot of `L` is non-positive
///   (i.e. `A` is not positive-definite).
///
/// # Examples
///
/// ```
/// use regit_curves::math::linear_solve::solve_spd;
///
/// // 2x2 SPD: [[4, 2], [2, 3]] x = [10, 8].
/// let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
/// let b = vec![10.0, 8.0];
/// let x = solve_spd(&a, &b).unwrap();
/// assert!((4.0 * x[0] + 2.0 * x[1] - 10.0).abs() < 1e-12);
/// assert!((2.0 * x[0] + 3.0 * x[1] - 8.0).abs() < 1e-12);
/// ```
// Cholesky factorisation indexes `l[i][j]` and `l[j][k]` simultaneously; an
// iterator transformation would obscure the textbook algorithm. Likewise,
// the standard matrix-computation names `a, b, l, x, y, n` are inherited
// from Golub & Van Loan and kept verbatim.
#[allow(clippy::needless_range_loop, clippy::many_single_char_names)]
pub fn solve_spd(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, MathError> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return Err(MathError::DimensionMismatch);
    }
    for row in a {
        if row.len() != n {
            return Err(MathError::DimensionMismatch);
        }
    }

    // Cholesky: build L lower-triangular such that A = L Lᵀ.
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return Err(MathError::NotSpd);
                }
                l[i][j] = sum.sqrt();
            } else {
                let pivot = l[j][j];
                if pivot.abs() < PIVOT_TOLERANCE {
                    return Err(MathError::NotSpd);
                }
                l[i][j] = sum / pivot;
            }
        }
    }

    // Forward solve L y = b.
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }

    // Back solve Lᵀ x = y.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Multiplies a (square) matrix by a vector. Used in test assertions only.
    fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        a.iter()
            .map(|row| row.iter().zip(x.iter()).map(|(&aij, &xj)| aij * xj).sum())
            .collect()
    }

    // ─── solve (Gaussian elimination) ────────────────────────────────────

    #[test]
    fn solve_identity_2x2() {
        let mut a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut b = vec![3.0, 5.0];
        solve(&mut a, &mut b).unwrap();
        assert!((b[0] - 3.0).abs() < 1e-15);
        assert!((b[1] - 5.0).abs() < 1e-15);
    }

    #[test]
    fn solve_3x3_known_inverse() {
        // A = [[1, 2, 3], [2, 5, 3], [1, 0, 8]]
        // Designed so x = [1, 2, 3]: b = A x = [14, 21, 25].
        let a_orig = [
            vec![1.0, 2.0, 3.0],
            vec![2.0, 5.0, 3.0],
            vec![1.0, 0.0, 8.0],
        ];
        let mut a: Vec<Vec<f64>> = a_orig.to_vec();
        let mut b = matvec(&a_orig, &[1.0, 2.0, 3.0]);
        solve(&mut a, &mut b).unwrap();
        assert!((b[0] - 1.0).abs() < 1e-12);
        assert!((b[1] - 2.0).abs() < 1e-12);
        assert!((b[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn solve_requires_pivoting() {
        // Top-left entry is zero so the naive algorithm without pivoting
        // would fail; with partial pivoting it must succeed.
        let mut a = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let mut b = vec![2.0, 3.0];
        solve(&mut a, &mut b).unwrap();
        // The system is x_2 = 2, x_1 = 3.
        assert!((b[0] - 3.0).abs() < 1e-15);
        assert!((b[1] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn solve_singular_matrix_rejected() {
        // Rank-1 (singular).
        let mut a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let mut b = vec![1.0, 2.0];
        let err = solve(&mut a, &mut b).unwrap_err();
        assert!(matches!(err, MathError::Singular));
    }

    #[test]
    fn solve_dimension_mismatch_rejected() {
        let mut a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut b = vec![1.0];
        let err = solve(&mut a, &mut b).unwrap_err();
        assert!(matches!(err, MathError::DimensionMismatch));

        let mut a = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let mut b = vec![1.0, 2.0];
        let err = solve(&mut a, &mut b).unwrap_err();
        assert!(matches!(err, MathError::DimensionMismatch));
    }

    #[test]
    fn solve_empty_rejected() {
        let mut a: Vec<Vec<f64>> = vec![];
        let mut b: Vec<f64> = vec![];
        let err = solve(&mut a, &mut b).unwrap_err();
        assert!(matches!(err, MathError::DimensionMismatch));
    }

    #[test]
    fn solve_random_5x5() {
        // A diagonally-dominant 5x5 system; designed to be well-conditioned.
        let a_orig: Vec<Vec<f64>> = vec![
            vec![10.0, 1.0, 0.0, 2.0, -1.0],
            vec![1.0, 12.0, -2.0, 1.0, 0.0],
            vec![0.0, -1.0, 9.0, 0.0, 2.0],
            vec![2.0, 0.0, 1.0, 8.0, -2.0],
            vec![-1.0, 0.0, 1.0, -2.0, 11.0],
        ];
        let x_true = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let b_true = matvec(&a_orig, &x_true);
        let mut a = a_orig.clone();
        let mut b = b_true.clone();
        solve(&mut a, &mut b).unwrap();
        for (got, exp) in b.iter().zip(x_true.iter()) {
            assert!((got - exp).abs() < 1e-10);
        }
    }

    // ─── solve_spd (Cholesky) ────────────────────────────────────────────

    #[test]
    fn solve_spd_identity_3x3() {
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let b = vec![2.0, 3.0, 5.0];
        let x = solve_spd(&a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-15);
        assert!((x[1] - 3.0).abs() < 1e-15);
        assert!((x[2] - 5.0).abs() < 1e-15);
    }

    #[test]
    fn solve_spd_standard_3x3() {
        // Standard SPD example: A = [[4,12,-16],[12,37,-43],[-16,-43,98]].
        // Cholesky factor is L = [[2,0,0],[6,1,0],[-8,5,3]].
        // Designed so x = [1, 1, 1]: b = A x = [0, 6, 39].
        let a = vec![
            vec![4.0, 12.0, -16.0],
            vec![12.0, 37.0, -43.0],
            vec![-16.0, -43.0, 98.0],
        ];
        let x_true = vec![1.0, 1.0, 1.0];
        let b = matvec(&a, &x_true);
        let x = solve_spd(&a, &b).unwrap();
        for (got, exp) in x.iter().zip(x_true.iter()) {
            assert!((got - exp).abs() < 1e-10);
        }
    }

    #[test]
    fn solve_spd_rejects_non_spd() {
        // Indefinite matrix (eigenvalues +3 and -1).
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        let b = vec![1.0, 1.0];
        let err = solve_spd(&a, &b).unwrap_err();
        assert!(matches!(err, MathError::NotSpd));
    }

    #[test]
    fn solve_spd_rejects_zero_diagonal() {
        let a = vec![vec![0.0, 0.0], vec![0.0, 1.0]];
        let b = vec![0.0, 1.0];
        let err = solve_spd(&a, &b).unwrap_err();
        assert!(matches!(err, MathError::NotSpd));
    }

    #[test]
    fn solve_spd_dimension_mismatch() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0];
        let err = solve_spd(&a, &b).unwrap_err();
        assert!(matches!(err, MathError::DimensionMismatch));

        let a = vec![vec![1.0, 0.0]];
        let b = vec![1.0];
        let err = solve_spd(&a, &b).unwrap_err();
        assert!(matches!(err, MathError::DimensionMismatch));
    }

    #[test]
    fn solve_spd_empty_rejected() {
        let a: Vec<Vec<f64>> = vec![];
        let b: Vec<f64> = vec![];
        let err = solve_spd(&a, &b).unwrap_err();
        assert!(matches!(err, MathError::DimensionMismatch));
    }

    #[test]
    fn solve_spd_cross_check_with_solve() {
        // Same SPD matrix solved via both routines: results must agree.
        let a = vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let x_true = vec![1.0, -1.0, 2.0];
        let b = matvec(&a, &x_true);
        let x_spd = solve_spd(&a, &b).unwrap();
        let mut a_mut = a.clone();
        let mut b_mut = b.clone();
        solve(&mut a_mut, &mut b_mut).unwrap();
        for (s, g) in x_spd.iter().zip(b_mut.iter()) {
            assert!((s - g).abs() < 1e-10);
        }
    }
}
