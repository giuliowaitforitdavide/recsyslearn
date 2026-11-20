//! Aggregation of per-user scores into the score of the system.
//!
//! nDCG and novelty are defined per user and reported as the mean over the users for
//! whom they are defined:
//!
//! ```text
//! metric = (1 / |U'|) sum_{u in U'} metric(u)
//! ```
//!
//! where `U'` are the users with a defined score (the functions of this crate return
//! `NaN` for the others).

/// Mean of the scores that are not `NaN`, or `None` when there is none.
///
/// The scores are added in the order given, so the result is the same for the same
/// users in the same order, whatever the number of threads.
#[must_use]
pub fn mean_over_users(scores: &[f64]) -> Option<f64> {
    let (total, defined) = scores
        .iter()
        .filter(|score| !score.is_nan())
        .fold((0.0f64, 0usize), |(total, defined), score| {
            (total + score, defined + 1)
        });
    (defined > 0).then(|| total / defined as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_skips_undefined_scores() {
        // (1 + 0 + 0.5) / 3: the NaN user is not counted in the denominator either.
        assert_eq!(mean_over_users(&[1.0, f64::NAN, 0.0, 0.5]), Some(0.5));
    }

    #[test]
    fn no_defined_score_has_no_mean() {
        assert_eq!(mean_over_users(&[]), None);
        assert_eq!(mean_over_users(&[f64::NAN, f64::NAN]), None);
    }
}
