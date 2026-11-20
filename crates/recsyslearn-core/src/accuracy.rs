//! Accuracy of the ranking: normalised discounted cumulative gain.
//!
//! # Definition
//!
//! For a user `u` with recommendation list `L_u` (best first) and relevance `rel_u(i)`
//! taken from the held-out ground truth,
//!
//! ```text
//! DCG@k(u)  = sum_{p=1..min(k, |L_u|)} g(rel_u(L_u[p])) / log2(p + 1)
//! IDCG@k(u) = DCG@k of the user's ground-truth items sorted by decreasing gain
//! nDCG@k(u) = DCG@k(u) / IDCG@k(u)
//! ```
//!
//! with gain `g(r) = r` ([`Gain::Linear`]; Järvelin & Kekäläinen 2002, Wang et al.
//! 2013) or `g(r) = 2^r - 1` ([`Gain::Exponential`]; Burges et al. 2005). With binary
//! relevance both gains coincide and `IDCG@k(u) = sum_{p=1..min(k, |Rel_u|)}
//! 1 / log2(p + 1)` (Krichene & Rendle 2020, Eq. 8).
//!
//! The ideal ranking is built from *all* the judged items of the user, not only from
//! the recommended ones; positions missing from a list shorter than `k` contribute no
//! gain. The metric is undefined (`NaN`) for users without any item of positive gain.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::lists::{GroundTruth, RankedLists};

/// How a relevance grade is turned into a gain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Gain {
    /// `g(r) = r` (Järvelin & Kekäläinen 2002).
    #[default]
    Linear,
    /// `g(r) = 2^r - 1` (Burges et al. 2005), which emphasises highly relevant items.
    Exponential,
}

impl Gain {
    fn apply(self, relevance: f64) -> f64 {
        match self {
            Self::Linear => relevance,
            // 2^r - 1 cancels for a small r: there, e^(r ln 2) - 1 is computed directly.
            Self::Exponential if relevance < 0.5 => (relevance * std::f64::consts::LN_2).exp_m1(),
            Self::Exponential => relevance.exp2() - 1.0,
        }
    }
}

/// Discount `1 / log2(p + 1)` of the 1-based positions `1..=len`.
pub(crate) fn log2_discounts(len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| 1.0 / (index as f64 + 2.0).log2())
        .collect()
}

/// nDCG@k of every user; `NaN` for users without any item of positive gain.
///
/// `k = None` evaluates the lists in full.
///
/// # Errors
///
/// Fails when `k` is zero, the two inputs do not describe the same number of users, a
/// relevance is too large for its exponential gain to be a finite number, or the gains
/// of a user are too large for their discounted sum to be one.
pub fn ndcg(
    lists: &RankedLists,
    truth: &GroundTruth,
    k: Option<usize>,
    gain: Gain,
) -> Result<Vec<f64>> {
    if k == Some(0) {
        return Err(Error::invalid("k must be a positive integer"));
    }
    if lists.n_users() != truth.n_users() {
        return Err(Error::invalid(format!(
            "recommendations describe {} users but the ground truth describes {}",
            lists.n_users(),
            truth.n_users()
        )));
    }

    if truth
        .all_relevances()
        .iter()
        .any(|&relevance| !gain.apply(relevance).is_finite())
    {
        return Err(Error::invalid(
            "a relevance is too large for the exponential gain: 2^relevance - 1 is not finite \
             (use the linear gain, or rescale the relevance)",
        ));
    }

    let longest_truth = (0..truth.n_users())
        .map(|user| truth.relevances(user).len())
        .max()
        .unwrap_or(0);
    let horizon = lists.max_len().max(longest_truth);
    let discounts = log2_discounts(k.map_or(horizon, |k| k.min(horizon)));

    let scores: Result<Vec<f64>> = (0..lists.n_users())
        .into_par_iter()
        .map(|user| {
            let mut ideal_gains: Vec<f64> = truth
                .relevances(user)
                .iter()
                .map(|&relevance| gain.apply(relevance))
                .filter(|&value| value > 0.0)
                .collect();
            if ideal_gains.is_empty() {
                return Ok(f64::NAN);
            }
            ideal_gains.sort_unstable_by(|a, b| b.total_cmp(a));
            let ideal: f64 = ideal_gains
                .iter()
                .zip(&discounts)
                .map(|(value, discount)| value * discount)
                .sum();

            // With tied scores, the discount of an item is the one it is expected to
            // receive over the orders of its tie.
            let achieved: f64 = lists
                .weighted(user, k, &discounts)
                .map(|(item, discount)| gain.apply(truth.relevance(user, item)) * discount)
                .sum();
            // No list beats the ideal one, but with ties the two sums are not rounded
            // alike: the achieved one adds the mean discount of each tie, the ideal one
            // the discounts one by one. Hence the limit of 1.
            //
            // The sum of no terms is -0.0 in Rust (the IEEE 754 identity of addition), so
            // a user whose list is empty would score -0.0, which prints as "-0.0" and has
            // its sign bit set. Under IEEE 754 -0.0 + 0.0 is +0.0, and every other value is
            // unchanged by the addition.
            // The limit comes first: once a minimum follows it, the compiler may drop
            // the addition, since the sign of a zero does not change a minimum.
            let score = (achieved / ideal).min(1.0) + 0.0;
            // Finite gains can still overflow when added: inf / inf would be NaN, which
            // means "no relevant item" to the caller.
            if ideal.is_finite() && achieved.is_finite() {
                Ok(score)
            } else {
                Err(Error::invalid(
                    "the gains of a user are too large to be added up: rescale the relevance",
                ))
            }
        })
        .collect();
    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::assert_fails;

    const TOLERANCE: f64 = 1e-6;

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < TOLERANCE
    }

    fn single_user(
        recommended: &[u32],
        relevant: &[u32],
        grades: Option<&[f64]>,
    ) -> (RankedLists, GroundTruth) {
        let lists = RankedLists::new(&vec![0; recommended.len()], recommended, None, 1).unwrap();
        let truth = GroundTruth::new(&vec![0; relevant.len()], relevant, grades, 1).unwrap();
        (lists, truth)
    }

    /// Krichene & Rendle 2020, Eq. 8, on rec = [a, b, c, d, e] and Rel = {b, e, z}:
    /// DCG = 1/log2(3) + 1/log2(6) and IDCG has min(k, 3) terms.
    #[test]
    fn binary_relevance_matches_the_formula() {
        let (lists, truth) = single_user(&[0, 1, 2, 3, 4], &[1, 4, 5], None);
        let dcg = 1.0 / 3f64.log2() + 1.0 / 6f64.log2();
        let idcg3 = 1.0 + 1.0 / 3f64.log2() + 0.5;
        for (k, expected) in [
            (Some(5), dcg / idcg3),
            (Some(10), dcg / idcg3),
            (None, dcg / idcg3),
            (Some(2), (1.0 / 3f64.log2()) / (1.0 + 1.0 / 3f64.log2())),
        ] {
            let score = ndcg(&lists, &truth, k, Gain::Linear).unwrap()[0];
            assert!(close(score, expected), "k={k:?}: {score} != {expected}");
        }
        let at5 = ndcg(&lists, &truth, Some(5), Gain::Linear).unwrap()[0];
        let at2 = ndcg(&lists, &truth, Some(2), Gain::Linear).unwrap()[0];
        assert!(close(at5, 0.477_624));
        assert!(close(at2, 0.386_853));
    }

    /// rec = [a, b, c, d, e], ground truth {a: 3, b: 2, c: 1, x: 3, y: 2}: the ideal
    /// gains are [3, 3, 2, 2, 1] because the ideal uses every judged item.
    #[test]
    fn graded_relevance_matches_the_formula() {
        let grades = [3.0, 2.0, 1.0, 3.0, 2.0];
        let (lists, truth) = single_user(&[0, 1, 2, 3, 4], &[0, 1, 2, 5, 6], Some(&grades));
        // Discounted sum of the first `k` relevances, written out from the definition.
        let discounted = |relevances: [f64; 5], k: usize, exponential: bool| -> f64 {
            relevances[..k]
                .iter()
                .zip([2f64, 3.0, 4.0, 5.0, 6.0])
                .map(|(&r, next)| (if exponential { r.exp2() - 1.0 } else { r }) / next.log2())
                .sum()
        };
        for (k, gain, rounded) in [
            (3, Gain::Linear, 0.808_082),
            (3, Gain::Exponential, 0.727_193),
            (5, Gain::Linear, 0.666_834),
            (5, Gain::Exponential, 0.643_545),
        ] {
            let exponential = gain == Gain::Exponential;
            // The list has relevances [3, 2, 1, 0, 0]; the ideal one [3, 3, 2, 2, 1].
            let expected = discounted([3.0, 2.0, 1.0, 0.0, 0.0], k, exponential)
                / discounted([3.0, 3.0, 2.0, 2.0, 1.0], k, exponential);
            let score = ndcg(&lists, &truth, Some(k), gain).unwrap()[0];
            assert!(
                close(score, expected),
                "k={k} {gain:?}: {score} != {expected}"
            );
            assert!(close(score, rounded));
        }
    }

    #[test]
    fn gains_coincide_on_binary_relevance() {
        let (lists, truth) = single_user(&[3, 1, 2], &[1, 2, 9], None);
        let linear = ndcg(&lists, &truth, Some(3), Gain::Linear).unwrap();
        let exponential = ndcg(&lists, &truth, Some(3), Gain::Exponential).unwrap();
        assert!(close(linear[0], exponential[0]));
    }

    #[test]
    fn short_lists_are_not_padded() {
        let (lists, truth) = single_user(&[7, 1], &[1, 4], None);
        let score = ndcg(&lists, &truth, Some(5), Gain::Linear).unwrap()[0];
        // One hit at position 2 of a list of two; two relevant items, so the ideal has
        // them at positions 1 and 2, whatever the cutoff.
        let expected = (1.0 / 3f64.log2()) / (1.0 + 1.0 / 3f64.log2());
        assert!(close(score, expected));
        assert!(close(score, 0.386_853));
    }

    #[test]
    fn perfect_and_useless_rankings_are_the_bounds() {
        let (lists, truth) = single_user(&[2, 0, 1], &[0, 1, 2], Some(&[2.0, 1.0, 3.0]));
        assert!(close(
            ndcg(&lists, &truth, None, Gain::Exponential).unwrap()[0],
            1.0
        ));
        let (lists, truth) = single_user(&[5, 6], &[0], None);
        assert_eq!(ndcg(&lists, &truth, Some(2), Gain::Linear).unwrap()[0], 0.0);
    }

    #[test]
    fn users_are_scored_independently() {
        // user 0: hit at position 2; user 1: no judged item; user 2: judged but no list.
        let lists = RankedLists::new(&[0, 0, 1], &[1, 2, 1], None, 3).unwrap();
        let truth = GroundTruth::new(&[0, 2], &[2, 5], None, 3).unwrap();
        let scores = ndcg(&lists, &truth, Some(2), Gain::Linear).unwrap();
        assert!(close(scores[0], 1.0 / 3f64.log2()));
        assert!(scores[1].is_nan());
        assert_eq!(scores[2], 0.0);
    }

    #[test]
    fn a_user_without_a_list_scores_positive_zero() {
        // No recommendation at all: the achieved DCG is a sum of no terms.
        let lists = RankedLists::new(&[], &[], None, 1).unwrap();
        let truth = GroundTruth::new(&[0], &[3], None, 1).unwrap();
        let score = ndcg(&lists, &truth, None, Gain::Linear).unwrap()[0];
        assert!(score == 0.0 && score.is_sign_positive());
    }

    #[test]
    fn zero_graded_items_are_not_relevant() {
        let (lists, truth) = single_user(&[0], &[0], Some(&[0.0]));
        assert!(ndcg(&lists, &truth, None, Gain::Linear).unwrap()[0].is_nan());
    }

    /// 2^1024 is not a finite f64: the gain cannot be computed, so the relevance is
    /// refused instead of turning a perfect ranking into NaN.
    #[test]
    fn relevance_too_large_for_the_exponential_gain_is_rejected() {
        let (lists, truth) = single_user(&[0, 1], &[0, 1], Some(&[1024.0, 1.0]));
        assert_fails!(
            ndcg(&lists, &truth, None, Gain::Exponential),
            "too large for the exponential gain"
        );
        assert!(close(
            ndcg(&lists, &truth, None, Gain::Linear).unwrap()[0],
            1.0
        ));
        let (lists, truth) = single_user(&[0, 1], &[0, 1], Some(&[1023.0, 1.0]));
        assert!(close(
            ndcg(&lists, &truth, None, Gain::Exponential).unwrap()[0],
            1.0
        ));
    }

    /// 2^r - 1 is about r ln 2 for a tiny r: it must not be rounded to "not relevant".
    #[test]
    fn tiny_relevance_keeps_its_exponential_gain() {
        let (lists, truth) = single_user(&[0], &[0], Some(&[1e-17]));
        assert!(close(
            ndcg(&lists, &truth, None, Gain::Exponential).unwrap()[0],
            1.0
        ));
    }

    #[test]
    fn users_without_hits_score_positive_zero() {
        let (lists, truth) = single_user(&[5], &[0], None);
        let score = ndcg(&lists, &truth, None, Gain::Linear).unwrap()[0];
        assert!(score == 0.0 && score.is_sign_positive());
    }

    /// `McSherry` & Najork 2008: with tied scores the DCG is its mean over the orders of
    /// the tied items. Items 0 (score 2), 1 and 2 (score 1, tied); only item 2 is relevant.
    #[test]
    fn tied_scores_give_the_expected_ndcg() {
        let lists = RankedLists::from_scores(&[0; 3], &[0, 1, 2], &[2.0, 1.0, 1.0], 1).unwrap();
        let truth = GroundTruth::new(&[0], &[2], None, 1).unwrap();
        // Order (0, 1, 2) puts the hit at position 3, order (0, 2, 1) at position 2.
        let expected = f64::midpoint(1.0 / 4f64.log2(), 1.0 / 3f64.log2());
        let score = ndcg(&lists, &truth, None, Gain::Linear).unwrap()[0];
        assert!(close(score, expected));
        // Cutoff 2: the hit is within it in one order out of two.
        let expected = f64::midpoint(0.0, 1.0 / 3f64.log2());
        let score = ndcg(&lists, &truth, Some(2), Gain::Linear).unwrap()[0];
        assert!(close(score, expected));
        // Cutoff 1: the tie starts after it.
        assert_eq!(ndcg(&lists, &truth, Some(1), Gain::Linear).unwrap()[0], 0.0);
    }

    /// A list whose ties agree with the relevance is perfect in every order, but the
    /// achieved DCG adds the mean discount of each tie and the ideal one adds the
    /// discounts one by one: the two sums round differently.
    #[test]
    fn a_perfect_list_with_ties_does_not_exceed_one() {
        for n in 1..200u32 {
            let items: Vec<u32> = (0..n).collect();
            let users = vec![0; items.len()];
            let scores = vec![1.0; items.len()];
            let grades = vec![3.0; items.len()];
            let lists = RankedLists::from_scores(&users, &items, &scores, 1).unwrap();
            let truth = GroundTruth::new(&users, &items, Some(&grades), 1).unwrap();
            for k in [None, Some(items.len() / 2 + 1)] {
                let score = ndcg(&lists, &truth, k, Gain::Linear).unwrap()[0];
                assert!(score <= 1.0, "{score} with {n} tied items");
                assert!(close(score, 1.0));
            }
        }
    }

    #[test]
    fn gains_too_large_to_add_are_rejected() {
        // Each gain 2^1023 - 1 is finite, but three of them, discounted, add up to
        // (1 + 0.63 + 0.5) * 2^1023, which is not: the user used to be silently dropped.
        let (lists, truth) = single_user(&[0, 1, 2], &[0, 1, 2], Some(&[1023.0; 3]));
        assert_fails!(
            ndcg(&lists, &truth, None, Gain::Exponential),
            "too large to be added up"
        );
        let (lists, truth) = single_user(&[0, 1, 2], &[0, 1, 2], Some(&[1e308; 3]));
        assert_fails!(
            ndcg(&lists, &truth, None, Gain::Linear),
            "too large to be added up"
        );
        // Two such gains still fit: (1 + 0.63) * 2^1023 is finite.
        let (lists, truth) = single_user(&[0, 1], &[0, 1], Some(&[1023.0; 2]));
        assert_eq!(
            ndcg(&lists, &truth, None, Gain::Exponential).unwrap(),
            vec![1.0]
        );
    }

    #[test]
    fn invalid_arguments_are_rejected() {
        let (lists, truth) = single_user(&[0], &[0], None);
        assert_fails!(
            ndcg(&lists, &truth, Some(0), Gain::Linear),
            "k must be a positive integer"
        );
        let other = GroundTruth::new(&[1], &[0], None, 2).unwrap();
        assert_fails!(
            ndcg(&lists, &other, None, Gain::Linear),
            "describe 1 users but the ground truth describes 2"
        );
    }
}
