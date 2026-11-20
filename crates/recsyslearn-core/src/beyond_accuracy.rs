//! Beyond-accuracy qualities of the recommendations: coverage and novelty.
//!
//! # Catalog coverage
//!
//! Share of the catalog that is recommended to at least one user (Ge,
//! Delgado-Battenfeld & Jannach 2010):
//!
//! ```text
//! coverage@k = | union_u L_u | / |I|
//! ```
//!
//! # Novelty
//!
//! Mean self-information of the recommended items (Zhou et al. 2010; Schedl et al.
//! 2018, Eq. 12). With `pop_i = |U_i| / |U|` the fraction of users who interacted
//! with item `i`,
//!
//! ```text
//! novelty@k(u) = (1 / |L_u|) sum_{i in L_u} -log2(pop_i)
//! ```
//!
//! and the novelty of the system is the mean over users. The papers do not say what
//! the self-information of an item nobody interacted with is (the formula diverges);
//! such an item is given the self-information of an item with a single user,
//! `log2 |U|`, the largest finite value the formula can produce.

use rayon::prelude::*;

use crate::error::{Error, Result, check_cardinality, filled};
use crate::lists::RankedLists;

/// Fraction of the `n_items` catalog items that appear in a top-`k` list.
///
/// # Errors
///
/// Fails when `k` is zero, the catalog is empty or impossibly large, a recommended
/// item is not part of the catalog (its code is not below `n_items`), or items with
/// the same score straddle the cutoff.
pub fn catalog_coverage(lists: &RankedLists, k: Option<usize>, n_items: usize) -> Result<f64> {
    if k == Some(0) {
        return Err(Error::invalid("k must be a positive integer"));
    }
    if n_items == 0 {
        return Err(Error::invalid("the catalog is empty"));
    }
    if let Some(item) = lists.max_item().filter(|&item| item as usize >= n_items) {
        return Err(Error::OutOfBounds {
            what: "item",
            code: item,
            bound: n_items,
        });
    }
    check_cardinality("catalog items", n_items)?;
    // Which items of a tie are within the cutoff depends on their order, and so does the
    // number of distinct items; no published definition of coverage says what to do.
    if lists.has_ties()
        && (0..lists.n_users())
            .any(|user| lists.reachable(user, k).len() > lists.top(user, k).len())
    {
        return Err(Error::invalid(
            "coverage is undefined when items with the same score straddle the cutoff: \
             choose a cutoff that does not split a tie, or give distinct ranks",
        ));
    }
    let mut recommended = filled(false, n_items, "catalog items")?;
    for user in 0..lists.n_users() {
        for &item in lists.top(user, k) {
            recommended[item as usize] = true;
        }
    }
    let distinct = recommended.iter().filter(|&&seen| seen).count();
    Ok(distinct as f64 / n_items as f64)
}

/// Novelty of the top-`k` list of every user; `NaN` for users with an empty list.
///
/// `degrees[i]` is the number of distinct users, out of `n_users`, who interacted
/// with item `i`; items beyond the end of `degrees` have no interactions.
///
/// # Errors
///
/// Fails when `k` is zero, `n_users` is zero, or a degree exceeds `n_users`.
pub fn novelty(
    lists: &RankedLists,
    k: Option<usize>,
    degrees: &[u64],
    n_users: usize,
) -> Result<Vec<f64>> {
    if k == Some(0) {
        return Err(Error::invalid("k must be a positive integer"));
    }
    if n_users == 0 {
        return Err(Error::invalid("the interactions contain no users"));
    }
    if degrees.iter().any(|&degree| degree > n_users as u64) {
        return Err(Error::invalid(
            "an item cannot have interacted with more users than there are",
        ));
    }
    let population = n_users as f64;
    let unseen = population.log2();
    let self_information: Vec<f64> = degrees
        .iter()
        .map(|&degree| (population / degree.max(1) as f64).log2())
        .collect();

    // Every position within the cutoff counts the same; with tied scores, an item
    // counts for the probability that it is within the cutoff.
    let presence = vec![1.0; lists.max_len()];
    let scores = (0..lists.n_users())
        .into_par_iter()
        .map(|user| {
            let length = lists.top(user, k).len();
            if length == 0 {
                return f64::NAN;
            }
            let total: f64 = lists
                .weighted(user, k, &presence)
                .map(|(item, probability)| {
                    probability
                        * self_information
                            .get(item as usize)
                            .copied()
                            .unwrap_or(unseen)
                })
                .sum();
            total / length as f64
        })
        .collect();
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::assert_fails;

    const TOLERANCE: f64 = 1e-6;

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < TOLERANCE
    }

    /// Five users, items A..D = 0..3, lists [A,B], [A,B], [A,C], [B,A], [A,B].
    fn example() -> RankedLists {
        let users = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
        let items = [0, 1, 0, 1, 0, 2, 1, 0, 0, 1];
        RankedLists::new(&users, &items, None, 5).unwrap()
    }

    #[test]
    fn coverage_is_the_recommended_share_of_the_catalog() {
        let lists = example();
        assert_eq!(catalog_coverage(&lists, None, 4).unwrap(), 0.75);
        // At k = 1 only A and B are recommended.
        assert_eq!(catalog_coverage(&lists, Some(1), 4).unwrap(), 0.5);
        assert_eq!(catalog_coverage(&lists, None, 3).unwrap(), 1.0);
        assert_eq!(catalog_coverage(&lists, None, 300).unwrap(), 0.01);
    }

    #[test]
    fn coverage_rejects_invalid_input() {
        let lists = example();
        assert_fails!(
            catalog_coverage(&lists, Some(0), 4),
            "k must be a positive integer"
        );
        assert_fails!(catalog_coverage(&lists, None, 0), "the catalog is empty");
        assert_fails!(
            catalog_coverage(&lists, None, usize::MAX),
            "catalog items cannot be identified"
        );
        assert!(matches!(
            catalog_coverage(&lists, None, 2),
            Err(Error::OutOfBounds { what: "item", .. })
        ));
    }

    /// Zhou et al. 2010: `I_i = log2(u / k_i)` with `u = 5` and degrees 4, 2, 1, 1.
    #[test]
    fn novelty_is_the_mean_self_information() {
        let scores = novelty(&example(), None, &[4, 2, 1, 1], 5).unwrap();
        let (a, b, c) = (1.25f64.log2(), 2.5f64.log2(), 5f64.log2());
        for user in [0, 1, 3, 4] {
            assert!(close(scores[user], f64::midpoint(a, b)));
            assert!(close(scores[user], 0.821_928));
        }
        assert!(close(scores[2], f64::midpoint(a, c)));
        // Four users with lists (A, B) and one with (A, C).
        let mean = (4.0 * f64::midpoint(a, b) + f64::midpoint(a, c)) / 5.0;
        assert!(close(scores.iter().sum::<f64>() / 5.0, mean));
        assert!(close(mean, 0.921_928));
    }

    #[test]
    fn novelty_respects_the_cutoff() {
        let scores = novelty(&example(), Some(1), &[4, 2, 1, 1], 5).unwrap();
        assert!(close(scores[0], 1.25f64.log2()));
        assert!(close(scores[3], 2.5f64.log2()));
    }

    #[test]
    fn items_everyone_knows_carry_no_information() {
        let lists = RankedLists::new(&[0], &[0], None, 1).unwrap();
        assert_eq!(novelty(&lists, None, &[8], 8).unwrap(), vec![0.0]);
    }

    #[test]
    fn unseen_items_are_as_novel_as_an_item_with_one_user() {
        let lists = RankedLists::new(&[0, 0], &[1, 7], None, 1).unwrap();
        // Item 1 has degree zero and item 7 is beyond the known items: both log2(4).
        assert_eq!(novelty(&lists, None, &[3, 0], 4).unwrap(), vec![2.0]);
    }

    #[test]
    fn users_without_recommendations_have_no_novelty() {
        let lists = RankedLists::new(&[1], &[0], None, 2).unwrap();
        let scores = novelty(&lists, None, &[1], 2).unwrap();
        assert!(scores[0].is_nan());
        assert_eq!(scores[1], 1.0);
    }

    /// Items 0 (score 2), 1 and 2 (score 1, tied) with degrees 4, 2 and 1 out of 4 users.
    #[test]
    fn tied_scores_give_the_expected_novelty() {
        let lists = RankedLists::from_scores(&[0; 3], &[0, 1, 2], &[2.0, 1.0, 1.0], 1).unwrap();
        let (a, b, c) = (0.0, 2f64.log2(), 4f64.log2());
        // Whole list: the order of the tie does not matter.
        let scores = novelty(&lists, None, &[4, 2, 1], 4).unwrap();
        assert!(close(scores[0], (a + b + c) / 3.0));
        // Cutoff 2: item 0 and, with probability 1/2 each, item 1 or item 2.
        let scores = novelty(&lists, Some(2), &[4, 2, 1], 4).unwrap();
        assert!(close(scores[0], f64::midpoint(a, f64::midpoint(b, c))));
    }

    /// Scores 2, 1, 1: the order of the tie does not matter for the whole list or for
    /// a cutoff of one, but it decides which item is second.
    #[test]
    fn coverage_refuses_a_tie_that_straddles_the_cutoff() {
        let lists = RankedLists::from_scores(&[0; 3], &[0, 1, 2], &[2.0, 1.0, 1.0], 1).unwrap();
        assert_eq!(catalog_coverage(&lists, None, 4).unwrap(), 0.75);
        assert_eq!(catalog_coverage(&lists, Some(1), 4).unwrap(), 0.25);
        assert_eq!(catalog_coverage(&lists, Some(3), 4).unwrap(), 0.75);
        assert_fails!(
            catalog_coverage(&lists, Some(2), 4),
            "items with the same score straddle the cutoff"
        );
    }

    #[test]
    fn novelty_rejects_invalid_input() {
        let lists = example();
        assert_fails!(
            novelty(&lists, Some(0), &[1], 5),
            "k must be a positive integer"
        );
        assert_fails!(
            novelty(&lists, None, &[1], 0),
            "the interactions contain no users"
        );
        assert_fails!(novelty(&lists, None, &[6], 5), "more users than there are");
    }
}
