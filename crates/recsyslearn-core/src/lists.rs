//! Columnar inputs regrouped by user.
//!
//! Every routine of the crate works on *codes*. Users are identified by the integers
//! in `0..n_users`, a number the caller declares and memory grows with; items by any
//! `u32`, dense or not (a function that needs the size of the catalog asks for it). The
//! two structures below turn the
//! long `(user, item[, value])` tables into a compressed sparse row layout, so that
//! metrics can iterate over the rows of one user without hashing.

use rayon::prelude::*;

use crate::error::{
    Error, Result, check_cardinality, check_codes, check_indexable, check_same_len, code, filled,
};

/// Offsets of each user's rows after a stable counting sort on the user code,
/// together with the row indices in grouped order.
fn group_rows_by_user(users: &[u32], n_users: usize) -> Result<(Vec<usize>, Vec<u32>)> {
    check_cardinality("users", n_users)?;
    let slots = n_users
        .checked_add(1)
        .ok_or_else(|| Error::invalid("too many users"))?;
    let mut offsets = filled(0usize, slots, "users")?;
    for &user in users {
        offsets[user as usize + 1] += 1;
    }
    for user in 0..n_users {
        offsets[user + 1] += offsets[user];
    }
    let mut cursor = filled(0usize, slots, "users")?;
    cursor.copy_from_slice(&offsets);
    let mut order = vec![0u32; users.len()];
    for (row, &user) in users.iter().enumerate() {
        let slot = &mut cursor[user as usize];
        order[*slot] = code(row);
        *slot += 1;
    }
    Ok((offsets, order))
}

/// Rows of `user` in a table grouped by `offsets`: an empty range past the last user,
/// so that the accessors built on it never panic.
fn rows_of(offsets: &[usize], user: usize) -> std::ops::Range<usize> {
    match (
        offsets.get(user),
        user.checked_add(1).and_then(|next| offsets.get(next)),
    ) {
        (Some(&start), Some(&end)) => start..end,
        _ => 0..0,
    }
}

/// Split `values` into one mutable slice per user, following `offsets`.
///
/// The number of users is declared by the caller, so the vector of slices is reserved
/// fallibly, like every other allocation that an argument sizes.
fn split_by_offsets<'a, T>(mut values: &'a mut [T], offsets: &[usize]) -> Result<Vec<&'a mut [T]>> {
    let mut chunks = Vec::new();
    chunks
        .try_reserve_exact(offsets.len().saturating_sub(1))
        .map_err(|_| Error::invalid("not enough memory for the users"))?;
    for bounds in offsets.windows(2) {
        let (head, tail) = values.split_at_mut(bounds[1] - bounds[0]);
        chunks.push(head);
        values = tail;
    }
    Ok(chunks)
}

fn has_duplicates(sorted: &[u32]) -> bool {
    sorted.windows(2).any(|pair| pair[0] == pair[1])
}

/// How the rows of a recommendation table are turned into ordered lists.
#[derive(Debug, Clone, Copy)]
enum Order<'a> {
    /// The order of the rows is the ranking.
    Rows,
    /// Ascending rank: the best item has the smallest one, and ranks are distinct.
    Rank(&'a [f64]),
    /// Descending score: the best item has the largest one, and scores can be tied.
    Score(&'a [f64]),
}

/// Recommendation lists: for every user, the recommended items from best to worst.
///
/// Lists built from scores can contain *ties*: items of a user with the same score,
/// whose relative order is unknown. Metrics treat every order of the tied items as
/// equally likely and return their expected value (`McSherry` & Najork 2008), which
/// [`RankedLists::weighted`] computes for any metric that adds up a weight per position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedLists {
    offsets: Vec<usize>,
    items: Vec<u32>,
    /// `tied[p]` when the item at `p` has the same score as the one before it in the
    /// list of the same user. Empty when no list has a tie.
    tied: Vec<bool>,
}

impl RankedLists {
    /// Build the lists from a long `(user, item[, rank])` table.
    ///
    /// When `ranks` is given, each user's rows are ordered by ascending rank (the best
    /// item has the smallest rank) and two items of a user cannot have the same rank: a
    /// rank is a position. Otherwise the order of the rows is the ranking. Either way,
    /// formulas use the resulting 1-based *position* of an item in the list, never the
    /// raw rank value.
    ///
    /// # Errors
    ///
    /// Fails when the columns differ in length, a user code is not below `n_users`,
    /// a rank is NaN, two items of a user have the same rank, or a user is recommended
    /// the same item twice.
    pub fn new(
        users: &[u32],
        items: &[u32],
        ranks: Option<&[f64]>,
        n_users: usize,
    ) -> Result<Self> {
        Self::build(
            users,
            items,
            ranks.map_or(Order::Rows, Order::Rank),
            n_users,
        )
    }

    /// Build the lists from a long `(user, item, score)` table.
    ///
    /// Each user's rows are ordered by descending score: the best item has the largest
    /// one. Items of a user with the same score are tied (see [`RankedLists`]); within a
    /// tie they are stored by ascending item code. Infinite scores are allowed.
    ///
    /// # Errors
    ///
    /// Fails when the columns differ in length, a user code is not below `n_users`,
    /// a score is NaN, or a user is recommended the same item twice.
    pub fn from_scores(
        users: &[u32],
        items: &[u32],
        scores: &[f64],
        n_users: usize,
    ) -> Result<Self> {
        Self::build(users, items, Order::Score(scores), n_users)
    }

    fn build(users: &[u32], items: &[u32], order: Order<'_>, n_users: usize) -> Result<Self> {
        check_indexable("recommendations", users.len())?;
        check_same_len("user", users.len(), "item", items.len())?;
        check_codes("user", users, n_users)?;
        match order {
            Order::Rows => {}
            Order::Rank(ranks) => {
                check_same_len("user", users.len(), "rank", ranks.len())?;
                if ranks.iter().any(|rank| rank.is_nan()) {
                    return Err(Error::invalid("recommendations contain a NaN rank"));
                }
            }
            Order::Score(scores) => {
                check_same_len("user", users.len(), "score", scores.len())?;
                if scores.iter().any(|score| score.is_nan()) {
                    return Err(Error::invalid("recommendations contain a NaN score"));
                }
            }
        }

        let (offsets, mut rows) = group_rows_by_user(users, n_users)?;
        let mut sorted_items = vec![0u32; items.len()];
        let mut tied = match order {
            Order::Score(_) => vec![false; items.len()],
            _ => Vec::new(),
        };
        // Without scores there are no flags to write: every user gets an empty slice.
        let flags = if tied.is_empty() {
            let mut none = Vec::new();
            none.try_reserve_exact(n_users)
                .map_err(|_| Error::invalid("not enough memory for the users"))?;
            none.resize_with(n_users, Default::default);
            none
        } else {
            split_by_offsets(&mut tied, &offsets)?
        };
        let failure = split_by_offsets(&mut rows, &offsets)?
            .into_par_iter()
            .zip(split_by_offsets(&mut sorted_items, &offsets)?)
            .zip(flags)
            .map(|((rows, list), flags): ((_, _), &mut [bool])| {
                match order {
                    Order::Rows => {}
                    Order::Rank(ranks) => {
                        rows.sort_unstable_by(|&a, &b| {
                            ranks[a as usize].total_cmp(&ranks[b as usize])
                        });
                        // Sorted and without NaN: a pair that is not increasing is a tie
                        // (which also covers 0.0 and -0.0).
                        let same = rows
                            .windows(2)
                            .any(|pair| ranks[pair[0] as usize] >= ranks[pair[1] as usize]);
                        if same {
                            return Err(Error::invalid(
                                "recommendations give the same rank to two items of a user",
                            ));
                        }
                    }
                    Order::Score(scores) => {
                        rows.sort_unstable_by(|&a, &b| {
                            scores[b as usize]
                                .total_cmp(&scores[a as usize])
                                .then_with(|| items[a as usize].cmp(&items[b as usize]))
                        });
                        for (flag, pair) in flags.iter_mut().skip(1).zip(rows.windows(2)) {
                            // The partial order, not the total one: 0.0 and -0.0 are the same score.
                            *flag = scores[pair[0] as usize].partial_cmp(&scores[pair[1] as usize])
                                == Some(std::cmp::Ordering::Equal);
                        }
                    }
                }
                for (slot, &row) in list.iter_mut().zip(rows.iter()) {
                    *slot = items[row as usize];
                }
                // `rows` is not needed any more: reuse it to look for duplicates.
                rows.copy_from_slice(list);
                rows.sort_unstable();
                if has_duplicates(rows) {
                    return Err(Error::DuplicatePair {
                        table: "recommendations",
                    });
                }
                Ok(())
            })
            // The error of the first user, not of the first thread to find one.
            .find_map_first(Result::err);
        if let Some(error) = failure {
            return Err(error);
        }
        if !tied.contains(&true) {
            tied = Vec::new();
        }

        Ok(Self {
            offsets,
            items: sorted_items,
            tied,
        })
    }

    /// Whether any list has two items with the same score.
    #[must_use]
    pub fn has_ties(&self) -> bool {
        !self.tied.is_empty()
    }

    /// Number of users, including those with an empty list.
    #[must_use]
    pub fn n_users(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Length of the longest list.
    #[must_use]
    pub fn max_len(&self) -> usize {
        self.offsets
            .windows(2)
            .map(|bounds| bounds[1] - bounds[0])
            .max()
            .unwrap_or(0)
    }

    /// Bounds of the rows of `user`, empty for a user that does not exist.
    fn rows(&self, user: usize) -> std::ops::Range<usize> {
        rows_of(&self.offsets, user)
    }

    /// Items recommended to `user`, best first, cut after `k` positions if given.
    ///
    /// A user that is not below [`RankedLists::n_users`] has an empty list.
    #[must_use]
    pub fn top(&self, user: usize, k: Option<usize>) -> &[u32] {
        let list = &self.items[self.rows(user)];
        match k {
            Some(k) if k < list.len() => &list[..k],
            _ => list,
        }
    }

    /// Items that can be among the first `k` of `user`: the top `k`, extended to the end
    /// of the tie that the cutoff falls in. Without ties it is [`RankedLists::top`].
    #[must_use]
    pub fn reachable(&self, user: usize, k: Option<usize>) -> &[u32] {
        let rows = self.rows(user);
        let limit = k.map_or(rows.len(), |k| k.min(rows.len()));
        let mut end = limit;
        if !self.tied.is_empty() && limit > 0 {
            let tied = &self.tied[rows.clone()];
            while end < tied.len() && tied[end] {
                end += 1;
            }
        }
        &self.items[rows][..end]
    }

    /// The items of `user` with the weight each receives from a metric that gives
    /// `weights[p]` to the item at position `p` (0-based) and nothing after `k`.
    ///
    /// Without ties, the item at `p` receives `weights[p]`. Items tied with each other
    /// share the positions of their group: over all the orders of the group, each is
    /// expected to receive the mean of the group's weights, counting as zero the
    /// positions after the cutoff (`McSherry` & Najork 2008). Items that cannot reach
    /// the first `k` positions are not yielded.
    ///
    /// `weights` must cover the positions up to the cutoff, i.e. have at least
    /// `min(k, length of the list)` values; missing ones count as zero.
    #[must_use]
    pub fn weighted<'a>(
        &'a self,
        user: usize,
        k: Option<usize>,
        weights: &'a [f64],
    ) -> Weighted<'a> {
        let rows = self.rows(user);
        let limit = k.map_or(rows.len(), |k| k.min(rows.len()));
        Weighted {
            tied: if self.tied.is_empty() {
                &[]
            } else {
                &self.tied[rows.clone()]
            },
            items: &self.items[rows],
            weights: &weights[..limit.min(weights.len())],
            position: 0,
            group_end: 0,
            weight: 0.0,
        }
    }

    /// Largest item code that appears in any list, if there is one.
    #[must_use]
    pub fn max_item(&self) -> Option<u32> {
        self.items.iter().copied().max()
    }
}

/// Iterator returned by [`RankedLists::weighted`].
#[derive(Debug, Clone)]
pub struct Weighted<'a> {
    items: &'a [u32],
    /// Tie flags of the list, or empty when the lists have no tie.
    tied: &'a [bool],
    /// Weight of every position before the cutoff.
    weights: &'a [f64],
    position: usize,
    /// End of the tie group the current position belongs to.
    group_end: usize,
    /// Expected weight of the items of the current group.
    weight: f64,
}

impl Iterator for Weighted<'_> {
    type Item = (u32, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position == self.group_end {
            // A group that starts after the cutoff cannot receive anything.
            if self.position >= self.weights.len() || self.position >= self.items.len() {
                return None;
            }
            let mut end = self.position + 1;
            while end < self.tied.len() && self.tied[end] {
                end += 1;
            }
            let within = end.min(self.weights.len());
            let total: f64 = self.weights[self.position..within].iter().sum();
            self.weight = total / (end - self.position) as f64;
            self.group_end = end;
        }
        let item = self.items[self.position];
        self.position += 1;
        Some((item, self.weight))
    }
}

/// Held-out relevance judgements: for every user, the relevant items and their gain.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundTruth {
    offsets: Vec<usize>,
    /// Item codes, ascending within each user.
    items: Vec<u32>,
    /// Relevance of the item at the same index.
    relevance: Vec<f64>,
}

impl GroundTruth {
    /// Build the judgements from a long `(user, item[, relevance])` table.
    ///
    /// Without `relevance` every listed pair is relevant with grade `1` (implicit
    /// feedback). With it, the value is the graded relevance of the pair (explicit
    /// feedback); pairs that are not listed have relevance `0`.
    ///
    /// # Errors
    ///
    /// Fails when the columns differ in length, a user code is not below `n_users`,
    /// a relevance is negative or not finite, or a pair is listed twice.
    pub fn new(
        users: &[u32],
        items: &[u32],
        relevance: Option<&[f64]>,
        n_users: usize,
    ) -> Result<Self> {
        check_indexable("ground-truth interactions", users.len())?;
        check_same_len("user", users.len(), "item", items.len())?;
        check_codes("user", users, n_users)?;
        if let Some(relevance) = relevance {
            check_same_len("user", users.len(), "relevance", relevance.len())?;
            if relevance
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(Error::invalid(
                    "relevance must be finite and non-negative for every ground-truth pair",
                ));
            }
        }

        let (offsets, mut order) = group_rows_by_user(users, n_users)?;
        let mut sorted_items = vec![0u32; items.len()];
        let mut sorted_relevance = vec![1.0f64; items.len()];
        let failure = split_by_offsets(&mut order, &offsets)?
            .into_par_iter()
            .zip(split_by_offsets(&mut sorted_items, &offsets)?)
            .zip(split_by_offsets(&mut sorted_relevance, &offsets)?)
            .map(|((rows, user_items), user_relevance)| {
                rows.sort_unstable_by_key(|&row| items[row as usize]);
                for (index, &row) in rows.iter().enumerate() {
                    user_items[index] = items[row as usize];
                    if let Some(relevance) = relevance {
                        user_relevance[index] = relevance[row as usize];
                    }
                }
                if has_duplicates(user_items) {
                    return Err(Error::DuplicatePair {
                        table: "ground-truth interactions",
                    });
                }
                Ok(())
            })
            .find_map_first(Result::err);
        if let Some(error) = failure {
            return Err(error);
        }

        Ok(Self {
            offsets,
            items: sorted_items,
            relevance: sorted_relevance,
        })
    }

    /// Number of users, including those without any judgement.
    #[must_use]
    pub fn n_users(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Relevance of `item` for `user`; `0` when the pair is not in the ground truth,
    /// which includes any user that is not below [`GroundTruth::n_users`].
    #[must_use]
    pub fn relevance(&self, user: usize, item: u32) -> f64 {
        let range = rows_of(&self.offsets, user);
        match self.items[range.clone()].binary_search(&item) {
            Ok(index) => self.relevance[range.start + index],
            Err(_) => 0.0,
        }
    }

    /// Relevance grades of every judged pair, in no particular order.
    pub(crate) fn all_relevances(&self) -> &[f64] {
        &self.relevance
    }

    /// Relevance grades of all the judged items of `user`, in no particular order;
    /// none for a user that is not below [`GroundTruth::n_users`].
    #[must_use]
    pub fn relevances(&self, user: usize) -> &[f64] {
        &self.relevance[rows_of(&self.offsets, user)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::assert_fails;

    #[test]
    fn lists_follow_row_order_without_ranks() {
        let lists = RankedLists::new(&[1, 0, 1, 0], &[7, 3, 5, 9], None, 3).unwrap();
        assert_eq!(lists.n_users(), 3);
        assert_eq!(lists.top(0, None), &[3, 9]);
        assert_eq!(lists.top(1, None), &[7, 5]);
        assert!(lists.top(2, None).is_empty());
        assert_eq!(lists.max_len(), 2);
        assert_eq!(lists.max_item(), Some(9));
    }

    #[test]
    fn lists_are_sorted_by_ascending_rank() {
        let users = [0, 0, 0, 1, 1];
        let items = [10, 11, 12, 20, 21];
        let ranks = [3.0, 1.0, 2.0, 2.0, 1.0];
        let lists = RankedLists::new(&users, &items, Some(&ranks), 2).unwrap();
        assert_eq!(lists.top(0, None), &[11, 12, 10]);
        assert_eq!(lists.top(0, Some(2)), &[11, 12]);
        assert_eq!(lists.top(1, Some(5)), &[21, 20]);
    }

    /// Two items of a user with the same rank have no defined order: the input is
    /// refused rather than ordered by the position of the rows.
    #[test]
    fn tied_ranks_are_rejected() {
        let tied = RankedLists::new(&[0, 0, 0], &[5, 4, 3], Some(&[1.0, 1.0, 0.0]), 1);
        assert_fails!(tied, "the same rank to two items");
        // 0.0 and -0.0 are the same rank.
        let zeros = RankedLists::new(&[0, 0], &[5, 4], Some(&[0.0, -0.0]), 1);
        assert_fails!(zeros, "the same rank to two items");
        // The same rank for two different users is fine.
        assert!(RankedLists::new(&[0, 1], &[5, 4], Some(&[1.0, 1.0]), 2).is_ok());
    }

    #[test]
    fn the_error_reported_is_the_one_of_the_first_user() {
        // The first half of the users have tied ranks, the second half a repeated item:
        // whichever thread finds an error first, the one reported is that of user 0.
        let n = 20_000u32;
        let users: Vec<u32> = (0..n).flat_map(|user| [user, user]).collect();
        let items: Vec<u32> = (0..n)
            .flat_map(|user| if user < n / 2 { [0, 1] } else { [0, 0] })
            .collect();
        let ranks: Vec<f64> = (0..n)
            .flat_map(|user| if user < n / 2 { [1.0, 1.0] } else { [1.0, 2.0] })
            .collect();
        for _ in 0..20 {
            assert_fails!(
                RankedLists::new(&users, &items, Some(&ranks), n as usize),
                "the same rank to two items"
            );
        }
    }

    #[test]
    fn scores_order_the_lists_from_the_largest() {
        // User 0: item 7 (0.9), item 5 (0.1), item 6 (0.5). User 1: one item, -inf.
        let users = [0, 0, 0, 1];
        let scores = [0.9, 0.1, 0.5, f64::NEG_INFINITY];
        let lists = RankedLists::from_scores(&users, &[7, 5, 6, 9], &scores, 2).unwrap();
        assert_eq!(lists.top(0, None), &[7, 6, 5]);
        assert_eq!(lists.top(1, None), &[9]);
        assert!(!lists.has_ties());
        // Without ties, scores and the ranks they imply build the same lists.
        let ranked = RankedLists::new(&users, &[7, 5, 6, 9], Some(&[1.0, 3.0, 2.0, 1.0]), 2);
        assert_eq!(lists, ranked.unwrap());
    }

    #[test]
    fn scores_reject_invalid_input() {
        assert_fails!(
            RankedLists::from_scores(&[0], &[1], &[f64::NAN], 1),
            "NaN score"
        );
        assert!(matches!(
            RankedLists::from_scores(&[0, 0], &[1, 2], &[1.0], 1),
            Err(Error::LengthMismatch { right: "score", .. })
        ));
        assert!(matches!(
            RankedLists::from_scores(&[0, 0], &[1, 1], &[2.0, 1.0], 1),
            Err(Error::DuplicatePair { .. })
        ));
    }

    /// One user, scores 3, 2, 2, 2, 1 for the items 10, 13, 11, 12, 14: the three items
    /// with score 2 are tied and share the positions 2, 3 and 4.
    fn tied_example() -> RankedLists {
        let scores = [3.0, 2.0, 2.0, 2.0, 1.0];
        RankedLists::from_scores(&[0; 5], &[10, 13, 11, 12, 14], &scores, 1).unwrap()
    }

    #[test]
    fn equal_scores_are_ties() {
        let lists = tied_example();
        assert!(lists.has_ties());
        // Within a tie the items are stored by ascending code.
        assert_eq!(lists.top(0, None), &[10, 11, 12, 13, 14]);
        // 0.0 and -0.0 are the same score; the same score for two users is no tie.
        let zeros = RankedLists::from_scores(&[0, 0], &[1, 2], &[0.0, -0.0], 1).unwrap();
        assert!(zeros.has_ties());
        let users = RankedLists::from_scores(&[0, 1], &[1, 2], &[5.0, 5.0], 2).unwrap();
        assert!(!users.has_ties());
    }

    #[test]
    fn tied_items_share_the_weights_of_their_positions() {
        let lists = tied_example();
        let weights = [16.0, 8.0, 4.0, 2.0, 1.0];
        // No cutoff: the tie spans the positions with weights 8, 4 and 2, mean 14/3.
        let shared = 14.0 / 3.0;
        let all: Vec<(u32, f64)> = lists.weighted(0, None, &weights).collect();
        assert_eq!(
            all,
            vec![
                (10, 16.0),
                (11, shared),
                (12, shared),
                (13, shared),
                (14, 1.0)
            ]
        );
        // Cutoff 2: only the first position of the tie is within it, so each of the
        // three items expects 8 / 3; the last item cannot reach the cutoff.
        let cut: Vec<(u32, f64)> = lists.weighted(0, Some(2), &weights).collect();
        assert_eq!(
            cut,
            vec![
                (10, 16.0),
                (11, 8.0 / 3.0),
                (12, 8.0 / 3.0),
                (13, 8.0 / 3.0)
            ]
        );
        // Cutoff 1: the tie starts after it.
        let first: Vec<(u32, f64)> = lists.weighted(0, Some(1), &weights).collect();
        assert_eq!(first, vec![(10, 16.0)]);
        assert_eq!(lists.weighted(1, None, &weights).count(), 0);
    }

    #[test]
    fn items_within_reach_of_the_cutoff() {
        let lists = tied_example();
        assert_eq!(lists.reachable(0, Some(1)), &[10]);
        assert_eq!(lists.reachable(0, Some(2)), &[10, 11, 12, 13]);
        assert_eq!(lists.reachable(0, Some(4)), &[10, 11, 12, 13]);
        assert_eq!(lists.reachable(0, None), &[10, 11, 12, 13, 14]);
        let ranked = RankedLists::new(&[0, 0], &[4, 3], None, 1).unwrap();
        assert_eq!(ranked.reachable(0, Some(1)), &[4]);
        assert!(ranked.reachable(3, Some(1)).is_empty());
    }

    #[test]
    fn users_that_do_not_exist_have_nothing() {
        let lists = RankedLists::new(&[0], &[3], None, 1).unwrap();
        assert!(lists.top(1, None).is_empty());
        assert!(lists.top(usize::MAX, Some(1)).is_empty());
        let truth = GroundTruth::new(&[0], &[3], None, 1).unwrap();
        assert_eq!(truth.relevance(0, 3), 1.0);
        assert_eq!(truth.relevance(1, 3), 0.0);
        assert!(truth.relevances(usize::MAX).is_empty());
    }

    #[test]
    fn ground_truth_rejects_users_out_of_bounds() {
        assert!(matches!(
            GroundTruth::new(&[2], &[1], None, 2),
            Err(Error::OutOfBounds { what: "user", .. })
        ));
    }

    #[test]
    fn impossible_cardinalities_are_rejected() {
        assert_fails!(
            RankedLists::new(&[], &[], None, usize::MAX),
            "users cannot be identified"
        );
        assert_fails!(
            GroundTruth::new(&[], &[], None, usize::MAX),
            "users cannot be identified"
        );
    }

    #[test]
    fn ground_truth_columns_must_have_the_same_length() {
        assert!(matches!(
            GroundTruth::new(&[0, 0], &[1], None, 1),
            Err(Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            GroundTruth::new(&[0, 0], &[1, 2], Some(&[1.0]), 1),
            Err(Error::LengthMismatch { .. })
        ));
    }

    #[test]
    fn lists_reject_invalid_input() {
        assert!(matches!(
            RankedLists::new(&[0, 0], &[1], None, 1),
            Err(Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            RankedLists::new(&[2], &[1], None, 2),
            Err(Error::OutOfBounds { what: "user", .. })
        ));
        assert!(matches!(
            RankedLists::new(&[0, 0], &[1, 1], None, 1),
            Err(Error::DuplicatePair { .. })
        ));
        assert_fails!(
            RankedLists::new(&[0], &[1], Some(&[f64::NAN]), 1),
            "NaN rank"
        );
        assert!(matches!(
            RankedLists::new(&[0, 0], &[1, 2], Some(&[1.0]), 1),
            Err(Error::LengthMismatch { right: "rank", .. })
        ));
    }

    #[test]
    fn ground_truth_looks_up_relevance() {
        let truth = GroundTruth::new(&[1, 0, 1], &[4, 2, 3], Some(&[2.0, 5.0, 0.0]), 2).unwrap();
        assert_eq!(truth.relevance(0, 2), 5.0);
        assert_eq!(truth.relevance(1, 4), 2.0);
        assert_eq!(truth.relevance(1, 3), 0.0);
        assert_eq!(truth.relevance(1, 99), 0.0);
        assert_eq!(truth.relevances(1), &[0.0, 2.0]);
    }

    #[test]
    fn ground_truth_is_binary_without_relevance() {
        let truth = GroundTruth::new(&[0, 0], &[8, 6], None, 1).unwrap();
        assert_eq!(truth.relevance(0, 6), 1.0);
        assert_eq!(truth.relevance(0, 7), 0.0);
    }

    #[test]
    fn ground_truth_rejects_invalid_input() {
        assert!(matches!(
            GroundTruth::new(&[0, 0], &[1, 1], None, 1),
            Err(Error::DuplicatePair { .. })
        ));
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            assert_fails!(
                GroundTruth::new(&[0], &[1], Some(&[bad]), 1),
                "relevance must be finite and non-negative"
            );
        }
    }
}
