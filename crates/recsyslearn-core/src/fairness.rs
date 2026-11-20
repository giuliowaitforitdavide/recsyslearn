//! Group fairness: how the benefit of being recommended is spread over groups.
//!
//! The module follows the probabilistic framework of Amigó, Deldjoo, Mizzaro &
//! Bellogín (2023). Every recommended pair receives a benefit `theta(u, i)`:
//!
//! * *exposure* `phi(u, i)`: the attention the position of `i` in the list of `u`
//!   receives, either `1 / log2(1 + position)` ([`Exposure::Logarithmic`]) or `1`
//!   for every position within the cutoff ([`Exposure::Flat`]);
//! * *effectiveness* `phi(u, i) * psi(u, i)`, where the utility `psi` is the relevance
//!   of the pair in the ground truth.
//!
//! The benefit is normalised over the whole user x item space,
//! `P(u, i) = theta(u, i) / sum theta`, and group probabilities are marginals of `P`.
//! On top of that distribution:
//!
//! ```text
//! entropy            H(P)      = - sum_x P(x) log2 P(x)
//! inequity   (Eq. 1) KL(P||Q)  =   sum_x P(x) log2( P(x) / Q(x) )
//! dependence (Eq. 2) I(X; Y)   =   sum_{x,y} P(x, y) log2( P(x, y) / (P(x) P(y)) )
//! ```
//!
//! with the convention `0 log 0 = 0`. All quantities are measured in bits.

use crate::accuracy::log2_discounts;
use crate::error::{Error, Result, check_cardinality, check_codes, code, filled};
use crate::lists::{GroundTruth, RankedLists};

/// Attention received by a position of the recommendation list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Exposure {
    /// `1 / log2(1 + position)`: the rank discount of DCG.
    #[default]
    Logarithmic,
    /// `1` for every position within the cutoff.
    Flat,
}

/// How one side (users or items) of the recommendations is split into blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Partition<'a> {
    /// A single block: the side is marginalised out.
    Whole,
    /// One block per individual.
    Singletons,
    /// Block of every individual, indexed by its code, out of `n_groups` blocks.
    Groups {
        /// Group code of each user (or item).
        codes: &'a [u32],
        /// Number of groups.
        n_groups: usize,
    },
}

impl Partition<'_> {
    fn n_blocks(&self, n_individuals: usize) -> usize {
        match self {
            Self::Whole => 1,
            Self::Singletons => n_individuals,
            Self::Groups { n_groups, .. } => *n_groups,
        }
    }

    fn check(&self, side: &'static str, n_individuals: usize) -> Result<()> {
        if let Self::Groups { codes, n_groups } = self {
            if codes.len() < n_individuals {
                return Err(Error::invalid(format!(
                    "{} {side}s have no group (got {} group codes for {n_individuals} {side}s)",
                    n_individuals - codes.len(),
                    codes.len(),
                )));
            }
            check_codes("group", codes, *n_groups)?;
        }
        Ok(())
    }

    fn block(&self, individual: usize) -> usize {
        match self {
            Self::Whole => 0,
            Self::Singletons => individual,
            Self::Groups { codes, .. } => codes[individual] as usize,
        }
    }
}

/// Joint distribution of the benefit over (user block, item block).
///
/// Only the cells that receive some benefit are stored, so the memory needed grows with
/// the number of recommendations, not with users x groups.
#[derive(Debug, Clone, PartialEq)]
pub struct BenefitDistribution {
    users: Vec<f64>,
    items: Vec<f64>,
    /// `(user block, item block, probability)`, sorted, positive probabilities only.
    cells: Vec<(u32, u32, f64)>,
}

/// Up to this many cells, the joint is accumulated in a dense table, which is faster
/// than sorting the recommendations by block.
const DENSE_CELLS: usize = 1 << 20;

/// Add up the benefits that fall in the same `(user block, item block)` cell and keep
/// the cells with a positive total, sorted by block.
fn joint_cells(
    benefits: impl Iterator<Item = (u32, u32, f64)>,
    n_user_blocks: usize,
    n_item_blocks: usize,
) -> Vec<(u32, u32, f64)> {
    let dense = n_user_blocks
        .checked_mul(n_item_blocks)
        .filter(|&cells| cells <= DENSE_CELLS);
    if let Some(n_cells) = dense {
        let mut table = vec![0.0f64; n_cells];
        for (row, column, benefit) in benefits {
            table[row as usize * n_item_blocks + column as usize] += benefit;
        }
        table
            .into_iter()
            .enumerate()
            .filter(|&(_, mass)| mass > 0.0)
            .map(|(cell, mass)| (code(cell / n_item_blocks), code(cell % n_item_blocks), mass))
            .collect()
    } else {
        let mut entries: Vec<(u32, u32, f64)> =
            benefits.filter(|&(_, _, benefit)| benefit > 0.0).collect();
        // Stable: benefits of the same cell are added in the order of the lists.
        entries.sort_by_key(|&(row, column, _)| (row, column));
        entries.dedup_by(|next, kept| {
            let same = (next.0, next.1) == (kept.0, kept.1);
            if same {
                kept.2 += next.2;
            }
            same
        });
        entries
    }
}

impl BenefitDistribution {
    /// Distribute the benefit of the top-`k` recommendations over the blocks.
    ///
    /// With `truth`, the benefit is the effectiveness (exposure times relevance);
    /// without it, the exposure alone.
    ///
    /// # Errors
    ///
    /// Fails when `k` is zero, a partition does not cover every user (or item), both
    /// sides are partitioned into singletons, or the total benefit is zero (nothing
    /// is recommended, or no recommendation is relevant) or not finite.
    pub fn new(
        lists: &RankedLists,
        k: Option<usize>,
        exposure: Exposure,
        truth: Option<&GroundTruth>,
        users: Partition<'_>,
        items: Partition<'_>,
    ) -> Result<Self> {
        if k == Some(0) {
            return Err(Error::invalid("k must be a positive integer"));
        }
        if let Some(truth) = truth.filter(|truth| truth.n_users() != lists.n_users()) {
            return Err(Error::invalid(format!(
                "recommendations describe {} users but the ground truth describes {}",
                lists.n_users(),
                truth.n_users()
            )));
        }
        if users == Partition::Singletons && items == Partition::Singletons {
            return Err(Error::invalid(
                "at least one of users and items must be partitioned into groups",
            ));
        }
        let n_users = lists.n_users();
        let n_items = lists.max_item().map_or(0, |item| item as usize + 1);
        users.check("user", n_users)?;
        items.check("item", n_items)?;

        // Item codes need not be dense: with one block per item, the blocks are the
        // items that are recommended, so that memory follows the recommendations and
        // not the largest code.
        let recommended: Option<Vec<u32>> = (items == Partition::Singletons).then(|| {
            let mut distinct: Vec<u32> = (0..n_users)
                .flat_map(|user| lists.reachable(user, k).iter().copied())
                .collect();
            distinct.sort_unstable();
            distinct.dedup();
            distinct
        });
        let item_block = |item: u32| match &recommended {
            // Every item of a list is in `recommended`, which was built from the lists.
            Some(distinct) => distinct.binary_search(&item).unwrap_or_default(),
            None => items.block(item as usize),
        };
        let n_user_blocks = users.n_blocks(n_users);
        let n_item_blocks = recommended
            .as_ref()
            .map_or_else(|| items.n_blocks(n_items), Vec::len);
        check_cardinality("user groups", n_user_blocks)?;
        check_cardinality("item groups", n_item_blocks)?;
        let discounts = match exposure {
            Exposure::Logarithmic => log2_discounts(lists.max_len()),
            Exposure::Flat => vec![1.0; lists.max_len()],
        };
        let benefits = (0..n_users).flat_map(|user| {
            let row = code(users.block(user));
            let (discounts, item_block) = (&discounts, &item_block);
            // With tied scores, `attention` is the exposure an item is expected to
            // receive over the orders of its tie.
            lists
                .weighted(user, k, discounts)
                .map(move |(item, attention)| {
                    let benefit = match truth {
                        Some(truth) => attention * truth.relevance(user, item),
                        None => attention,
                    };
                    (row, code(item_block(item)), benefit)
                })
        });

        let mut cells = joint_cells(benefits, n_user_blocks, n_item_blocks);

        let total: f64 = cells.iter().map(|&(_, _, mass)| mass).sum();
        if total <= 0.0 {
            return Err(Error::invalid(
                "the total benefit is zero: there are no recommendations, or none of them is relevant",
            ));
        }
        if !total.is_finite() {
            return Err(Error::invalid(
                "the total benefit is not finite: the relevance values are too large to add up",
            ));
        }
        let mut user_marginal = filled(0.0f64, n_user_blocks, "user groups")?;
        let mut item_marginal = filled(0.0f64, n_item_blocks, "item groups")?;
        for (row, column, mass) in &mut cells {
            *mass /= total;
            user_marginal[*row as usize] += *mass;
            item_marginal[*column as usize] += *mass;
        }
        Ok(Self {
            users: user_marginal,
            items: item_marginal,
            cells,
        })
    }

    /// The cells of the joint distribution that have a positive probability, as
    /// `(user block, item block, probability)`, sorted by block.
    #[must_use]
    pub fn cells(&self) -> &[(u32, u32, f64)] {
        &self.cells
    }

    /// Marginal distribution over the user blocks.
    #[must_use]
    pub fn user_marginal(&self) -> &[f64] {
        &self.users
    }

    /// Marginal distribution over the item blocks. With [`Partition::Singletons`] the
    /// blocks are the items found in the top-`k` lists, by ascending code.
    #[must_use]
    pub fn item_marginal(&self) -> &[f64] {
        &self.items
    }

    /// Mutual information, in bits, between the user block and the item block.
    #[must_use]
    pub fn mutual_information(&self) -> f64 {
        let information: f64 = self
            .cells
            .iter()
            .map(|&(row, column, p)| {
                // Dividing twice, rather than by the product of the marginals, cannot
                // underflow: p / p_user is at most 1.
                p * (p / self.users[row as usize] / self.items[column as usize]).log2()
            })
            .sum();
        // Mutual information is non-negative; clamp the rounding noise around zero.
        information.max(0.0)
    }
}

fn check_distribution(name: &str, distribution: &[f64]) -> Result<()> {
    if distribution.iter().any(|p| !p.is_finite() || *p < 0.0) {
        return Err(Error::invalid(format!(
            "{name} must contain finite, non-negative probabilities"
        )));
    }
    let total: f64 = distribution.iter().sum();
    if (total - 1.0).abs() > 1e-6 {
        return Err(Error::invalid(format!(
            "{name} must sum to 1, but sums to {total}"
        )));
    }
    Ok(())
}

/// Shannon entropy, in bits, of a probability distribution.
///
/// It ranges from `0` (all the mass in one group) to `log2(n)` (uniform).
///
/// # Errors
///
/// Fails when `distribution` is not a probability distribution.
pub fn entropy(distribution: &[f64]) -> Result<f64> {
    check_distribution("the distribution", distribution)?;
    let bits: f64 = distribution
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum();
    Ok(bits.max(0.0))
}

/// Shannon entropy divided by its maximum, `log2 n` for `n` groups (Patro et al. 2020).
///
/// It ranges from `0` to `1`; the ratio is capped at `1`, which the quotient of two
/// rounded logarithms could otherwise exceed by one unit in the last place.
///
/// # Errors
///
/// Fails when `distribution` is not a probability distribution or has fewer than two
/// groups (the maximum would be zero).
pub fn normalized_entropy(distribution: &[f64]) -> Result<f64> {
    if distribution.len() < 2 {
        return Err(Error::invalid(
            "normalised entropy needs at least two groups",
        ));
    }
    let bits = entropy(distribution)?;
    Ok((bits / (distribution.len() as f64).log2()).min(1.0))
}

/// Kullback-Leibler divergence, in bits, of `distribution` from parity: the target that
/// gives every group the same probability, `Q(g) = 1 / n`.
///
/// # Errors
///
/// Fails when `distribution` is empty or is not a probability distribution.
pub fn divergence_from_parity(distribution: &[f64]) -> Result<f64> {
    if distribution.is_empty() {
        return Err(Error::invalid("parity needs at least one group"));
    }
    let parity = vec![1.0 / distribution.len() as f64; distribution.len()];
    kl_divergence(distribution, &parity)
}

/// Kullback-Leibler divergence `KL(P || Q)`, in bits, of `distribution` from `target`.
///
/// It is `0` when the two coincide and infinite when `target` gives no mass to a
/// group that receives some.
///
/// # Errors
///
/// Fails when the inputs differ in length or are not probability distributions.
pub fn kl_divergence(distribution: &[f64], target: &[f64]) -> Result<f64> {
    if distribution.len() != target.len() {
        return Err(Error::invalid(format!(
            "the distribution covers {} groups but the target covers {}",
            distribution.len(),
            target.len()
        )));
    }
    check_distribution("the distribution", distribution)?;
    check_distribution("the target distribution", target)?;
    let bits: f64 = distribution
        .iter()
        .zip(target)
        .filter(|&(&p, _)| p > 0.0)
        .map(|(&p, &q)| p * (p / q).log2())
        .sum();
    Ok(bits.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::assert_fails;

    const TOLERANCE: f64 = 1e-6;

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() < TOLERANCE
    }

    /// The 2 x 2 joint as a row-major table.
    fn joint(distribution: &BenefitDistribution) -> [f64; 4] {
        let mut table = [0.0; 4];
        for &(row, column, p) in distribution.cells() {
            table[row as usize * 2 + column as usize] = p;
        }
        table
    }

    /// Users u1 (group A) and u2 (group B); items i1, i2 in G1 and i3, i4 in G2.
    /// Lists: u1 = [i1, i2, i3], u2 = [i1, i3, i4].
    fn example() -> RankedLists {
        RankedLists::new(&[0, 0, 0, 1, 1, 1], &[0, 1, 2, 0, 2, 3], None, 2).unwrap()
    }

    const USER_GROUPS: Partition<'static> = Partition::Groups {
        codes: &[0, 1],
        n_groups: 2,
    };
    const ITEM_GROUPS: Partition<'static> = Partition::Groups {
        codes: &[0, 0, 1, 1],
        n_groups: 2,
    };

    #[test]
    fn logarithmic_exposure_is_normalised_over_all_pairs() {
        // phi = 1, 1/log2(3), 1/2 at positions 1, 2, 3; total mass = 2 * (1.5 + 1/log2(3)).
        let second = 1.0 / 3f64.log2();
        let total = 2.0 * (1.5 + second);
        let distribution = BenefitDistribution::new(
            &example(),
            None,
            Exposure::Logarithmic,
            None,
            USER_GROUPS,
            ITEM_GROUPS,
        )
        .unwrap();
        let expected = [
            (1.0 + second) / total,
            0.5 / total,
            1.0 / total,
            (second + 0.5) / total,
        ];
        for (actual, expected) in joint(&distribution).iter().zip(expected) {
            assert!(close(*actual, expected));
        }
        let items = distribution.item_marginal();
        // Item group 0 receives the first two cells of the joint above, group 1 the others.
        let (first, other) = ((2.0 + second) / total, (second + 1.0) / total);
        assert!(close(items[0], first));
        assert!(close(items[1], other));
        assert!(close(items[0], 0.617_320));
        assert!(close(items[1], 0.382_680));
        assert!(close(distribution.user_marginal()[0], 0.5));
        // H = -sum p log2 p; against the uniform target KL = log2(2) - H.
        let bits = -first * first.log2() - other * other.log2();
        assert!(close(entropy(items).unwrap(), bits));
        assert!(close(entropy(items).unwrap(), 0.959_913));
        assert!(close(
            kl_divergence(items, &[0.5, 0.5]).unwrap(),
            1.0 - bits
        ));
        assert!(close(kl_divergence(items, &[0.5, 0.5]).unwrap(), 0.040_087));
        // I = sum p(x, y) log2(p(x, y) / (p(x) p(y))), with p(x) = 1/2 for both user groups.
        let information: f64 = expected
            .iter()
            .zip([first, other, first, other])
            .map(|(&p, item)| p * (p / (0.5 * item)).log2())
            .sum();
        assert!(close(distribution.mutual_information(), information));
        assert!(close(distribution.mutual_information(), 0.068_270));
    }

    #[test]
    fn flat_exposure_counts_recommendations() {
        let distribution = BenefitDistribution::new(
            &example(),
            None,
            Exposure::Flat,
            None,
            USER_GROUPS,
            ITEM_GROUPS,
        )
        .unwrap();
        let expected = [2.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0];
        for (actual, expected) in joint(&distribution).iter().zip(expected) {
            assert!(close(*actual, expected));
        }
        let items = distribution.item_marginal();
        assert!(close(entropy(items).unwrap(), 1.0));
        assert!(close(kl_divergence(items, &[0.5, 0.5]).unwrap(), 0.0));
        // I = 2 * (1/3) log2((1/3)/(1/4)) + 2 * (1/6) log2((1/6)/(1/4))
        let expected = 2.0 / 3.0 * (4f64 / 3.0).log2() + 1.0 / 3.0 * (2f64 / 3.0).log2();
        assert!(close(distribution.mutual_information(), expected));
        assert!(close(distribution.mutual_information(), 0.081_704));
    }

    #[test]
    fn cutoff_restricts_the_benefit_to_the_top_positions() {
        let distribution = BenefitDistribution::new(
            &example(),
            Some(1),
            Exposure::Logarithmic,
            None,
            Partition::Whole,
            ITEM_GROUPS,
        )
        .unwrap();
        assert_eq!(distribution.item_marginal(), &[1.0, 0.0]);
        assert_eq!(entropy(distribution.item_marginal()).unwrap(), 0.0);
    }

    #[test]
    fn effectiveness_weights_exposure_by_relevance() {
        // Relevant pairs: (u1, i2) with grade 2 and (u2, i4) with grade 1.
        let truth = GroundTruth::new(&[0, 1], &[1, 3], Some(&[2.0, 1.0]), 2).unwrap();
        let distribution = BenefitDistribution::new(
            &example(),
            None,
            Exposure::Logarithmic,
            Some(&truth),
            USER_GROUPS,
            Partition::Whole,
        )
        .unwrap();
        let first = 2.0 / 3f64.log2();
        let second = 0.5;
        let users = distribution.user_marginal();
        assert!(close(users[0], first / (first + second)));
        assert!(close(users[1], second / (first + second)));
    }

    #[test]
    fn singletons_give_the_dependence_on_individuals() {
        // Each user only sees one item group: knowing the user determines the group.
        let lists = RankedLists::new(&[0, 0, 1, 1], &[0, 1, 2, 3], None, 2).unwrap();
        let distribution = BenefitDistribution::new(
            &lists,
            None,
            Exposure::Flat,
            None,
            Partition::Singletons,
            ITEM_GROUPS,
        )
        .unwrap();
        assert!(close(distribution.mutual_information(), 1.0));

        // Both users see the same list: user and item group are independent.
        let lists = RankedLists::new(&[0, 0, 1, 1], &[0, 2, 0, 2], None, 2).unwrap();
        let distribution = BenefitDistribution::new(
            &lists,
            None,
            Exposure::Logarithmic,
            None,
            USER_GROUPS,
            Partition::Singletons,
        )
        .unwrap();
        assert!(close(distribution.mutual_information(), 0.0));
    }

    /// One user per row and one group per item: the joint has as many cells as
    /// recommendations, not users x groups.
    #[test]
    fn large_partitions_do_not_need_a_dense_joint() {
        let n = 200_000u32;
        let users: Vec<u32> = (0..n).collect();
        let lists = RankedLists::new(&users, &users, None, n as usize).unwrap();
        let groups = Partition::Groups {
            codes: &users,
            n_groups: n as usize,
        };
        let distribution = BenefitDistribution::new(
            &lists,
            None,
            Exposure::Flat,
            None,
            Partition::Singletons,
            groups,
        )
        .unwrap();
        // Each user determines the group: I = H = log2(n).
        assert!(close(
            distribution.mutual_information(),
            f64::from(n).log2()
        ));
    }

    #[test]
    fn impossible_numbers_of_groups_are_rejected() {
        let lists = example();
        let huge = Partition::Groups {
            codes: &[0, 1],
            n_groups: usize::MAX / 2 + 1,
        };
        let built = BenefitDistribution::new(&lists, None, Exposure::Flat, None, huge, ITEM_GROUPS);
        assert_fails!(built, "user groups cannot be identified");
        let huge = Partition::Groups {
            codes: &[0, 0, 1, 1],
            n_groups: usize::MAX / 2 + 1,
        };
        let built = BenefitDistribution::new(&lists, None, Exposure::Flat, None, USER_GROUPS, huge);
        assert_fails!(built, "item groups cannot be identified");
    }

    /// More than `DENSE_CELLS` cells: the joint is accumulated by sorting, and must be
    /// the same as with the dense table.
    #[test]
    fn sparse_accumulation_adds_the_benefits_of_a_cell() {
        // Flat exposure with relevance: user 0 gets items 0 (relevant, grade 2) and 1
        // (relevant, grade 1), user 1 gets item 0 (not relevant) and item 2 (grade 1).
        // Both users are in user group 0; items 0 and 1 are in item group 0, item 2 in 5.
        let lists = RankedLists::new(&[0, 0, 1, 1], &[0, 1, 0, 2], None, 2).unwrap();
        let truth = GroundTruth::new(&[0, 0, 1], &[0, 1, 2], Some(&[2.0, 1.0, 1.0]), 2).unwrap();
        let build = |n_groups| {
            BenefitDistribution::new(
                &lists,
                None,
                Exposure::Flat,
                Some(&truth),
                Partition::Groups {
                    codes: &[0, 0],
                    n_groups,
                },
                Partition::Groups {
                    codes: &[0, 0, 5],
                    n_groups,
                },
            )
            .unwrap()
        };
        // Cell (0, 0) receives 2 + 1 of the total 4, cell (0, 5) receives 1; the
        // recommendation that is not relevant adds no cell.
        let expected = [(0, 0, 0.75), (0, 5, 0.25)];
        assert_eq!(build(6).cells(), &expected); // 36 cells: dense table
        assert_eq!(build(1 << 11).cells(), &expected); // 2^22 cells: sorted entries
    }

    /// Two items with the same score, in different groups, and a cutoff of one: each
    /// is expected to receive half of the exposure of the first position.
    #[test]
    fn tied_scores_share_the_exposure() {
        let lists = RankedLists::from_scores(&[0, 0, 0], &[0, 1, 2], &[5.0, 5.0, 1.0], 1).unwrap();
        let users = Partition::Groups {
            codes: &[0],
            n_groups: 1,
        };
        let items = Partition::Groups {
            codes: &[0, 1, 1],
            n_groups: 2,
        };
        let build = |k, exposure| {
            BenefitDistribution::new(&lists, k, exposure, None, users, items).unwrap()
        };
        assert_eq!(build(Some(1), Exposure::Flat).item_marginal(), &[0.5, 0.5]);
        // Whole list, logarithmic exposure: the tied items expect the mean of the
        // exposures of positions 1 and 2, the third item has that of position 3.
        let shared = f64::midpoint(1.0, 1.0 / 3f64.log2());
        let total = 2.0 * shared + 0.5;
        let marginal = build(None, Exposure::Logarithmic);
        assert!(close(marginal.item_marginal()[0], shared / total));
        assert!(close(marginal.item_marginal()[1], (shared + 0.5) / total));
        // One block per item: the item that cannot reach the cutoff is no block.
        let singles = BenefitDistribution::new(
            &lists,
            Some(1),
            Exposure::Flat,
            None,
            users,
            Partition::Singletons,
        )
        .unwrap();
        assert_eq!(singles.item_marginal(), &[0.5, 0.5]);
    }

    #[test]
    fn parity_is_the_uniform_target() {
        // Against (1/2, 1/2): KL = log2(2) - H(1/4, 3/4).
        let bits = -(0.25 * 0.25f64.log2() + 0.75 * 0.75f64.log2());
        assert!(close(
            divergence_from_parity(&[0.25, 0.75]).unwrap(),
            1.0 - bits
        ));
        assert_eq!(
            divergence_from_parity(&[0.25, 0.75]).unwrap(),
            kl_divergence(&[0.25, 0.75], &[0.5, 0.5]).unwrap()
        );
        assert_eq!(divergence_from_parity(&[0.2; 5]).unwrap(), 0.0);
        assert_fails!(
            divergence_from_parity(&[0.5, 0.6]),
            "the distribution must sum to 1"
        );
        assert_fails!(divergence_from_parity(&[]), "at least one group");
    }

    #[test]
    fn normalised_entropy_divides_by_log2_of_the_number_of_groups() {
        // H([1/2, 1/4, 1/4, 0]) = 1.5 bits, out of log2(4) = 2.
        assert!(close(
            normalized_entropy(&[0.5, 0.25, 0.25, 0.0]).unwrap(),
            0.75
        ));
        assert_eq!(normalized_entropy(&[0.5, 0.5]).unwrap(), 1.0);
        // The ratio of two rounded logarithms can exceed 1 by one ulp: it is capped.
        for n in [11usize, 2999] {
            let uniform = vec![1.0 / n as f64; n];
            assert_eq!(normalized_entropy(&uniform).unwrap(), 1.0);
        }
        assert_fails!(normalized_entropy(&[1.0]), "at least two groups");
    }

    #[test]
    fn entropy_and_divergence_follow_their_definitions() {
        assert!(close(entropy(&[0.25; 4]).unwrap(), 2.0));
        assert_eq!(entropy(&[1.0, 0.0]).unwrap(), 0.0);
        // KL([3/4, 1/4] || [1/2, 1/2]) = 3/4 log2(3/2) + 1/4 log2(1/2)
        let expected = 0.75 * 1.5f64.log2() - 0.25;
        assert!(close(
            kl_divergence(&[0.75, 0.25], &[0.5, 0.5]).unwrap(),
            expected
        ));
        assert_eq!(
            kl_divergence(&[0.5, 0.5], &[1.0, 0.0]).unwrap(),
            f64::INFINITY
        );
        assert_eq!(kl_divergence(&[1.0, 0.0], &[1.0, 0.0]).unwrap(), 0.0);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert_fails!(entropy(&[0.5, 0.6]), "must sum to 1");
        assert_fails!(entropy(&[1.5, -0.5]), "finite, non-negative");
        assert_fails!(
            kl_divergence(&[1.0], &[0.5, 0.5]),
            "covers 1 groups but the target covers 2"
        );
        assert_fails!(
            kl_divergence(&[0.5, 0.6], &[0.5, 0.5]),
            "the distribution must sum to 1"
        );
        assert_fails!(
            kl_divergence(&[0.5, 0.5], &[0.9, 0.3]),
            "the target distribution must sum to 1"
        );

        let lists = example();
        let build = |k, truth: Option<&GroundTruth>, users, items| {
            BenefitDistribution::new(&lists, k, Exposure::Flat, truth, users, items)
        };
        assert_fails!(
            build(Some(0), None, USER_GROUPS, ITEM_GROUPS),
            "k must be a positive integer"
        );
        assert_fails!(
            build(None, None, Partition::Singletons, Partition::Singletons),
            "at least one of users and items"
        );
        let short = Partition::Groups {
            codes: &[0, 0],
            n_groups: 1,
        };
        assert_fails!(build(None, None, USER_GROUPS, short), "items have no group");
        let out_of_bounds = Partition::Groups {
            codes: &[0, 3],
            n_groups: 2,
        };
        assert_fails!(
            build(None, None, out_of_bounds, ITEM_GROUPS),
            "group code 3 is out of bounds"
        );
        // No recommendation is relevant: the effectiveness distribution is undefined.
        let truth = GroundTruth::new(&[0], &[3], None, 2).unwrap();
        assert_fails!(
            build(None, Some(&truth), USER_GROUPS, ITEM_GROUPS),
            "the total benefit is zero"
        );
        let empty = RankedLists::new(&[], &[], None, 2).unwrap();
        let nothing = BenefitDistribution::new(
            &empty,
            None,
            Exposure::Flat,
            None,
            USER_GROUPS,
            Partition::Whole,
        );
        assert_fails!(nothing, "the total benefit is zero");
    }

    #[test]
    fn benefits_too_large_to_add_are_rejected() {
        // Two relevant recommendations of 1e308 each: their sum is not a finite float.
        let lists = example();
        let truth = GroundTruth::new(&[0, 1], &[0, 0], Some(&[1e308, 1e308]), 2).unwrap();
        let built = BenefitDistribution::new(
            &lists,
            None,
            Exposure::Flat,
            Some(&truth),
            USER_GROUPS,
            ITEM_GROUPS,
        );
        assert_fails!(built, "the total benefit is not finite");
    }

    #[test]
    fn ground_truth_must_describe_the_same_users() {
        let lists = example();
        let truth = GroundTruth::new(&[0], &[0], None, 3).unwrap();
        let built = BenefitDistribution::new(
            &lists,
            None,
            Exposure::Flat,
            Some(&truth),
            USER_GROUPS,
            ITEM_GROUPS,
        );
        assert_fails!(built, "describe 2 users but the ground truth describes 3");
    }

    #[test]
    fn memory_follows_the_recommendations_not_the_largest_item_code() {
        // One recommendation of the item with the largest possible code: one item block,
        // not four billion.
        let lists = RankedLists::new(&[0], &[u32::MAX], None, 1).unwrap();
        let users = Partition::Groups {
            codes: &[0],
            n_groups: 1,
        };
        let distribution = BenefitDistribution::new(
            &lists,
            None,
            Exposure::Flat,
            None,
            users,
            Partition::Singletons,
        )
        .unwrap();
        assert_eq!(distribution.item_marginal(), &[1.0]);
        assert_eq!(distribution.cells(), &[(0, 0, 1.0)]);
        assert_eq!(distribution.mutual_information(), 0.0);
    }

    #[test]
    fn sparse_item_codes_give_the_same_dependence_as_dense_ones() {
        // The same lists with items {0, 1} and with items {7, 4_000_000_000}.
        let users = Partition::Groups {
            codes: &[0, 1],
            n_groups: 2,
        };
        let information = |items: &[u32]| {
            let lists = RankedLists::new(&[0, 0, 1], items, None, 2).unwrap();
            BenefitDistribution::new(
                &lists,
                None,
                Exposure::Flat,
                None,
                users,
                Partition::Singletons,
            )
            .unwrap()
            .mutual_information()
        };
        assert_eq!(information(&[0, 1, 0]), information(&[7, 4_000_000_000, 7]));
    }
}
