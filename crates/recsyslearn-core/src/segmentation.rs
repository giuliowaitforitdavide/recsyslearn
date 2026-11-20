//! Segmentation of items (or users) into groups by how much they interact.
//!
//! The functions work on one side of the interaction table at a time: pass the item
//! codes to segment items, the user codes to segment users. Entities are always
//! sorted by decreasing number of interactions; the literature does not say how ties
//! are ordered, so they are broken by ascending code, which makes every segmentation
//! deterministic. Group `0` is the most popular (or most active) one.
//!
//! # By cumulative share of interactions
//!
//! Short head / long tail (Abdollahpouri, Burke & Mobasher 2019: the head items
//! account for 80% of the ratings) and head / mid / tail (Abdollahpouri et al. 2021:
//! 20% / 60% / 20% of the ratings). The first group is the smallest set of most
//! popular entities whose interactions reach the first proportion of the total, the
//! second group continues until the first two proportions are reached, and so on: an
//! entity belongs to group `g` when the interactions of the entities before it reach
//! `g` of the cumulative thresholds.
//!
//! # By share of the population
//!
//! The first group is the top `ceil(p_1 n)` entities, e.g. the 5% most active users
//! (Li et al. 2021) or the 20% most popular items (Abdollahpouri et al. 2019).

use rayon::prelude::*;

use crate::error::{
    Error, Result, check_cardinality, check_codes, check_indexable, check_same_len, code, filled,
};

/// Number of rows in which each of the `n` entities appears.
///
/// # Errors
///
/// Fails when a code is not below `n`, or `n` is impossibly large.
pub fn interaction_counts(codes: &[u32], n: usize) -> Result<Vec<u64>> {
    check_cardinality("entities", n)?;
    check_codes("entity", codes, n)?;
    let mut counts = filled(0u64, n, "entities")?;
    for &code in codes {
        counts[code as usize] += 1;
    }
    Ok(counts)
}

/// Number of *distinct* counterparts each of the `n` entities interacted with.
///
/// With items as `entities` and users as `counterparts` this is the degree of the
/// item: the number of users who interacted with it, however many times.
///
/// # Errors
///
/// Fails when the columns differ in length, an entity code is not below `n`, or `n` is
/// impossibly large.
pub fn distinct_counts(entities: &[u32], counterparts: &[u32], n: usize) -> Result<Vec<u64>> {
    check_same_len("entity", entities.len(), "counterpart", counterparts.len())?;
    check_cardinality("entities", n)?;
    check_codes("entity", entities, n)?;
    let mut pairs: Vec<u64> = entities
        .iter()
        .zip(counterparts)
        .map(|(&entity, &counterpart)| (u64::from(entity) << 32) | u64::from(counterpart))
        .collect();
    pairs.par_sort_unstable();
    pairs.dedup();
    let mut counts = filled(0u64, n, "entities")?;
    for pair in pairs {
        counts[(pair >> 32) as usize] += 1;
    }
    Ok(counts)
}

/// Each count divided by `total`.
///
/// # Errors
///
/// Fails when `total` is zero.
pub fn shares(counts: &[u64], total: u64) -> Result<Vec<f64>> {
    if total == 0 {
        return Err(Error::invalid("there are no interactions to share out"));
    }
    Ok(counts
        .iter()
        .map(|&count| count as f64 / total as f64)
        .collect())
}

/// Most decimal places with which the thresholds are computed exactly: a cumulative
/// proportion times a 64-bit total then fits in 128 bits.
const EXACT_DECIMALS: usize = 19;

/// `proportion` as the decimal number it is written as: its digits as an integer, and
/// how many of them follow the decimal point.
///
/// Rust prints a float as the shortest decimal that reads back as the same float, so
/// `0.1` is `(1, 1)`, not the binary fraction closest to one tenth. `None` when there
/// are more decimal places than [`EXACT_DECIMALS`].
fn decimal(proportion: f64) -> Option<(u128, usize)> {
    let written = format!("{proportion}");
    let (whole, fraction) = written.split_once('.').unwrap_or((&written, ""));
    if fraction.len() > EXACT_DECIMALS {
        return None;
    }
    let digits = format!("{whole}{fraction}").parse().ok()?;
    Some((digits, fraction.len()))
}

/// The cumulative proportions times `total`, rounded up, computed on integers.
///
/// `None` when a proportion has too many decimal places for that.
fn exact_thresholds(proportions: &[f64], total: u64) -> Option<Vec<u64>> {
    let parts = proportions
        .iter()
        .map(|&proportion| decimal(proportion))
        .collect::<Option<Vec<_>>>()?;
    let places = parts.iter().map(|&(_, places)| places).max()?;
    let unit = 10u128.pow(u32::try_from(places).ok()?);
    let mut cumulative = 0u128;
    parts
        .iter()
        .map(|&(digits, own)| {
            // Every proportion over the same denominator, `unit`.
            cumulative += digits * 10u128.pow(u32::try_from(places - own).ok()?);
            let scaled = cumulative.checked_mul(u128::from(total))?;
            Some(u64::try_from(scaled.div_ceil(unit)).unwrap_or(u64::MAX))
        })
        .collect()
}

/// The same in floating point, for proportions with more decimal places than the
/// exact arithmetic holds.
///
/// Sums such as `0.1 + 0.2` are not exact in floating point: a threshold that lands
/// within rounding error of an integer (a few units in the last place per proportion
/// summed, whatever the scale) is taken to be that integer before it is rounded up.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a positive float no larger than about `total`; the cast saturates"
)]
fn rounded_thresholds(proportions: &[f64], total: u64) -> Vec<u64> {
    let rounding = 2.0 * f64::EPSILON * (proportions.len() as f64 + 2.0);
    let mut cumulative = 0.0;
    proportions
        .iter()
        .map(|p| {
            cumulative += p;
            let threshold = cumulative * total as f64;
            let nearest = threshold.round();
            // Never down to zero: a positive share, however small, takes one entity.
            let snapped = nearest > 0.0 && (threshold - nearest).abs() <= rounding * nearest;
            (if snapped { nearest } else { threshold.ceil() }) as u64
        })
        .collect()
}

/// Cumulative proportions of `total`, each rounded up to an integer, without the last
/// one (always `total`).
///
/// Counts are integers, so "reaching a share of the total" is reaching the first
/// integer that is not below it. Proportions are read as the decimal numbers they are
/// written as (`0.1 + 0.2` is exactly `0.3`) and the thresholds are computed on
/// integers, exactly at any scale. Only a proportion with more than
/// [`EXACT_DECIMALS`] decimal places is handled in floating point, see
/// [`rounded_thresholds`].
fn thresholds(proportions: &[f64], total: u64) -> Result<Vec<u64>> {
    if proportions.is_empty() {
        return Err(Error::invalid(
            "proportions must contain at least one value",
        ));
    }
    if proportions.iter().any(|p| !p.is_finite() || *p <= 0.0) {
        return Err(Error::invalid(
            "proportions must be finite and strictly positive",
        ));
    }
    check_cardinality("groups", proportions.len())?;
    let sum: f64 = proportions.iter().sum();
    if (sum - 1.0).abs() > 1e-9 {
        return Err(Error::invalid(format!(
            "proportions must sum to 1, but sum to {sum}"
        )));
    }
    let leading = &proportions[..proportions.len() - 1];
    Ok(exact_thresholds(leading, total).unwrap_or_else(|| rounded_thresholds(leading, total)))
}

/// Entities by decreasing count, ties by ascending code.
fn by_decreasing_count(counts: &[u64]) -> Result<Vec<u32>> {
    check_indexable("the entities to segment", counts.len())?;
    let mut order: Vec<u32> = (0..code(counts.len())).collect();
    order.par_sort_unstable_by(|&a, &b| {
        counts[b as usize]
            .cmp(&counts[a as usize])
            .then_with(|| a.cmp(&b))
    });
    Ok(order)
}

/// Group of every entity when the groups take the given shares of the *interactions*.
///
/// # Errors
///
/// Fails when the proportions are not strictly positive or do not sum to 1, or there
/// are more entities than a `u32` can index.
pub fn by_cumulative_interactions(counts: &[u64], proportions: &[f64]) -> Result<Vec<u32>> {
    let total = counts
        .iter()
        .try_fold(0u64, |total, &count| total.checked_add(count))
        .ok_or_else(|| Error::invalid("the number of interactions does not fit in 64 bits"))?;
    let thresholds = thresholds(proportions, total)?;
    let mut groups = vec![0u32; counts.len()];
    let mut before = 0u64;
    let mut group = 0usize;
    for entity in by_decreasing_count(counts)? {
        while group < thresholds.len() && before >= thresholds[group] {
            group += 1;
        }
        groups[entity as usize] = code(group);
        before += counts[entity as usize];
    }
    Ok(groups)
}

/// Group of every entity when the groups take the given shares of the *population*.
///
/// # Errors
///
/// Fails when the proportions are not strictly positive or do not sum to 1, or there
/// are more entities than a `u32` can index.
pub fn by_interaction_count(counts: &[u64], proportions: &[f64]) -> Result<Vec<u32>> {
    let boundaries = thresholds(proportions, counts.len() as u64)?;
    let mut groups = vec![0u32; counts.len()];
    let mut group = 0usize;
    for (position, entity) in by_decreasing_count(counts)?.into_iter().enumerate() {
        while group < boundaries.len() && position as u64 >= boundaries[group] {
            group += 1;
        }
        groups[entity as usize] = code(group);
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::assert_fails;

    #[test]
    fn counts_rows_per_entity() {
        assert_eq!(
            interaction_counts(&[2, 0, 2, 2], 4).unwrap(),
            vec![1, 0, 3, 0]
        );
        assert_fails!(
            interaction_counts(&[4], 4),
            "entity code 4 is out of bounds"
        );
    }

    #[test]
    fn distinct_counts_need_columns_of_the_same_length() {
        assert!(matches!(
            distinct_counts(&[0, 1], &[5], 2),
            Err(Error::LengthMismatch { .. })
        ));
    }

    #[test]
    fn distinct_counts_reject_entities_out_of_bounds() {
        assert!(matches!(
            distinct_counts(&[0, 3], &[5, 5], 3),
            Err(Error::OutOfBounds { what: "entity", .. })
        ));
    }

    #[test]
    fn distinct_counts_ignore_repeated_pairs() {
        // Item 0: users 5, 5, 6 -> 2 distinct; item 1: user 5 -> 1.
        let counts = distinct_counts(&[0, 0, 1, 0], &[5, 5, 5, 6], 3).unwrap();
        assert_eq!(counts, vec![2, 1, 0]);
    }

    #[test]
    fn shares_of_nothing_are_rejected_not_divided_by_zero() {
        assert_fails!(shares(&[0, 0], 0), "no interactions to share out");
    }

    #[test]
    fn shares_divide_by_the_total() {
        assert_eq!(shares(&[1, 3], 4).unwrap(), vec![0.25, 0.75]);
        assert_fails!(shares(&[], 0), "no interactions to share out");
    }

    #[test]
    fn head_is_the_smallest_set_reaching_its_share() {
        // The two most popular items account for exactly 80% of 100 interactions.
        let groups = by_cumulative_interactions(&[50, 30, 10, 5, 5], &[0.8, 0.2]).unwrap();
        assert_eq!(groups, vec![0, 0, 1, 1, 1]);
        // 50 + 25 = 75% < 80%: the item that crosses the threshold is in the head.
        let groups = by_cumulative_interactions(&[50, 25, 15, 5, 5], &[0.8, 0.2]).unwrap();
        assert_eq!(groups, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn input_order_does_not_matter() {
        let groups = by_cumulative_interactions(&[5, 50, 5, 30, 10], &[0.8, 0.2]).unwrap();
        assert_eq!(groups, vec![1, 0, 1, 0, 1]);
    }

    #[test]
    fn head_mid_tail() {
        // Thresholds at 20 and 80 interactions out of 100.
        let counts = [30, 20, 20, 10, 10, 5, 5];
        let groups = by_cumulative_interactions(&counts, &[0.2, 0.6, 0.2]).unwrap();
        assert_eq!(groups, vec![0, 1, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn ties_are_broken_by_ascending_code() {
        let groups = by_cumulative_interactions(&[1, 1, 1, 1], &[0.5, 0.5]).unwrap();
        assert_eq!(groups, vec![0, 0, 1, 1]);
        let groups = by_interaction_count(&[1, 1, 1, 1], &[0.5, 0.5]).unwrap();
        assert_eq!(groups, vec![0, 0, 1, 1]);
    }

    #[test]
    fn thresholds_are_robust_to_floating_point_sums() {
        // 0.1 + 0.2 = 0.30000000000000004 must still cut after exactly 3 of 10.
        let groups = by_cumulative_interactions(&[1; 10], &[0.1, 0.2, 0.7]).unwrap();
        assert_eq!(groups, vec![0, 1, 1, 2, 2, 2, 2, 2, 2, 2]);
        let groups = by_interaction_count(&[1; 10], &[0.1, 0.2, 0.7]).unwrap();
        assert_eq!(groups, vec![0, 1, 1, 2, 2, 2, 2, 2, 2, 2]);
    }

    /// The threshold `0.8 * 1_000_000_004 = 800_000_003.2` is not an integer: the first
    /// item (`800_000_003` interactions) does not reach it, so the second is in the head
    /// too. The same shape at small scale is `[83, 20, 1]`.
    #[test]
    fn thresholds_keep_their_fractional_part_at_large_scale() {
        let groups = by_cumulative_interactions(&[83, 20, 1], &[0.8, 0.2]).unwrap();
        assert_eq!(groups, vec![0, 0, 1]);
        let counts = [800_000_003, 200_000_000, 1];
        let groups = by_cumulative_interactions(&counts, &[0.8, 0.2]).unwrap();
        assert_eq!(groups, vec![0, 0, 1]);
        // 0.33 * 100_000_097 = 33_000_032.01
        let counts = [33_000_032, 33_000_032, 33_000_032, 1_000_001];
        let groups = by_cumulative_interactions(&counts, &[0.33, 0.67]).unwrap();
        assert_eq!(groups, vec![0, 0, 1, 1]);
    }

    /// Proportions are read as the decimal numbers they are written as, and the
    /// thresholds are computed on integers: no rounding at any scale.
    #[test]
    fn thresholds_are_exact_whatever_the_total() {
        // 0.8000001 * 2_000_000_005 = 1_600_000_204.0000005: the first entity, with
        // 1_600_000_204 interactions, does not reach it, so the second is in the head.
        let counts = [1_600_000_204, 399_999_801];
        let groups = by_cumulative_interactions(&counts, &[0.800_000_1, 0.199_999_9]).unwrap();
        assert_eq!(groups, vec![0, 0]);
        // Total 808_777_006_829_317. The thresholds are 0.6, 0.8 and 0.9 of it; the
        // third is 727_899_306_146_385.3, which the first entity misses by 0.3: the
        // second entity is in the third group, not in the fourth.
        let counts = [727_899_306_146_385, 80_877_700_682_932];
        let groups = by_cumulative_interactions(&counts, &[0.6, 0.2, 0.1, 0.1]).unwrap();
        assert_eq!(groups, vec![0, 2]);
        // With a round total, 10^15, the third threshold is the integer 9 * 10^14 and
        // the first entity reaches it: the second is in the last group.
        let counts = [900_000_000_000_000, 100_000_000_000_000];
        let groups = by_cumulative_interactions(&counts, &[0.6, 0.2, 0.1, 0.1]).unwrap();
        assert_eq!(groups, vec![0, 3]);
        // By count: ceil(0.2000001 * 10_000_000) = ceil(2_000_001) = 2_000_001.
        let groups = by_interaction_count(&vec![1; 10_000_000], &[0.200_000_1, 0.799_999_9]);
        let groups = groups.unwrap();
        assert_eq!(
            groups.iter().filter(|&&group| group == 0).count(),
            2_000_001
        );
    }

    /// A proportion with more decimal places than the exact arithmetic holds falls
    /// back to floating point, where a threshold within rounding error of an integer
    /// is that integer.
    #[test]
    fn proportions_with_many_decimals_fall_back_to_floating_point() {
        let third = 1.0 / 3.0; // 0.3333333333333333 as written; fine for the exact path
        let groups = by_interaction_count(&[1; 3], &[third, third, third]).unwrap();
        assert_eq!(groups, vec![0, 1, 2]);
        // 1e-25 has 25 decimal places: of 3 entities it still takes one.
        let groups = by_interaction_count(&[3, 1, 1], &[1e-25, 1.0]).unwrap();
        assert_eq!(groups, vec![0, 1, 1]);
        let groups = by_cumulative_interactions(&[3, 1, 1], &[1e-25, 1.0]).unwrap();
        assert_eq!(groups, vec![0, 1, 1]);
        // 0.1 + 0.2 next to a proportion of 25 decimal places: in floating point the
        // second threshold is 3.0000000000000004, and is still 3. The third group,
        // 1e-25 of 10 entities, starts and ends there: it is empty.
        let long = [0.1, 0.2, 1e-25, 0.7];
        assert_eq!(
            by_interaction_count(&[1; 10], &long).unwrap(),
            vec![0, 1, 1, 3, 3, 3, 3, 3, 3, 3]
        );
    }

    #[test]
    fn counts_too_large_to_sum_are_rejected() {
        let half = 1u64 << 63;
        assert_fails!(
            by_cumulative_interactions(&[half, half], &[0.8, 0.2]),
            "does not fit in 64 bits"
        );
    }

    #[test]
    fn impossible_cardinalities_are_rejected() {
        assert_fails!(
            interaction_counts(&[], usize::MAX),
            "entities cannot be identified"
        );
        assert_fails!(
            distinct_counts(&[], &[], usize::MAX),
            "entities cannot be identified"
        );
    }

    #[test]
    fn a_single_group_contains_everything() {
        assert_eq!(
            by_cumulative_interactions(&[3, 1], &[1.0]).unwrap(),
            vec![0, 0]
        );
        assert_eq!(by_interaction_count(&[3, 1], &[1.0]).unwrap(), vec![0, 0]);
    }

    #[test]
    fn top_group_is_a_share_of_the_population() {
        let groups = by_interaction_count(&[9, 7, 5, 3, 1], &[0.2, 0.8]).unwrap();
        assert_eq!(groups, vec![0, 1, 1, 1, 1]);
        // ceil(0.05 * 10) = 1: the top group is never empty.
        let counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let groups = by_interaction_count(&counts, &[0.05, 0.95]).unwrap();
        assert_eq!(groups, vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 0]);
    }

    #[test]
    fn population_shares_round_up() {
        // n = 7 and 20/60/20: boundaries at ceil(1.4) = 2 and ceil(5.6) = 6.
        let counts = [70, 60, 50, 40, 30, 20, 10];
        let groups = by_interaction_count(&counts, &[0.2, 0.6, 0.2]).unwrap();
        assert_eq!(groups, vec![0, 0, 1, 1, 1, 1, 2]);
    }

    #[test]
    fn empty_input_gives_empty_output() {
        assert!(
            by_cumulative_interactions(&[], &[0.8, 0.2])
                .unwrap()
                .is_empty()
        );
        assert!(by_interaction_count(&[], &[0.8, 0.2]).unwrap().is_empty());
    }

    #[test]
    fn proportions_sum_to_one_within_a_billionth() {
        // Documented tolerance on the sum: 1e-9.
        assert!(by_interaction_count(&[1, 2], &[0.8, 0.2 + 5e-10]).is_ok());
        assert_fails!(
            by_interaction_count(&[1, 2], &[0.8, 0.2 + 5e-9]),
            "must sum to 1"
        );
    }

    #[test]
    fn a_positive_share_is_never_rounded_down_to_nothing() {
        // 1e-16 of 3 entities (or of 5 interactions) is far below one, but the first
        // group must still take the first entity: ceil(3e-16) = 1.
        let tiny = [1e-16, 1.0 - 1e-16];
        assert_eq!(
            by_interaction_count(&[3, 1, 1], &tiny).unwrap(),
            vec![0, 1, 1]
        );
        assert_eq!(
            by_cumulative_interactions(&[3, 1, 1], &tiny).unwrap(),
            vec![0, 1, 1]
        );
    }

    #[test]
    fn a_later_group_can_be_empty() {
        // n = 2 and 30/30/40: boundaries at ceil(0.6) = 1 and ceil(1.2) = 2, so the
        // third group receives nothing.
        assert_eq!(
            by_interaction_count(&[5, 3], &[0.3, 0.3, 0.4]).unwrap(),
            vec![0, 1]
        );
    }

    #[test]
    fn invalid_proportions_are_rejected() {
        let cases: [(&[f64], &str); 5] = [
            (&[], "at least one value"),
            (&[0.5, 0.4], "must sum to 1"),
            (&[1.2, -0.2], "finite and strictly positive"),
            (&[0.0, 1.0], "finite and strictly positive"),
            (&[f64::NAN, 1.0], "finite and strictly positive"),
        ];
        for (proportions, text) in cases {
            assert_fails!(by_cumulative_interactions(&[1, 2], proportions), text);
            assert_fails!(by_interaction_count(&[1, 2], proportions), text);
        }
    }
}
