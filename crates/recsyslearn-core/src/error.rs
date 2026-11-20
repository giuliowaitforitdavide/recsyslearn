//! Error type shared by every routine of the crate.

use thiserror::Error;

/// Convenient alias for results produced by this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Reasons why an input cannot be evaluated.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Error {
    /// Parallel columns of the same table do not have the same number of rows.
    #[error("columns `{left}` ({left_len} rows) and `{right}` ({right_len} rows) differ in length")]
    LengthMismatch {
        /// Name of the first column.
        left: &'static str,
        /// Number of rows of the first column.
        left_len: usize,
        /// Name of the second column.
        right: &'static str,
        /// Number of rows of the second column.
        right_len: usize,
    },
    /// An integer code does not fit the declared cardinality.
    #[error("{what} code {code} is out of bounds (expected a value below {bound})")]
    OutOfBounds {
        /// What the code identifies (user, item, group, ...).
        what: &'static str,
        /// The offending code.
        code: u32,
        /// The exclusive upper bound.
        bound: usize,
    },
    /// The same `(user, item)` pair appears more than once in a table.
    #[error("{table} contain the same (user, item) pair more than once")]
    DuplicatePair {
        /// Human readable name of the table.
        table: &'static str,
    },
    /// A value violates the domain of the formula being computed.
    #[error("{0}")]
    InvalidValue(String),
}

impl Error {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidValue(message.into())
    }
}

pub(crate) fn check_same_len(
    left: &'static str,
    left_len: usize,
    right: &'static str,
    right_len: usize,
) -> Result<()> {
    if left_len == right_len {
        Ok(())
    } else {
        Err(Error::LengthMismatch {
            left,
            left_len,
            right,
            right_len,
        })
    }
}

/// Rows and entities are indexed with `u32`: refuse what would not fit.
pub(crate) fn check_indexable(what: &'static str, len: usize) -> Result<()> {
    if u32::try_from(len).is_ok() {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "{what} have {len} rows, more than the {} supported",
            u32::MAX
        )))
    }
}

/// Codes are `u32`: there cannot be more than 2^32 users, items or groups.
pub(crate) fn check_cardinality(what: &'static str, n: usize) -> Result<()> {
    if u64::try_from(n).is_ok_and(|n| n <= u64::from(u32::MAX) + 1) {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "{n} {what} cannot be identified by 32-bit codes"
        )))
    }
}

/// The `u32` code of an index that one of the checks above already bounds.
#[expect(
    clippy::cast_possible_truncation,
    reason = "callers pass an index below a length accepted by check_indexable or check_cardinality"
)]
pub(crate) fn code(index: usize) -> u32 {
    debug_assert!(u32::try_from(index).is_ok(), "index not checked");
    index as u32
}

/// A vector of `len` copies of `value`, or an error when the memory cannot be reserved
/// (instead of aborting the process).
pub(crate) fn filled<T: Clone>(value: T, len: usize, what: &'static str) -> Result<Vec<T>> {
    let mut vector = Vec::new();
    vector
        .try_reserve_exact(len)
        .map_err(|_| Error::invalid(format!("not enough memory for {len} {what}")))?;
    vector.resize(len, value);
    Ok(vector)
}

pub(crate) fn check_codes(what: &'static str, codes: &[u32], bound: usize) -> Result<()> {
    match codes.iter().find(|&&code| code as usize >= bound) {
        Some(&code) => Err(Error::OutOfBounds { what, code, bound }),
        None => Ok(()),
    }
}

/// Assert that a result is an error whose message contains the given text.
#[cfg(test)]
macro_rules! assert_fails {
    ($result:expr, $text:expr) => {{
        match $result {
            Ok(_) => panic!("an error containing {:?} was expected", $text),
            Err(error) => {
                let message = error.to_string();
                assert!(message.contains($text), "unexpected error: {message}");
            }
        }
    }};
}
#[cfg(test)]
pub(crate) use assert_fails;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_must_be_below_the_bound() {
        assert_eq!(check_codes("item", &[0, 2], 3), Ok(()));
        assert_eq!(
            check_codes("item", &[0, 3], 3),
            Err(Error::OutOfBounds {
                what: "item",
                code: 3,
                bound: 3
            })
        );
    }

    #[test]
    fn columns_must_have_the_same_length() {
        assert_eq!(check_same_len("user", 2, "item", 2), Ok(()));
        assert!(matches!(
            check_same_len("user", 2, "item", 3),
            Err(Error::LengthMismatch { .. })
        ));
    }

    /// Rows and entities are indexed with `u32`: larger inputs must be refused rather
    /// than silently truncated.
    #[test]
    #[cfg(target_pointer_width = "64")]
    fn inputs_too_large_to_index_are_rejected() {
        let limit = u32::MAX as usize;
        assert_eq!(check_indexable("recommendations", limit), Ok(()));
        assert_fails!(
            check_indexable("recommendations", limit + 1),
            "more than the 4294967295 supported"
        );
    }
}
