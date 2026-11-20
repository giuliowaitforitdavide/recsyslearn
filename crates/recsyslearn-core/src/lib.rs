//! Evaluation of recommender systems.
//!
//! The crate implements, from their definition in the literature:
//!
//! * [`accuracy`] - nDCG@k;
//! * [`beyond_accuracy`] - catalog coverage and novelty;
//! * [`fairness`] - entropy, Kullback-Leibler divergence and mutual information of the
//!   distribution of recommendations over groups of users or items;
//! * [`segmentation`] - splitting users or items into groups by popularity or activity;
//! * [`aggregate`] - the mean of per-user scores.
//!
//! Users, items and groups are identified by integer codes (dense for users and groups);
//! mapping arbitrary identifiers to codes is left to the caller (the Python package does
//! it with pandas).

// Tests compare against values that are exact in binary floating point.
#![cfg_attr(test, allow(clippy::float_cmp))]

pub mod accuracy;
pub mod aggregate;
pub mod beyond_accuracy;
pub mod error;
pub mod fairness;
pub mod lists;
pub mod segmentation;

pub use error::{Error, Result};
pub use lists::{GroundTruth, RankedLists};
