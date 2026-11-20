//! Python bindings for `recsyslearn-core`.
//!
//! The bindings are deliberately thin: the Python package turns identifiers into
//! dense codes with pandas and hands contiguous `NumPy` arrays over; everything here
//! borrows them without copying and releases the GIL while the core runs.

// PyO3 extracts arguments by value; borrowed NumPy arrays cannot be taken by reference.
#![allow(clippy::needless_pass_by_value)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyType;
use recsyslearn_core as core;

/// Turn a core error into `recsyslearn.errors.InvalidInputError`.
fn invalid_input(py: Python<'_>, error: &core::Error) -> PyErr {
    let exception = py
        .import("recsyslearn.errors")
        .and_then(|module| module.getattr("InvalidInputError"))
        .and_then(|class| Ok(class.cast_into::<PyType>()?));
    match exception {
        Ok(class) => PyErr::from_type(class, error.to_string()),
        Err(import_error) => import_error,
    }
}

/// Run `compute` without holding the GIL and convert its error.
fn run<T, F>(py: Python<'_>, compute: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce() -> core::Result<T> + Send,
{
    py.detach(compute)
        .map_err(|error| invalid_input(py, &error))
}

fn parse_exposure(py: Python<'_>, name: &str) -> PyResult<core::fairness::Exposure> {
    match name {
        "logarithmic" => Ok(core::fairness::Exposure::Logarithmic),
        "flat" => Ok(core::fairness::Exposure::Flat),
        other => Err(invalid_input(
            py,
            &core::Error::InvalidValue(format!(
                "exposure must be 'logarithmic' or 'flat', got {other:?}"
            )),
        )),
    }
}

fn parse_gain(py: Python<'_>, name: &str) -> PyResult<core::accuracy::Gain> {
    match name {
        "linear" => Ok(core::accuracy::Gain::Linear),
        "exponential" => Ok(core::accuracy::Gain::Exponential),
        other => Err(invalid_input(
            py,
            &core::Error::InvalidValue(format!(
                "gain must be 'linear' or 'exponential', got {other:?}"
            )),
        )),
    }
}

/// Native implementation of the `recsyslearn` Python package.
#[pymodule]
mod _core {
    use super::{
        Bound, IntoPyArray, PyArray1, PyReadonlyArray1, PyResult, Python, core, parse_exposure,
        parse_gain, pyclass, pyfunction, pymethods, run,
    };
    use core::fairness::{BenefitDistribution, Partition};

    /// Recommendation lists grouped by user, best item first.
    #[pyclass(frozen, module = "recsyslearn._core")]
    struct RankedLists(core::RankedLists);

    #[pymethods]
    impl RankedLists {
        #[new]
        #[pyo3(signature = (users, items, ranks, n_users))]
        fn new(
            py: Python<'_>,
            users: PyReadonlyArray1<'_, u32>,
            items: PyReadonlyArray1<'_, u32>,
            ranks: Option<PyReadonlyArray1<'_, f64>>,
            n_users: usize,
        ) -> PyResult<Self> {
            let users = users.as_slice()?;
            let items = items.as_slice()?;
            let ranks = ranks.as_ref().map(PyReadonlyArray1::as_slice).transpose()?;
            run(py, || core::RankedLists::new(users, items, ranks, n_users)).map(Self)
        }

        /// Lists ordered by descending score; equal scores of a user are ties.
        #[staticmethod]
        #[pyo3(signature = (users, items, scores, n_users))]
        fn from_scores(
            py: Python<'_>,
            users: PyReadonlyArray1<'_, u32>,
            items: PyReadonlyArray1<'_, u32>,
            scores: PyReadonlyArray1<'_, f64>,
            n_users: usize,
        ) -> PyResult<Self> {
            let users = users.as_slice()?;
            let items = items.as_slice()?;
            let scores = scores.as_slice()?;
            run(py, || {
                core::RankedLists::from_scores(users, items, scores, n_users)
            })
            .map(Self)
        }
    }

    /// Held-out relevance judgements grouped by user.
    #[pyclass(frozen, module = "recsyslearn._core")]
    struct GroundTruth(core::GroundTruth);

    #[pymethods]
    impl GroundTruth {
        #[new]
        #[pyo3(signature = (users, items, relevance, n_users))]
        fn new(
            py: Python<'_>,
            users: PyReadonlyArray1<'_, u32>,
            items: PyReadonlyArray1<'_, u32>,
            relevance: Option<PyReadonlyArray1<'_, f64>>,
            n_users: usize,
        ) -> PyResult<Self> {
            let users = users.as_slice()?;
            let items = items.as_slice()?;
            let relevance = relevance
                .as_ref()
                .map(PyReadonlyArray1::as_slice)
                .transpose()?;
            run(py, || {
                core::GroundTruth::new(users, items, relevance, n_users)
            })
            .map(Self)
        }
    }

    /// nDCG@k of every user (NaN where the user has no relevant item).
    #[pyfunction]
    #[pyo3(signature = (lists, truth, k, gain))]
    fn ndcg<'py>(
        py: Python<'py>,
        lists: &Bound<'py, RankedLists>,
        truth: &Bound<'py, GroundTruth>,
        k: Option<usize>,
        gain: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let gain = parse_gain(py, gain)?;
        let (lists, truth) = (&lists.get().0, &truth.get().0);
        let scores = run(py, || core::accuracy::ndcg(lists, truth, k, gain))?;
        Ok(scores.into_pyarray(py))
    }

    /// Mean of the per-user scores that are defined (not NaN); `None` without any.
    #[pyfunction]
    fn mean_over_users(py: Python<'_>, scores: PyReadonlyArray1<'_, f64>) -> PyResult<Option<f64>> {
        let scores = scores.as_slice()?;
        Ok(py.detach(|| core::aggregate::mean_over_users(scores)))
    }

    /// Fraction of the `n_items` catalog items found in the top-k lists.
    #[pyfunction]
    #[pyo3(signature = (lists, k, n_items))]
    fn catalog_coverage(
        py: Python<'_>,
        lists: &Bound<'_, RankedLists>,
        k: Option<usize>,
        n_items: usize,
    ) -> PyResult<f64> {
        let lists = &lists.get().0;
        run(py, || {
            core::beyond_accuracy::catalog_coverage(lists, k, n_items)
        })
    }

    /// Mean self-information of every user's top-k list (NaN for empty lists).
    #[pyfunction]
    #[pyo3(signature = (lists, k, degrees, n_users))]
    fn novelty<'py>(
        py: Python<'py>,
        lists: &Bound<'py, RankedLists>,
        k: Option<usize>,
        degrees: PyReadonlyArray1<'py, u64>,
        n_users: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let lists = &lists.get().0;
        let degrees = degrees.as_slice()?;
        let scores = run(py, || {
            core::beyond_accuracy::novelty(lists, k, degrees, n_users)
        })?;
        Ok(scores.into_pyarray(py))
    }

    /// Marginals over user blocks and item blocks of the normalised benefit, and
    /// their mutual information in bits.
    ///
    /// A side without group codes is a single block, or one block per individual
    /// when listed in `singletons` (`"user"` or `"item"`).
    #[pyfunction]
    #[pyo3(signature = (lists, k, exposure, truth, user_groups, n_user_groups, item_groups, n_item_groups, singletons))]
    // One binding serves the three fairness metrics: its flat list of arguments and its
    // tuple of results mirror the Python call, and would only be wrapped to be unwrapped.
    #[expect(clippy::too_many_arguments, clippy::type_complexity)]
    fn benefit_distribution<'py>(
        py: Python<'py>,
        lists: &Bound<'py, RankedLists>,
        k: Option<usize>,
        exposure: &str,
        truth: Option<&Bound<'py, GroundTruth>>,
        user_groups: Option<PyReadonlyArray1<'py, u32>>,
        n_user_groups: usize,
        item_groups: Option<PyReadonlyArray1<'py, u32>>,
        n_item_groups: usize,
        singletons: Option<&str>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
        let exposure = parse_exposure(py, exposure)?;
        let lists = &lists.get().0;
        let truth = truth.map(|truth| &truth.get().0);
        let user_codes = user_groups
            .as_ref()
            .map(PyReadonlyArray1::as_slice)
            .transpose()?;
        let item_codes = item_groups
            .as_ref()
            .map(PyReadonlyArray1::as_slice)
            .transpose()?;
        let partition = |codes, n_groups, side| -> Partition<'_> {
            match codes {
                Some(codes) => Partition::Groups { codes, n_groups },
                None if singletons == Some(side) => Partition::Singletons,
                None => Partition::Whole,
            }
        };
        let users = partition(user_codes, n_user_groups, "user");
        let items = partition(item_codes, n_item_groups, "item");
        let (user_marginal, item_marginal, information) = run(py, || {
            let distribution = BenefitDistribution::new(lists, k, exposure, truth, users, items)?;
            Ok((
                distribution.user_marginal().to_vec(),
                distribution.item_marginal().to_vec(),
                distribution.mutual_information(),
            ))
        })?;
        Ok((
            user_marginal.into_pyarray(py),
            item_marginal.into_pyarray(py),
            information,
        ))
    }

    /// Shannon entropy of a probability distribution: in bits, or divided by its
    /// maximum when `normalize` is set.
    #[pyfunction]
    fn entropy(
        py: Python<'_>,
        distribution: PyReadonlyArray1<'_, f64>,
        normalize: bool,
    ) -> PyResult<f64> {
        let distribution = distribution.as_slice()?;
        run(py, || {
            if normalize {
                core::fairness::normalized_entropy(distribution)
            } else {
                core::fairness::entropy(distribution)
            }
        })
    }

    /// Kullback-Leibler divergence, in bits, of `distribution` from `target`, or from
    /// parity (the same probability for every group) without one.
    #[pyfunction]
    fn kl_divergence(
        py: Python<'_>,
        distribution: PyReadonlyArray1<'_, f64>,
        target: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<f64> {
        let distribution = distribution.as_slice()?;
        let target = target
            .as_ref()
            .map(PyReadonlyArray1::as_slice)
            .transpose()?;
        run(py, || match target {
            Some(target) => core::fairness::kl_divergence(distribution, target),
            None => core::fairness::divergence_from_parity(distribution),
        })
    }

    /// Number of rows in which each of the `n` entities appears.
    #[pyfunction]
    fn interaction_counts<'py>(
        py: Python<'py>,
        codes: PyReadonlyArray1<'py, u32>,
        n: usize,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let codes = codes.as_slice()?;
        let counts = run(py, || core::segmentation::interaction_counts(codes, n))?;
        Ok(counts.into_pyarray(py))
    }

    /// Number of distinct counterparts each of the `n` entities interacted with.
    #[pyfunction]
    fn distinct_counts<'py>(
        py: Python<'py>,
        entities: PyReadonlyArray1<'py, u32>,
        counterparts: PyReadonlyArray1<'py, u32>,
        n: usize,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let entities = entities.as_slice()?;
        let counterparts = counterparts.as_slice()?;
        let counts = run(py, || {
            core::segmentation::distinct_counts(entities, counterparts, n)
        })?;
        Ok(counts.into_pyarray(py))
    }

    /// Each count divided by `total`.
    #[pyfunction]
    fn shares<'py>(
        py: Python<'py>,
        counts: PyReadonlyArray1<'py, u64>,
        total: u64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let counts = counts.as_slice()?;
        let shares = run(py, || core::segmentation::shares(counts, total))?;
        Ok(shares.into_pyarray(py))
    }

    /// Group (0 = most popular) of every entity by cumulative share of interactions.
    #[pyfunction]
    fn by_cumulative_interactions<'py>(
        py: Python<'py>,
        counts: PyReadonlyArray1<'py, u64>,
        proportions: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let counts = counts.as_slice()?;
        let groups = run(py, || {
            core::segmentation::by_cumulative_interactions(counts, &proportions)
        })?;
        Ok(groups.into_pyarray(py))
    }

    /// Group (0 = most popular) of every entity by share of the population.
    #[pyfunction]
    fn by_interaction_count<'py>(
        py: Python<'py>,
        counts: PyReadonlyArray1<'py, u64>,
        proportions: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let counts = counts.as_slice()?;
        let groups = run(py, || {
            core::segmentation::by_interaction_count(counts, &proportions)
        })?;
        Ok(groups.into_pyarray(py))
    }
}
