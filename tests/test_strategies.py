"""The Hypothesis strategies generate inputs that follow the data contract of AGENTS.md."""

import unittest

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from tests import strategies
from tests.case import TestCase


class TopNStrategyTest(TestCase):
    @given(strategies.top_n())
    def test_has_the_user_item_rank_columns(self, top_n) -> None:
        self.assertEqual(list(top_n.columns), ["user", "item", "rank"])

    @given(strategies.top_n())
    def test_ranks_go_from_one_to_the_list_length(self, top_n) -> None:
        for _, user_list in top_n.groupby("user"):
            self.assertEqual(
                user_list["rank"].tolist(), list(range(1, len(user_list) + 1))
            )

    @given(strategies.top_n())
    def test_items_are_distinct_within_a_list(self, top_n) -> None:
        for _, user_list in top_n.groupby("user"):
            self.assertTrue(user_list["item"].is_unique)


class PosItemsStrategyTest(TestCase):
    @given(st.data())
    def test_has_the_user_pos_items_columns(self, data) -> None:
        pos_items = self._draw(data)
        self.assertEqual(list(pos_items.columns), ["user", "pos_items"])

    @given(st.data())
    def test_relevant_items_are_not_empty_by_default(self, data) -> None:
        for relevant in self._draw(data)["pos_items"]:
            self.assertGreater(len(relevant), 0)

    @given(st.data())
    def test_relevant_items_are_distinct(self, data) -> None:
        for relevant in self._draw(data)["pos_items"]:
            self.assertEqual(len(set(relevant)), len(relevant))

    @given(st.data())
    def test_relevant_items_come_from_the_catalogue(self, data) -> None:
        items = data.draw(strategies.catalogues())
        pos_items = data.draw(strategies.pos_items(["u1", "u2"], items))
        for relevant in pos_items["pos_items"]:
            self.assertTrue(set(relevant) <= set(items))

    @staticmethod
    def _draw(data):
        """Draw relevant items of two users from a generated catalogue."""
        items = data.draw(strategies.catalogues())
        return data.draw(strategies.pos_items(["u1", "u2"], items))


class GroupsStrategyTest(TestCase):
    @given(st.data())
    def test_assigns_a_group_to_every_id(self, data) -> None:
        items = data.draw(strategies.catalogues())
        item_groups = data.draw(strategies.groups(items, key="item"))
        self.assertEqual(item_groups["item"].tolist(), items)


class TargetRepresentationStrategyTest(TestCase):
    @given(st.data())
    def test_shares_are_positive(self, data) -> None:
        target = self._draw(data)
        self.assertTrue((target["target_representation"] > 0).all())

    @given(st.data())
    def test_shares_sum_to_one(self, data) -> None:
        target = self._draw(data)
        self.assertTrue(np.isclose(target["target_representation"].sum(), 1.0))

    @staticmethod
    def _draw(data):
        """Draw a target representation over 1 to 4 groups."""
        count = data.draw(st.integers(min_value=1, max_value=4))
        return data.draw(
            strategies.target_representations([f"g{i}" for i in range(count)])
        )


if __name__ == "__main__":
    unittest.main()
