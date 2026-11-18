import unittest

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from recsyslearn.utils import check_columns_exist
from tests import strategies


class StrategiesTest(unittest.TestCase):
    """The Hypothesis strategies generate inputs that follow the data contract of AGENTS.md."""

    @given(strategies.top_n())
    def test_top_n(self, top_n) -> None:
        """Every user has distinct items ranked 1, 2, …, length of the list."""
        check_columns_exist(top_n, ["user", "item", "rank"])
        for _, user_list in top_n.groupby("user"):
            self.assertEqual(
                user_list["rank"].tolist(), list(range(1, len(user_list) + 1))
            )
            self.assertTrue(user_list["item"].is_unique)

    @given(st.data())
    def test_pos_items(self, data) -> None:
        """Relevant items come from the catalogue, are distinct and, by default, not empty."""
        items = data.draw(strategies.catalogues())
        pos_items = data.draw(strategies.pos_items(["u1", "u2"], items))
        check_columns_exist(pos_items, ["user", "pos_items"])
        for relevant in pos_items["pos_items"]:
            self.assertGreater(len(relevant), 0)
            self.assertEqual(len(set(relevant)), len(relevant))
            self.assertTrue(set(relevant) <= set(items))

    @given(st.data())
    def test_groups_and_target_representation(self, data) -> None:
        """Every item gets a group, and the target representation is positive and sums to 1."""
        items = data.draw(strategies.catalogues())
        item_groups = data.draw(strategies.groups(items, key="item"))
        self.assertEqual(item_groups["item"].tolist(), items)

        group_names = sorted(item_groups["group"].unique())
        target = data.draw(strategies.target_representations(group_names))
        check_columns_exist(target, ["group", "target_representation"])
        self.assertTrue((target["target_representation"] > 0).all())
        self.assertTrue(np.isclose(target["target_representation"].sum(), 1.0))


if __name__ == "__main__":
    unittest.main()
