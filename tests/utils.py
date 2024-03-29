import numpy as np
import pandas as pd

from recsyslearn.dataset.utils import find_relevant_items

item_groups = pd.DataFrame(
    [
        ["1", "1"],
        ["2", "1"],
        ["3", "2"],
        ["4", "2"],
        ["5", "2"],
        ["6", "3"],
        ["7", "3"],
        ["8", "3"],
        ["9", "3"],
        ["10", "3"],
    ],
    columns=["item", "group"],
)

user_groups = pd.DataFrame(
    [["1", "1"], ["2", "1"], ["3", "2"], ["4", "2"], ["5", "2"], ["6", "2"]],
    columns=["user", "group"],
)

first_example = pd.DataFrame(
    [
        ["1", "3", 1],
        ["1", "4", 1],
        ["1", "6", 1],
        ["1", "7", 1],
        ["1", "8", 1],
        ["2", "1", 1],
        ["2", "2", 1],
        ["2", "5", 1],
        ["2", "6", 1],
        ["2", "7", 1],
        ["3", "2", 1],
        ["3", "3", 1],
        ["3", "6", 1],
        ["3", "9", 1],
        ["3", "10", 1],
        ["4", "1", 1],
        ["4", "3", 1],
        ["4", "6", 1],
        ["4", "7", 1],
        ["4", "9", 1],
        ["5", "1", 1],
        ["5", "3", 1],
        ["5", "5", 1],
        ["5", "7", 1],
        ["5", "9", 1],
        ["6", "2", 1],
        ["6", "3", 1],
        ["6", "5", 1],
        ["6", "9", 1],
        ["6", "10", 1],
    ],
    columns=["user", "item", "rank"],
)

second_example = pd.DataFrame(
    [
        ["1", "1", 1],
        ["1", "3", 6],
        ["1", "4", 3],
        ["1", "5", 4],
        ["1", "8", 5],
        ["1", "9", 2],
        ["2", "2", 6],
        ["2", "3", 5],
        ["2", "6", 4],
        ["2", "7", 3],
        ["2", "8", 2],
        ["2", "9", 1],
        ["3", "2", 1],
        ["3", "3", 2],
        ["3", "4", 3],
        ["3", "7", 4],
        ["3", "8", 5],
        ["3", "9", 6],
        ["4", "1", 4],
        ["4", "3", 5],
        ["4", "4", 6],
        ["4", "7", 3],
        ["4", "8", 2],
        ["4", "9", 1],
        ["5", "1", 6],
        ["5", "3", 2],
        ["5", "5", 3],
        ["5", "6", 4],
        ["5", "8", 5],
        ["5", "9", 1],
        ["6", "2", 3],
        ["6", "3", 5],
        ["6", "5", 4],
        ["6", "6", 6],
        ["6", "8", 2],
        ["6", "9", 1],
    ],
    columns=["user", "item", "rank"],
)

top_n_1 = pd.DataFrame(
    [
        ["1", "2", 1],
        ["1", "8", 2],
        ["1", "5", 3],
        ["1", "3", 4],
        ["1", "6", 5],
        ["2", "1", 1],
        ["2", "2", 2],
        ["2", "6", 3],
        ["2", "7", 4],
        ["2", "9", 5],
        ["3", "6", 1],
        ["3", "5", 2],
        ["3", "8", 3],
        ["3", "7", 4],
        ["3", "9", 5],
        ["4", "1", 1],
        ["4", "2", 2],
        ["4", "3", 3],
        ["4", "4", 4],
        ["4", "5", 5],
        ["5", "5", 1],
        ["5", "4", 2],
        ["5", "3", 3],
        ["5", "2", 4],
        ["5", "1", 5],
        ["6", "9", 1],
        ["6", "8", 2],
        ["6", "6", 3],
        ["6", "3", 4],
        ["6", "4", 5],
    ],
    columns=["user", "item", "rank"],
)

rel_matrix_1 = pd.DataFrame(
    [
        ["1", "2", 1],
        ["1", "9", 1],
        ["2", "6", 1],
        ["2", "7", 1],
        ["3", "1", 1],
        ["3", "7", 1],
        ["3", "9", 1],
        ["4", "3", 1],
        ["4", "7", 1],
        ["6", "2", 1],
        ["6", "9", 1],
    ],
    columns=["user", "item", "rank"],
)

rel_matrix_2 = pd.DataFrame(
    [
        ["1", "2", 1],
        ["1", "9", 1],
        ["2", "6", 1],
        ["2", "7", 1],
        ["3", "1", 1],
        ["3", "3", 1],
        ["3", "4", 1],
        ["3", "5", 1],
        ["3", "7", 1],
        ["3", "9", 1],
        ["4", "2", 1],
        ["4", "3", 1],
        ["4", "5", 1],
        ["4", "7", 1],
        ["4", "8", 1],
        ["4", "9", 1],
        ["5", "1", 1],
        ["5", "2", 1],
        ["5", "4", 1],
        ["5", "5", 1],
        ["5", "6", 1],
        ["5", "9", 1],
        ["6", "1", 1],
        ["6", "2", 1],
        ["6", "6", 1],
        ["6", "8", 1],
        ["6", "9", 1],
    ],
    columns=["user", "item", "rank"],
)

rel_matrix_3 = pd.DataFrame(
    [
        ["1", "2", 0.143],
        ["1", "9", 0.077],
        ["2", "6", 0.077],
        ["2", "7", 0.077],
        ["3", "1", 0.143],
        ["3", "3", 0.143],
        ["3", "4", 0.143],
        ["3", "5", 0.143],
        ["3", "7", 0.077],
        ["3", "9", 0.077],
        ["4", "2", 0.143],
        ["4", "3", 0.143],
        ["4", "5", 0.07743],
        ["4", "7", 0.077],
        ["4", "8", 0.077],
        ["4", "9", 0.077],
        ["5", "1", 0.143],
        ["5", "2", 0.143],
        ["5", "4", 0.143],
        ["5", "5", 0.143],
        ["5", "6", 0.077],
        ["5", "9", 0.077],
        ["6", "1", 0.143],
        ["6", "2", 0.143],
        ["6", "6", 0.077],
        ["6", "8", 0.077],
        ["6", "9", 0.077],
    ],
    columns=["user", "item", "rank"],
)

rel_matrix_4 = pd.DataFrame(
    [
        ["1", "2", 1],
        ["1", "9", 1],
        ["2", "6", 1],
        ["2", "7", 1],
        ["3", "1", 1],
        ["3", "3", 1],
        ["3", "4", 1],
        ["3", "5", 1],
        ["3", "7", 1],
        ["3", "9", 1],
        ["4", "2", 1],
        ["4", "3", 1],
        ["4", "5", 1],
        ["4", "7", 1],
        ["4", "8", 1],
        ["4", "9", 1],
        ["5", "1", 1],
        ["5", "2", 1],
        ["5", "4", 1],
        ["5", "5", 1],
        ["5", "6", 1],
        ["5", "9", 1],
        ["6", "9", 1],
    ],
    columns=["user", "item", "rank"],
)

dataset_item_example = pd.DataFrame(
    [["1", "1", 1], ["2", "1", 1], ["3", "1", 1], ["1", "2", 1], ["1", "2", 1]],
    columns=["user", "item", "rank"],
)

dataset_user_example = pd.DataFrame(
    [
        ["1", "1", 1],
        ["1", "2", 1],
        ["1", "3", 1],
        ["1", "4", 1],
        ["1", "5", 1],
        ["1", "6", 1],
        ["1", "7", 1],
        ["1", "8", 1],
        ["1", "9", 1],
        ["1", "1", 1],
        ["1", "2", 1],
        ["1", "3", 1],
        ["1", "4", 1],
        ["1", "5", 1],
        ["1", "6", 1],
        ["1", "7", 1],
        ["1", "8", 1],
        ["1", "9", 1],
        ["1", "1", 1],
        ["1", "2", 1],
        ["1", "3", 1],
        ["1", "4", 1],
        ["1", "5", 1],
        ["1", "6", 1],
        ["1", "7", 1],
        ["1", "8", 1],
        ["1", "9", 1],
        ["1", "1", 1],
        ["1", "2", 1],
        ["1", "3", 1],
        ["1", "4", 1],
        ["1", "5", 1],
        ["1", "6", 1],
        ["1", "7", 1],
        ["1", "8", 1],
        ["1", "9", 1],
        ["2", "1", 1],
        ["2", "1", 1],
        ["3", "1", 1],
    ],
    columns=["user", "item", "rank"],
)

dataset_popularity = pd.DataFrame(
    [
        ["1", "1", 1],
        ["1", "1", 1],
        ["1", "1", 1],
        ["1", "1", 1],
        ["2", "1", 1],
        ["3", "1", 1],
        ["1", "2", 1],
        ["1", "2", 1],
        ["2", "2", 1],
        ["1", "3", 1],
    ],
    columns=["user", "item", "rank"],
)

item_pop_perc = pd.DataFrame(
    [
        ["1", 0.05],
        ["2", 0.01],
        ["3", 0.03],
        ["4", 0.23],
        ["5", 0.4],
        ["6", 0.6],
        ["7", 0.15],
        ["8", 0.34],
        ["9", 0.07],
        ["10", 0.02],
    ],
    columns=["item", "percentage"],
)

user_pop_perc = pd.DataFrame(
    [
        ["1", 0.7],
        ["2", 0.2],
        ["3", 0.1],
    ],
    columns=["user", "percentage"],
)

item_feature = pd.DataFrame(
    [
        ["1", "pop"],
        ["2", "pop"],
        ["3", "electronic"],
        ["4", "rock"],
    ],
    columns=["item", "genre"],
)

user_feature = pd.DataFrame(
    [
        ["1", "m"],
        ["2", "f"],
        ["3", None],
        ["4", np.nan],
    ],
    columns=["user", "gender"],
)

user_error_feature = pd.DataFrame(
    [
        ["1", 19],
        ["2", 30],
        ["3", -1],
    ],
    columns=["user", "age"],
)

pos_items = find_relevant_items(rel_matrix_4)
