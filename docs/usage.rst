=====
Usage
=====

To use recsyslearn in a project::

    import recsyslearn

To use its features in a project you just need a recommendation list in the form of a ``user, item, rank, group`` Dataframe. The library will do the rest.
If you don't have the info about the groups, you can use the library itself to segment the dataset into the different groups. The dataset has to be in the form of ``user, item, rating`` or ``user, item, timestamp``.

