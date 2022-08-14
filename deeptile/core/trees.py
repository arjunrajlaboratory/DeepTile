from collections.abc import Mapping, Sequence, MutableMapping, MutableSequence


def tree_scan(tree):

    """ Scan a tree for all branches and leaves.

    Parameters
    ----------
        tree
            Tree object.

    Returns
    -------
        istree : bool
            Whether the object is a tree.
        branch_indices : list of tuple
            List of all branch indices.
        leaf_indices: list of tuple
            List of all leaf indices.
    """

    istree = _check_istree(tree)

    if istree:

        current_index = [0]
        branch_sizes = [len(tree)]
        branch_indices = []
        leaf_indices = []

        while len(current_index) > 0:

            if current_index[-1] < branch_sizes[-1]:
                current_obj = tree_index(tree, current_index)
                if _check_istree(current_obj):
                    branch_sizes.append(len(current_obj))
                    branch_indices.append(tuple(current_index))
                    current_index.append(0)
                else:
                    leaf_indices.append(tuple(current_index))
                    current_index[-1] += 1
            else:
                current_index.pop()
                if len(current_index) > 0:
                    current_index[-1] += 1
                branch_sizes.pop()

    else:

        branch_indices = []
        leaf_indices = []

    return istree, branch_indices, leaf_indices


def tree_apply(tree, leaf_indices, func):

    """ Apply a function to all given leaves of a tree.

    Parameters
    ----------
        tree
            Tree object.
        leaf_indices : list of tuple
            List of leaf indices.
        func : Callable
            Callable to be applied on tree leaves.

    Returns
    -------
        tree
            Tree object.
    """

    if _check_istree(tree):

        if len(leaf_indices) > 0:

            if isinstance(tree, Sequence):
                tree = list(tree)
            elif isinstance(tree, Mapping):
                tree = dict(tree)

            branch_indices = _get_branches_from_leaves(leaf_indices)

            for branch_index in branch_indices:

                branch = tree_index(tree, branch_index)

                if isinstance(branch, Sequence):
                    tree_replace(tree, branch_index, list(branch))
                elif isinstance(branch, Mapping):
                    tree_replace(tree, branch_index, dict(branch))

            for leaf_index in leaf_indices:
                tree_replace(tree, leaf_index, func(tree_index(tree, leaf_index)))

    else:

        tree = func(tree)

    return tree


def tree_replace(tree, index, new_obj):

    """ Replace a branch or leaf in a tree.

    Parameters
    ----------
        tree
            Tree object.
        index : tuple of int
            Branch or leaf index.
        new_obj : obj
            New object for replacement.
    """

    branch_index = index[:-1]
    branch = tree_index(tree, branch_index)

    if isinstance(branch, MutableSequence):
        branch[index[-1]] = new_obj

    if isinstance(branch, MutableMapping):
        key = tuple(branch.keys())[index[-1]]
        branch[key] = new_obj


def tree_index(tree, index):

    """ Index a tree.

    Parameters
    ----------
        tree
            Tree object.
        index : tuple of int
            Branch or leaf index.

    Returns
    -------
        obj
            Object at given index.
    """

    obj = tree

    for i in index:

        if isinstance(obj, Sequence):
            obj = obj[i]
        elif isinstance(obj, Mapping):
            obj = tuple(obj.values())[i]
        else:
            raise ValueError('tree branches be sequences or mappings.')

    return obj


def _check_istree(obj):

    """ (For internal use) Check if an object is a tree.

    Parameters
    ----------
        obj
            Object to be checked.

    Returns
    -------
        istree : bool
            Whether the object is a tree.
    """

    istree = isinstance(obj, (Mapping, Sequence)) and not isinstance(obj, str)

    return istree


def _get_branches_from_leaves(leaf_indices):

    """ (For internal use) Get a list of branch indices from leaf indices.

    Parameters
    ----------
        leaf_indices: list of tuple
            List of leaf indices.

    Returns
    -------
        branch_indices : list of tuple
            List of branch indices.
    """

    branch_indices = set(index[:-1] for index in leaf_indices if len(index) > 1)
    num_branches_i = None
    num_branches_f = None
    while (num_branches_f is None) or (num_branches_i < num_branches_f):
        num_branches_i = len(branch_indices)
        branch_indices = branch_indices | set(branch_index[:-1] for branch_index in branch_indices
                                              if len(branch_index) > 1)
        num_branches_f = len(branch_indices)
    branch_indices = list(branch_indices)
    branch_indices.sort(key=lambda index: len(index))

    return branch_indices
