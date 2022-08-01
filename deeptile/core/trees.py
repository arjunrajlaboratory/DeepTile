from collections.abc import Mapping, Sequence, MutableMapping, MutableSequence


def tree_scan(tree):

    """ Scan a tree for all branches and leaves.

    Parameters
    ----------
        tree
            Tree object.

    Returns
    -------
        branch_indices : list of tuple
            List of all branch indices.
        leaf_indices: list of tuple
            List of all leaf indices.
    """

    current_index = [0]
    branch_sizes = [len(tree)]
    branch_indices = []
    leaf_indices = []

    while len(current_index) > 0:

        if current_index[-1] < branch_sizes[-1]:
            current_leaf = tree_index(tree, current_index)
            if isinstance(current_leaf, (Sequence, Mapping)):
                branch_sizes.append(len(current_leaf))
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

    return branch_indices, leaf_indices


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

    if isinstance(tree, Sequence):
        tree = list(tree)
    elif isinstance(tree, Mapping):
        tree = dict(tree)

    for leaf_index in leaf_indices:

        branch_index = leaf_index[:-1]
        branch = tree_index(tree, branch_index)

        if isinstance(branch, Sequence):
            if not isinstance(branch, MutableSequence):
                tree_replace(tree, branch_index, list(branch))
        elif isinstance(branch, Mapping):
            if not isinstance(branch, MutableMapping):
                tree_replace(tree, branch_index, dict(branch))

        tree_replace(tree, leaf_index, func(tree_index(tree, leaf_index)))

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
