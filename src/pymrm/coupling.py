"""Sparse-matrix utilities for multi-domain coupling and interface assembly."""

import math
import numbers
import warnings
import numpy as np
from scipy.sparse import coo_array, csc_array, csr_array
from pymrm.helpers import unwrap_bc_coeff, _sparse_array


def translate_indices_to_larger_array(linear_indices, shape, new_shape, offset=None):
    """Map flat indices from a local array shape to a larger embedding shape.

    Parameters
    ----------
    linear_indices : array_like
        Flat indices defined in ``shape``.
    shape : tuple[int, ...]
        Local array shape.
    new_shape : tuple[int, ...]
        Embedding array shape.
    offset : tuple[int, ...], optional
        Offset of the local array origin in the embedding array.

    Returns
    -------
    numpy.ndarray
        Flat indices in ``new_shape``.
    """

    # Convert linear indices to multi-indices based on the original subarray shape
    multi_indices = np.unravel_index(linear_indices, shape)

    # Shift multi-indices by the offset to their position in the larger array
    if offset is not None:
        adjusted_multi_indices = tuple(m + o for m, o in zip(multi_indices, offset))
    else:
        adjusted_multi_indices = multi_indices

    # Convert back to linear indices in the larger ND array
    new_linear_indices = np.ravel_multi_index(adjusted_multi_indices, new_shape)

    return new_linear_indices


def _parse_shape_offset(shape, new_shape, offset):
    """Normalise *shape*, *new_shape* and *offset* into ((rows), (cols)) form.

    This is a shared helper for :func:`update_csc_array_indices` and
    :func:`update_csr_array_indices`.
    """
    shape = tuple(shape)
    if all(isinstance(dim, numbers.Integral) for dim in shape):
        shape = (shape, shape)
    new_shape = tuple(new_shape)
    if all(isinstance(dim, numbers.Integral) for dim in new_shape):
        new_shape = (new_shape, new_shape)
    if offset is None:
        offset = (None, None)
    else:
        offset = tuple(offset)
        if all(isinstance(dim, numbers.Integral) for dim in offset):
            offset = (offset, offset)
    return shape, new_shape, offset


def update_csc_array_indices(sparse_mat, shape, new_shape, offset=None):
    """Update CSC matrix row/column indexing for embedding in a larger domain.

    .. deprecated::
       Use :func:`update_array_indices` for automatic format dispatch.
    """
    warnings.warn(
        "update_csc_array_indices is deprecated. Use update_array_indices instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _update_csc_array_indices(sparse_mat, shape, new_shape, offset)


def _update_csc_array_indices(sparse_mat, shape, new_shape, offset=None):
    """Internal CSC implementation for :func:`update_csc_array_indices`."""
    shape, new_shape, offset = _parse_shape_offset(shape, new_shape, offset)

    # Extract matrix data and original indices
    data = sparse_mat.data
    row_indices = sparse_mat.indices  # Row indices (modifiable)
    col_pointers = sparse_mat.indptr  # Column pointers
    num_rows = sparse_mat.shape[0]  # Number of rows in the original matrix
    num_cols = sparse_mat.shape[1]  # Number of columns in the original matrix

    # Generate original linear row and column indices
    original_linear_rows = row_indices
    original_linear_cols = np.arange(num_cols)  # Columns as a sequential array

    if (shape[0] is None) or (new_shape[0] is None):
        new_row_indices = original_linear_rows
    else:
        new_row_indices = translate_indices_to_larger_array(
            original_linear_rows, shape[0], new_shape[0], offset[0]
        )
        num_rows = np.prod(new_shape[0])

    if (shape[1] is None) or (new_shape[1] is None):
        new_col_pointers = col_pointers
    else:
        new_col_indices = translate_indices_to_larger_array(
            original_linear_cols, shape[1], new_shape[1], offset[1]
        )
        num_cols = math.prod(new_shape[1])
        new_col_pointers = np.zeros(num_cols + 1, dtype=int)
        new_col_pointers[new_col_indices + 1] = np.diff(col_pointers)
        new_col_pointers = np.cumsum(new_col_pointers)

    # Create a new sparse matrix with the corrected 2D shape
    updated_mat = csc_array(
        (data, new_row_indices, new_col_pointers), shape=(num_rows, num_cols)
    )

    return updated_mat


def update_csr_array_indices(sparse_mat, shape, new_shape, offset=None):
    """Update CSR matrix row/column indexing for embedding in a larger domain."""
    shape, new_shape, offset = _parse_shape_offset(shape, new_shape, offset)

    # Extract matrix data and original indices
    data = sparse_mat.data
    col_indices = sparse_mat.indices    # Column indices
    row_pointers = sparse_mat.indptr    # Row pointers
    num_rows = sparse_mat.shape[0]
    num_cols = sparse_mat.shape[1]

    original_linear_cols = col_indices
    original_linear_rows = np.arange(num_rows)

    if (shape[1] is None) or (new_shape[1] is None):
        new_col_indices = original_linear_cols
    else:
        new_col_indices = translate_indices_to_larger_array(
            original_linear_cols, shape[1], new_shape[1], offset[1]
        )
        num_cols = math.prod(new_shape[1])

    if (shape[0] is None) or (new_shape[0] is None):
        new_row_pointers = row_pointers
    else:
        new_row_indices = translate_indices_to_larger_array(
            original_linear_rows, shape[0], new_shape[0], offset[0]
        )
        num_rows = math.prod(new_shape[0])
        new_row_pointers = np.zeros(num_rows + 1, dtype=int)
        new_row_pointers[new_row_indices + 1] = np.diff(row_pointers)
        new_row_pointers = np.cumsum(new_row_pointers)

    # Create a new sparse matrix with the corrected 2D shape
    updated_mat = csr_array(
        (data, new_col_indices, new_row_pointers), shape=(num_rows, num_cols)
    )

    return updated_mat


def update_array_indices(sparse_mat, shape, new_shape, offset=None):
    """Update sparse-matrix indices for a new embedding shape.

    Parameters
    ----------
    sparse_mat : scipy.sparse.sparray
        Input matrix. CSR and CSC are supported.
    shape, new_shape : tuple or tuple[tuple, tuple]
        Original and target logical shapes for rows and columns. A single tuple
        applies to both rows and columns.
    offset : tuple or tuple[tuple, tuple], optional
        Optional row/column offsets of the local block in the larger domain.

    Returns
    -------
    scipy.sparse.sparray
        Matrix with updated shape and indices, preserving the input format where
        possible.
    """
    if isinstance(sparse_mat, csr_array):
        return update_csr_array_indices(sparse_mat, shape, new_shape, offset)
    # Default: convert to CSC if needed, then update
    sparse_mat = csc_array(sparse_mat)
    return _update_csc_array_indices(sparse_mat, shape, new_shape, offset)


class BlockLayout:
    """Named domains concatenated into one state, as ``np.concatenate(parts, axis)``.

    A monolithic model often stacks several domains (a particle and its bulk, a
    wall and a fluid, the fields of one grid, the regions of a membrane stack)
    into one state vector. ``BlockLayout`` derives where each domain sits from
    the concatenation, so no offset or flat index is typed by hand, and places
    domain-by-domain blocks of the Jacobian into the full system.

    "Block" here means a domain-by-domain sub-matrix of any size, not a
    block-sparse storage format such as SciPy's ``bsr_array``.

    Parameters
    ----------
    domains : dict[str, tuple[int, ...]]
        Domain name to domain shape, in storage order.
    axis : int or None
        Keyword-only and required. With an integer ``k`` the state has the
        shape of ``np.concatenate(parts, axis=k)``: all domains have the same
        number of dimensions and equal sizes on every other axis. ``axis=1``
        stacks particle cells and a bulk cell, ``axis=-1`` stacks fields on one
        grid, an ``(n, 2)`` state from two ``(n, 1)`` domains. With ``None`` the
        layout is region-major: the state is the concatenation of the
        flattened domains, shape ``(size,)``, and the domains may differ in
        every dimension (for example in their number of fields).

    Attributes
    ----------
    shape : tuple[int, ...]
        Shape of the state, ``(size,)`` for ``axis=None``.
    size : int
        Total number of unknowns.
    shapes : dict[str, tuple[int, ...]]
        Domain (and subdomain, see :meth:`subdomain`) name to shape.

    Raises
    ------
    ValueError
        If a size is not a positive integer, the axis is out of range, or the
        domains differ on a non-concatenation axis.

    Notes
    -----
    Arbitrary offset boxes are not supported: a gap would leave unknowns
    without equations and an overlap is ambiguous; the concatenation rules
    out both.

    Jacobians of local terms: with an integer axis use ``NumJac(layout.shape)``
    on the full state, as usual. With ``axis=None`` apply ``NumJac`` per domain
    on ``layout.shapes[name]`` to ``layout.view(u, name)`` and place the result
    with ``layout.place(jac, name, name)``. Never use ``NumJac`` on the flat
    ``(size,)`` shape: it couples its last axis in full and builds a dense
    Jacobian.

    Examples
    --------
    The packed bed of resolved particles, ``(n_z, n_r, m)`` particle cells and
    an ``(n_z, 1, m)`` bulk in one ``(n_z, n_r + 1, m)`` state (operators as in
    a model that places blocks with :func:`update_array_indices`)::

        layout = BlockLayout({"particle": shape_p, "bulk": shape_b}, axis=1)
        place = layout.place
        lin = (place(div_r @ flux, "particle", "particle")
               + place(div_r @ flux_b, "particle", "bulk")
               + place(div_z @ conv - 3 * alpha * outer @ flux_b, "bulk", "bulk")
               + place(-3 * alpha * outer @ flux, "bulk", "particle")).tocsc()
        g_0 = np.zeros(layout.size)
        layout.put(g_0, "bulk", (div_z @ conv_bc).toarray())
        numjac = NumJac(layout.shape)            # local terms on the full state

    Fields as domains on one grid, a state ``(n, 2) = [psi, xi]``
    (:meth:`from_fields` is short for ``{"psi": (n, 1), "xi": (n, 1)}, axis=-1``)::

        layout = BlockLayout.from_fields(["psi", "xi"], (n,))
        jac = layout.place(lap, "psi", "psi")    # lap built on the (n, 1) shape

    A wall reaction evaluated in the first fluid row of a channel, a part of a
    domain named with :meth:`subdomain`::

        layout = BlockLayout({"fluid": (n_x, n_y, 1)}, axis=1)
        layout.subdomain("near_wall", "fluid", np.s_[:, 0])
        rate_to_c = layout.place(-D * (div_y @ grad_r), "fluid", "near_wall")
        layout.add(g, "fluid", div_x @ flux)     # a term on the fluid rows only
    """

    def __init__(self, domains, *, axis):
        if not isinstance(domains, dict) or len(domains) == 0:
            raise ValueError("domains must be a non-empty dict of name -> shape")
        shapes = {}
        for name, shape in domains.items():
            shape = tuple(shape)
            if len(shape) == 0 or not all(
                isinstance(dim, numbers.Integral) and dim > 0 for dim in shape
            ):
                raise ValueError(
                    f"domain {name!r}: shape {shape} must be a non-empty tuple of positive ints"
                )
            shapes[name] = tuple(int(dim) for dim in shape)
        self.shapes = shapes
        self._domains = list(shapes)          # storage domains; subdomains are only added to shapes
        self._sizes = {name: math.prod(shape) for name, shape in shapes.items()}
        self.size = sum(self._sizes.values())

        if axis is None:
            # region-major: domain `name` is a contiguous stretch of the flat state
            self.axis = None
            self.shape = (self.size,)
            self._local, self._offset, self._box = {}, {}, {}
            start = 0
            for name, n in self._sizes.items():
                self._local[name] = (n,)
                self._offset[name] = (start,)
                self._box[name] = slice(start, start + n)
                start += n
            return

        ndims = {len(shape) for shape in shapes.values()}
        if len(ndims) != 1:
            raise ValueError(
                f"with axis={axis} all domains need the same number of dimensions, got {shapes}; "
                "use axis=None for a region-major layout"
            )
        ndim = ndims.pop()
        if not isinstance(axis, numbers.Integral) or not -ndim <= axis < ndim:
            raise ValueError(f"axis {axis} is out of range for {ndim}-dimensional domains")
        axis = int(axis) % ndim
        first = next(iter(shapes.values()))
        for name, shape in shapes.items():
            if any(s != f for i, (s, f) in enumerate(zip(shape, first)) if i != axis):
                raise ValueError(
                    f"domain {name!r} has shape {shape}; with axis={axis} all domains must "
                    f"match {first} on every other axis (the np.concatenate rule). For "
                    "different trailing sizes use axis=None (region-major), or stack fields "
                    "along axis=-1"
                )
        self.axis = axis
        self._local = dict(shapes)
        self._offset, self._box = {}, {}
        start = 0
        for name, shape in shapes.items():
            self._offset[name] = tuple(start if i == axis else 0 for i in range(ndim))
            self._box[name] = tuple(slice(o, o + s) for o, s in zip(self._offset[name], shape))
            start += shape[axis]
        self.shape = tuple(start if i == axis else s for i, s in enumerate(first))

    @classmethod
    def from_fields(cls, names, shape):
        """Layout of scalar fields on one grid, interleaved in a state ``shape + (len(names),)``.

        Shorthand for ``BlockLayout({name: shape + (1,) for name in names}, axis=-1)``.
        The trailing axis of length 1 is what makes the fields interleave: domains of
        shape ``(n,)`` with ``axis=-1`` would be stacked one after another, a ``(2n,)``
        state on which ``NumJac`` builds a dense Jacobian.

        Parameters
        ----------
        names : sequence of str
            Field names, in the order of the last axis of the state.
        shape : tuple[int, ...]
            Grid shape shared by the fields, for example ``(n,)`` or ``(n_x, n_y)``.

        Returns
        -------
        BlockLayout
        """
        shape = tuple(shape)
        return cls({name: shape + (1,) for name in names}, axis=-1)

    def __repr__(self):
        domains = {name: self.shapes[name] for name in self._domains}
        return f"BlockLayout({domains!r}, axis={self.axis!r})"

    def subdomain(self, name, parent, index):
        """Name a box of an existing domain, for example the cells next to a wall.

        The subdomain adds no unknowns: it is another name for part of ``parent``,
        and can be used in :meth:`place`, :meth:`assemble`, :meth:`view`,
        :meth:`put`, :meth:`add`, :meth:`pick` and :meth:`index`, so a block whose
        rows or columns are only some cells of a domain needs no offsets either.

        Parameters
        ----------
        name : str
            New name.
        parent : str
            Domain (or subdomain) the box lies in.
        index : int, slice or tuple of them
            Basic index into the parent's shape, with unit steps. An integer keeps
            its axis with length 1, so ``np.s_[:, 0]`` of an ``(n_x, n_y, 1)``
            domain is an ``(n_x, 1, 1)`` subdomain. With ``axis=None`` the box must
            be contiguous in C order (a range along the parent's first axis).

        Raises
        ------
        ValueError
            If the name exists, the index is not a non-empty unit-step box, or
            (``axis=None``) the box is not contiguous.
        """
        self._check_name(parent)
        if name in self.shapes:
            raise ValueError(f"name {name!r} is already used")
        parent_shape = self.shapes[parent]
        index = index if isinstance(index, tuple) else (index,)
        if len(index) > len(parent_shape):
            raise ValueError(f"index {index} has more axes than domain {parent!r} {parent_shape}")
        index = index + (slice(None),) * (len(parent_shape) - len(index))
        starts, shape = [], []
        for ix, dim in zip(index, parent_shape):
            if isinstance(ix, numbers.Integral):
                ix = int(ix) + (dim if ix < 0 else 0)
                if not 0 <= ix < dim:
                    raise ValueError(f"index {index} is out of range for {parent!r} {parent_shape}")
                ix = slice(ix, ix + 1)
            if not isinstance(ix, slice):
                raise ValueError("a subdomain index holds integers and slices only")
            start, stop, step = ix.indices(dim)
            if step != 1 or stop <= start:
                raise ValueError(f"index {index} is not a non-empty unit-step box of {parent!r}")
            starts.append(start)
            shape.append(stop - start)
        shape = tuple(shape)
        if self.axis is None:
            box = tuple(slice(s, s + n) for s, n in zip(starts, shape))
            flat = self.index(parent)[box].ravel()
            if np.any(np.diff(flat) != 1):
                raise ValueError(
                    f"subdomain {name!r} is not contiguous in the region-major state; "
                    "with axis=None take a range along the first axis of the domain"
                )
            start = int(flat[0])
            self._local[name], self._offset[name] = (flat.size,), (start,)
            self._box[name] = slice(start, start + flat.size)
        else:
            offset = tuple(o + s for o, s in zip(self._offset[parent], starts))
            self._local[name], self._offset[name] = shape, offset
            self._box[name] = tuple(slice(o, o + n) for o, n in zip(offset, shape))
        self.shapes[name] = shape
        self._sizes[name] = math.prod(shape)

    def _check_name(self, name):
        if name not in self.shapes:
            raise KeyError(f"unknown domain {name!r}; domains are {list(self.shapes)}")

    def _domain_slice(self, name):
        """Index into the state reshaped to ``self.shape`` that selects ``name``."""
        return self._box[name]

    def index(self, name):
        """Flat state indices of domain ``name``.

        Parameters
        ----------
        name : str
            Domain name.

        Returns
        -------
        numpy.ndarray
            Integer array of the domain's shape; entry ``[i, j, ...]`` is the
            position of that unknown in the flat state.
        """
        self._check_name(name)
        idx = np.arange(self.size).reshape(self.shape)[self._domain_slice(name)]
        return np.ascontiguousarray(idx).reshape(self.shapes[name])

    def view(self, u, name):
        """Writable view of domain ``name`` in the state ``u``, in the domain's shape.

        Parameters
        ----------
        u : numpy.ndarray
            State with ``size`` entries, of any shape, C-contiguous.
        name : str
            Domain name.

        Returns
        -------
        numpy.ndarray
            A view: writing to it writes into ``u``.

        Raises
        ------
        TypeError
            If ``u`` is not a NumPy array.
        ValueError
            If ``u`` has the wrong size, or is not C-contiguous (Fortran order,
            a strided slice): a view is then impossible and a copy would
            silently lose writes.
        """
        self._check_name(name)
        if not isinstance(u, np.ndarray):
            raise TypeError(f"u must be a numpy array, got {type(u).__name__}")
        if u.size != self.size:
            raise ValueError(f"u has {u.size} entries, the layout has {self.size}")
        if not u.flags.c_contiguous:
            raise ValueError("u is not C-contiguous; a view is impossible (use np.ascontiguousarray)")
        part = u.reshape(self.shape)[self._domain_slice(name)]
        if self.axis is None:
            part = part.reshape(self.shapes[name])  # contiguous slice: still a view
        return part

    def put(self, out, name, values):
        """Write ``values`` into the part ``name`` of the state ``out``.

        Parameters
        ----------
        out : numpy.ndarray
            State with ``size`` entries, C-contiguous; modified in place.
        name : str
            Domain name.
        values : array_like
            Any shape with the domain's number of entries, in the domain's
            C order (for example a column from ``(op @ c).toarray()``).

        Raises
        ------
        ValueError
            If ``values`` has the wrong number of entries, or ``out`` is not a
            valid state (see :meth:`view`).
        """
        part = self.view(out, name)
        part[...] = self._as_part(values, part, name)

    def add(self, out, name, values):
        """Add ``values`` to the part ``name`` of the state ``out``.

        As :meth:`put`, but accumulating: for a residual term that acts on one
        domain only, for example ``layout.add(g, "fluid", div_x @ flux)``.

        Parameters
        ----------
        out : numpy.ndarray
            State with ``size`` entries, C-contiguous; modified in place.
        name : str
            Domain name.
        values : array_like
            Any shape with the domain's number of entries, in the domain's
            C order.

        Raises
        ------
        ValueError
            As for :meth:`put`.
        """
        part = self.view(out, name)
        part += self._as_part(values, part, name)

    @staticmethod
    def _as_part(values, part, name):
        values = np.asarray(values)
        if values.size != part.size:
            raise ValueError(
                f"values has {values.size} entries, domain {name!r} has {part.size}"
            )
        return values.reshape(part.shape)

    def pick(self, name):
        """Selection matrix of domain ``name``.

        Parameters
        ----------
        name : str
            Domain name.

        Returns
        -------
        scipy.sparse.csr_array
            Shape ``(n_name, size)``: ``pick @ u`` extracts the domain in its C
            order, ``pick.T @ v`` scatters a domain vector into the state.
        """
        cols = self.index(name).ravel()
        return csr_array(
            (np.ones(cols.size), cols, np.arange(cols.size + 1)), shape=(cols.size, self.size)
        )

    def place(self, block, rows, cols):
        """Embed a domain-by-domain block into the full system.

        Sum the placed blocks of a model with ``+``, for example
        ``lin = layout.place(a, "x", "x") + layout.place(b, "x", "y")``; see
        :meth:`assemble` for systems of many blocks.

        Parameters
        ----------
        block : scipy.sparse.sparray or array_like
            Matrix with ``n_rows`` rows and ``n_cols`` columns, the numbers of
            unknowns of domains ``rows`` and ``cols``.
        rows, cols : str or None
            Domain names. ``None`` leaves that side as it is: a block whose
            rows (columns) already span the full state, or a selection whose
            columns (rows) stay local, for example
            ``place(eye_array(n), None, name)`` equals ``pick(name)``.

        Returns
        -------
        scipy.sparse.csr_array or scipy.sparse.csc_array
            CSR for CSR input, CSC otherwise; ``size`` rows unless ``rows`` is
            None, ``size`` columns unless ``cols`` is None.

        Raises
        ------
        ValueError
            If the block size does not match the named domain.
        KeyError
            If a name is unknown.
        """
        sides = []
        for side, name in ((0, rows), (1, cols)):
            if name is None:
                sides.append((None, None))
                continue
            self._check_name(name)
            n = self._sizes[name]
            if block.shape[side] != n:
                kind = "rows" if side == 0 else "columns"
                raise ValueError(
                    f"block has {block.shape[side]} {kind}, domain {name!r} has {n} unknowns"
                )
            sides.append((self._local[name], self._offset[name]))
        (shape_r, offset_r), (shape_c, offset_c) = sides
        # Always pass explicit (rows, cols) pairs: an all-integer tuple would be
        # read as "same for rows and columns". The CSR/CSC pointer rebuild in
        # update_array_indices needs a strictly increasing map from local to
        # state indices; concatenation guarantees it, so a permuting layout
        # must not be added through this path.
        return update_array_indices(
            block, (shape_r, shape_c), (self.shape, self.shape), offset=(offset_r, offset_c)
        )

    def assemble(self, blocks):
        """Sum of placed blocks as one square matrix, for systems of many blocks.

        For a handful of blocks, sum the results of :meth:`place` with ``+``: it
        is as fast or faster and reads like the equations. Each ``+`` rebuilds
        the sum, so with dozens of blocks or more ``assemble``, which builds it in
        one pass, is faster (about 3x at 144 blocks).

        Parameters
        ----------
        blocks : iterable of (block, rows, cols)
            Each tuple as for :meth:`place`; with ``None`` the block must
            already span the full state on that side.

        Returns
        -------
        scipy.sparse.csc_array
            ``size x size``; entries placed more than once are summed, and
            explicit zeros (stored zeros of the blocks, and sums that cancel) are
            removed, as in a sum of sparse arrays. An empty list gives a zero
            matrix.
        """
        data, row, col = [], [], []
        for block, rows, cols in blocks:
            placed = self.place(block, rows, cols)
            if placed.shape != (self.size, self.size):
                raise ValueError(
                    f"placed block ({rows!r}, {cols!r}) has shape {placed.shape}, "
                    f"expected {(self.size, self.size)}"
                )
            placed = placed.tocoo()
            data.append(placed.data)
            row.append(placed.row)
            col.append(placed.col)
        if not data:
            return csc_array((self.size, self.size))
        # COO -> CSC sums duplicate entries; the zeros left are not needed by any solver
        jac = coo_array(
            (np.concatenate(data), (np.concatenate(row), np.concatenate(col))),
            shape=(self.size, self.size),
        ).tocsc()
        jac.eliminate_zeros()
        return jac


def construct_interface_matrices(
    shapes,
    x_fs,
    x_cs=(None, None),
    ic=({"a": (1, 1), "b": (0, 0), "d": 0}, {"a": (0, 0), "b": (1, -1), "d": 0}),
    axis=0,
    shapes_d=(None, None),
    format="csc",
):
    """Construct implicit interface-coupling matrices for two adjacent domains.

    Parameters
    ----------
    shapes : tuple[tuple[int, ...], tuple[int, ...]]
        Shapes of the two subdomains.
    x_fs : tuple[array_like, array_like]
        Face coordinates for each subdomain along ``axis``.
    x_cs : tuple[array_like | None, array_like | None], optional
        Cell-center coordinates for each subdomain.
    ic : tuple[dict, dict], optional
        Two interface equations. Each dictionary may define ``a``, ``b``, and
        ``d`` terms with coefficients for both subdomains.
    axis : int, optional
        Interface-normal axis.
    shapes_d : tuple[tuple | None, tuple | None], optional
        Optional source-vector shapes for decomposed inhomogeneous terms.
    format : {'csc', 'csr'}, optional
        Sparse format for the homogeneous interface matrices.

    Returns
    -------
    tuple
        Without ``shapes_d``:
        ``(mat0, bc0, mat1, bc1)``.
        With ``shapes_d``:
        ``(mat0, bc00, bc01, mat1, bc10, bc11)``.
    """

    if not all(
        s1 == s2 for i, (s1, s2) in enumerate(zip(shapes[0], shapes[1])) if i != axis
    ):
        raise ValueError(
            "Tuples shapes[0] and shapes[1] must be equal except for the specified axis."
        )
    shape = tuple(
        s1 + s2 if i == axis else s1
        for i, (s1, s2) in enumerate(zip(shapes[0], shapes[1]))
    )
    shape_i = tuple(1 if i == axis else s for i, s in enumerate(shape))

    # Extract the cell-centered grid points for the two subdomains
    for i in range(2):
        if x_cs[i] is None:
            x_cs = list(x_cs)
            x_cs[i] = 0.5 * (x_fs[i][1:] + x_fs[i][:-1])

    a, b = [
        [
            tuple(
                (
                    unwrap_bc_coeff(shape, ic_elem.get(key, (0, 0))[j], axis=axis)
                    if ic_elem and key in ic_elem
                    else np.zeros((1,) * len(shape_i))
                )
                for j in range(2)
            )
            for ic_elem in ic
        ]
        for key in ["a", "b"]
    ]

    d = [
        (
            unwrap_bc_coeff(shape, ic_elem.get("d", 0), axis=axis)
            if ic_elem and "d" in ic_elem
            else np.zeros((1,) * len(shape_i))
        )
        for ic_elem in ic
    ]

    alpha_1 = [None, None]
    alpha_1[0] = -(x_cs[0][-2] - x_fs[0][-1]) / (
        (x_cs[0][-1] - x_fs[0][-1]) * (x_cs[0][-2] - x_cs[0][-1])
    )
    alpha_1[1] = (x_cs[1][1] - x_fs[1][0]) / (
        (x_cs[1][0] - x_fs[1][0]) * (x_cs[1][1] - x_cs[1][0])
    )
    alpha_2 = [None, None]
    alpha_2[0] = -(x_cs[0][-1] - x_fs[0][-1]) / (
        (x_cs[0][-2] - x_fs[0][-1]) * (x_cs[0][-2] - x_cs[0][-1])
    )
    alpha_2[1] = (x_cs[1][0] - x_fs[1][0]) / (
        (x_cs[1][1] - x_fs[1][0]) * (x_cs[1][1] - x_cs[1][0])
    )
    alpha_0 = [alpha_1[0] - alpha_2[0], alpha_1[1] - alpha_2[1]]

    # reminder of notation
    # dc/dn[0] = alpha_0[0] c_i - alpha_1[0] c[-1] + alpha_2[0] c[-2]
    # dc/dn[1] = -alpha_0[1] c_i + alpha_1[1] c[0] - alpha_2[1] c[1]
    # interface conditions of the form:

    m = [[None for _ in range(2)] for _ in range(2)]
    v = [[None for _ in range(4)] for _ in range(2)]
    m_inv = [[None for _ in range(2)] for _ in range(2)]
    values = [[None for _ in range(4)] for _ in range(2)]
    for i in range(2):  # loop over the two conditions
        m[i][0] = a[i][0] * alpha_0[0] + b[i][0]
        m[i][1] = a[i][1] * alpha_0[1] + b[i][1]
        v[i][0] = -a[i][0] * alpha_2[0]
        v[i][1] = a[i][0] * alpha_1[0]
        v[i][2] = a[i][1] * alpha_1[1]
        v[i][3] = -a[i][1] * alpha_2[1]
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    det_inv = np.where(det != 0.0, 1.0 / det, 0.0)
    m_inv[0][0] = m[1][1] * det_inv
    m_inv[0][1] = -m[0][1] * det_inv
    m_inv[1][0] = -m[1][0] * det_inv
    m_inv[1][1] = m[0][0] * det_inv
    for j in range(2):
        for i in range(4):
            values[j][i] = m_inv[j][0] * v[0][i] + m_inv[j][1] * v[1][i]
            values[j][i] = np.broadcast_to(values[j][i], shape_i)
        values[j] = np.concatenate(values[j], axis=axis)

    shape_t = [math.prod(shape[0:axis]), shape[axis], math.prod(shape[axis + 1:])]
    n0, n1, n2 = shape_t
    i0 = np.arange(n0).reshape(-1, 1, 1)
    i2 = np.arange(n2).reshape(1, 1, -1)
    # Row indices: flat index into the interface (n0 x n2) array
    row_indices = np.ravel_multi_index(
        (i0, np.zeros((1, 4, 1), dtype=int), i2), (n0, 1, n2)
    )
    # Column indices: 4-point stencil around the interface
    stencil = (shapes[0][axis] + np.array([-2, -1, 0, 1])).reshape(1, -1, 1)
    col_indices = np.ravel_multi_index((i0, stencil, i2), shape_t)

    # Create the sparse matrix representing the interface
    interface_matrix = [None] * 2
    for j in range(2):
        fltr = values[j].ravel() != 0
        values_filtered = values[j].ravel()[fltr]
        row_indices_filtered = row_indices.ravel()[fltr]
        col_indices_filtered = col_indices.ravel()[fltr]
        interface_matrix[j] = _sparse_array(
            (values_filtered, (row_indices_filtered, col_indices_filtered)),
            shape=(shape_t[0] * shape_t[2], math.prod(shape_t)),
            format=format,
        )

    row_indices_bc = np.ravel_multi_index(
        (np.arange(n0).reshape(-1, 1), np.arange(n2).reshape(1, -1)), (n0, n2)
    )
    if shapes_d[0] is None and shapes_d[1] is None:
        interface_bc = [None] * 2
        for j in range(2):
            values_bc = m_inv[j][0] * d[0] + m_inv[j][1] * d[1]
            values_bc = np.broadcast_to(values_bc, shape_i)
            fltr = values_bc.ravel() != 0
            values_filtered = values_bc.ravel()[fltr]
            row_indices_filtered = row_indices_bc.ravel()[fltr]
            interface_bc[j] = csc_array(
                (values_filtered, row_indices_filtered, np.array([0, row_indices_filtered.size])),
                shape=(shape_t[0] * shape_t[2], 1),
            )
        return (
            interface_matrix[0],
            interface_bc[0],
            interface_matrix[1],
            interface_bc[1],
        )
    else:
        interface_bc = [[None for _ in range(2)] for _ in range(2)]
        for j in range(2):
            for i in range(2):
                values_bc = m_inv[j][i] * d[i]
                values_bc = np.broadcast_to(values_bc, shape_i)
                fltr = values_bc.ravel() != 0
                values_filtered = values_bc.ravel()[fltr]
                row_indices_filtered = row_indices_bc.ravel()[fltr]
                if shapes_d[j] is None:
                    interface_bc[j][i] = csc_array(
                        (
                            values_filtered,
                            row_indices_filtered,
                            np.array([0, row_indices_filtered.size]),
                        ),
                        shape=(shape_t[0] * shape_t[2], 1),
                    )
                else:
                    num_cols = math.prod(shapes_d[j])
                    col_indices_bc = np.arange(num_cols, dtype=int).reshape(shapes_d[j])
                    col_indices_bc = np.broadcast_to(col_indices_bc, shape_i)
                    col_indices_filtered = col_indices_bc.ravel()[fltr]
                    interface_bc[j][i] = csc_array(
                        (values_filtered, (row_indices_filtered, col_indices_filtered)),
                        shape=(shape_t[0] * shape_t[2], num_cols),
                    )
        return (
            interface_matrix[0],
            interface_bc[0][0],
            interface_bc[0][1],
            interface_matrix[1],
            interface_bc[1][0],
            interface_bc[1][1],
        )
