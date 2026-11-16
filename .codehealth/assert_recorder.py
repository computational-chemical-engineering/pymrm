"""pytest plugin for the codehealth reference tier: record every tolerance comparison.

Loaded with ``-p assert_recorder`` (PYTHONPATH includes ``.codehealth``). Whenever a test
compares numbers with a tolerance (``np.testing.assert_allclose``, ``np.allclose``,
``np.isclose``, ``pytest.approx``), both operands are recorded, keyed by test node id and call
index. The records are written to ``$CODEHEALTH_RECORD`` as one deterministic binary file, so
the gate compares them byte for byte between the campaign base and a step: a change in
round-off that the tolerance would hide fails the reference tier.

Only calls made directly from a test file (``test/``) are recorded, not calls inside pymrm.
Works with pytest-xdist: each worker writes a part file, the controller merges them.
"""
import hashlib
import os
import pickle
import sys

import numpy as np
import pytest

_RECORDS = []          # (key, payload bytes)
_STATE = {"node": "<collect>", "count": 0}
_TEST_DIR = os.sep + "test" + os.sep


def _encode(value):
    """Deterministic bytes for a compared operand."""
    try:
        import scipy.sparse as sp
        if sp.issparse(value):
            m = value.tocsr()
            m.sort_indices()
            return b"sparse" + _encode(m.data) + _encode(m.indices) + _encode(m.indptr) \
                + repr(m.shape).encode()
    except ImportError:
        pass
    try:
        a = np.asarray(value)
    except Exception:
        return b"repr:" + repr(value).encode()
    if a.dtype == object:
        return b"obj:" + repr(value).encode()
    a = np.ascontiguousarray(a)
    return (a.dtype.str + repr(a.shape)).encode() + b":" + a.tobytes()


def _from_test(depth):
    try:
        fname = sys._getframe(depth).f_code.co_filename
    except ValueError:
        return False
    return _TEST_DIR in fname


def _record(kind, *operands):
    key = f"{_STATE['node']}#{_STATE['count']}:{kind}"
    _STATE["count"] += 1
    _RECORDS.append((key, b"|".join(_encode(v) for v in operands)))


def _wrap(module, name, kind, nargs):
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        if _from_test(2):
            _record(kind, *args[:nargs])
        return original(*args, **kwargs)

    wrapper.__wrapped__ = original
    setattr(module, name, wrapper)


def _wrap_approx():
    from _pytest import python_api
    for cls in list(vars(python_api).values()):
        if isinstance(cls, type) and issubclass(cls, python_api.ApproxBase) \
                and "__eq__" in vars(cls):
            original = cls.__eq__

            def __eq__(self, actual, _orig=original):
                if _from_test(2):
                    _record("approx", self.expected, actual)
                return _orig(self, actual)

            cls.__eq__ = __eq__


def pytest_configure(config):
    _wrap(np.testing, "assert_allclose", "assert_allclose", 2)
    _wrap(np, "allclose", "allclose", 2)
    _wrap(np, "isclose", "isclose", 2)
    _wrap_approx()


_SPREAD = ("test_plugin_exemplars.py",)   # independent slow tests, spread over workers


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Keep each test file on one xdist worker, in file order (run with --dist loadgroup).

    Some test modules draw from a module-level random generator, so a test's values depend
    on which tests of that module ran before it in the same process. Grouping by file makes
    the records independent of how xdist distributes the tests."""
    for item in items:
        fname = os.path.basename(str(item.path))
        if fname not in _SPREAD:
            item.add_marker(pytest.mark.xdist_group(name=fname))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _STATE["node"] = item.nodeid.split("@")[0]   # xdist loadgroup appends @group
    _STATE["count"] = 0


def _target():
    return os.environ.get("CODEHEALTH_RECORD")


def pytest_sessionfinish(session):
    target = _target()
    if not target:
        return
    worker = getattr(session.config, "workerinput", None)
    if worker is not None:
        part = f"{target}.part-{worker['workerid']}"
        with open(part, "wb") as fh:
            pickle.dump(_RECORDS, fh, protocol=4)
        return
    records = list(_RECORDS)
    folder = os.path.dirname(os.path.abspath(target)) or "."
    base = os.path.basename(target) + ".part-"
    for name in sorted(os.listdir(folder)):
        if name.startswith(base):
            path = os.path.join(folder, name)
            with open(path, "rb") as fh:
                records.extend(pickle.load(fh))
            os.remove(path)
    records.sort(key=lambda r: r[0])
    keys = [k for k, _ in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("assert_recorder: duplicate record keys")
    with open(target, "wb") as fh:
        for key, payload in records:
            kb = key.encode()
            fh.write(len(kb).to_bytes(4, "little") + kb)
            fh.write(len(payload).to_bytes(8, "little") + payload)
    with open(target, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    print(f"\nassert_recorder: {len(records)} records, sha256 {digest}")
