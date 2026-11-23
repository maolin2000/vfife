import numpy as np
import pytest

from vfife.core.dof import (DofLayout, ROTATION, SparseAssembler, TRANSLATION,
                            repeated_rotation)


def test_frame_layout_maps_named_fields_and_elements():
    layout = DofLayout.frame(3)
    assert layout.size == 18
    assert layout.node_dofs([0, 2], TRANSLATION).tolist() == [[0, 1, 2], [12, 13, 14]]
    assert layout.node_dofs([1], ROTATION).tolist() == [[9, 10, 11]]
    assert layout.element_dofs([0, 2]).tolist() == list(range(6)) + list(range(12, 18))
    q = np.arange(18.0)
    assert np.array_equal(layout.field(q, ROTATION),
                          [[3, 4, 5], [9, 10, 11], [15, 16, 17]])


def test_named_nodal_assembly_does_not_touch_other_components():
    layout = DofLayout.frame(3)
    force = np.zeros(layout.size)
    layout.add_nodal(force, [0, 2], [1.0, 2.0, 3.0], TRANSLATION)
    assert np.array_equal(layout.field(force, TRANSLATION),
                          [[1, 2, 3], [0, 0, 0], [1, 2, 3]])
    assert not layout.field(force, ROTATION).any()


def test_sparse_assembler_uses_explicit_element_map_and_checks_shape():
    layout = DofLayout.translations(3)
    dofs = layout.element_dofs([0, 2])
    local = np.eye(6)
    assembly = SparseAssembler(layout.size)
    assembly.add_matrix(dofs, local)
    K = assembly.csr().toarray()
    assert np.array_equal(np.diag(K), [1, 1, 1, 0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="shape"):
        assembly.add_matrix(dofs, np.eye(5))


def test_layout_rejects_unknown_components_and_bad_nodes():
    layout = DofLayout.translations(2)
    with pytest.raises(ValueError, match="unknown DOF"):
        layout.node_dofs([0], ["rz"])
    with pytest.raises(IndexError, match="outside"):
        layout.node_dofs([2])


def test_repeated_rotation_is_the_exact_node_major_transform():
    R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    T = repeated_rotation(R)
    assert T.shape == (12, 12)
    assert np.array_equal(T @ np.arange(12.0),
                          np.concatenate([R @ np.arange(i, i+3.0) for i in range(0, 12, 3)]))
