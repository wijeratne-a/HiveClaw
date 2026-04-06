#include <nanobind/nanobind.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <vector>

#include <mlx/array.h>
#include <mlx/mlx.h>

#include <stdexcept>

#include "slab_layout.h"
#include "slab_primitives.h"

namespace nb = nanobind;

using mlx::core::Shape;
using mlx::core::array;

static Shape list_to_shape(const std::vector<int>& shape) {
    Shape out;
    out.reserve(static_cast<int>(shape.size()));
    for (int d : shape) {
        out.push_back(static_cast<mlx::core::ShapeElem>(d));
    }
    return out;
}

/// NB_DOMAIN mlx: std::vector<int> does not bind from Python list; use nb::list / nb::tuple.
template <typename Seq>
static Shape shape_from_seq(const Seq& s) {
    std::vector<int> v;
    v.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        v.push_back(nb::cast<int>(s[i]));
    }
    return list_to_shape(v);
}

static std::vector<uint32_t> slots_array_to_indices(const array& slot_indices) {
    mlx::core::eval({slot_indices});
    if (slot_indices.dtype() != mlx::core::int32) {
        throw std::invalid_argument("slot_indices must be int32 mlx.array");
    }
    if (slot_indices.ndim() != 1) {
        throw std::invalid_argument("slot_indices must be 1-D [B]");
    }
    const int32_t* p = slot_indices.data<int32_t>();
    const size_t n = static_cast<size_t>(slot_indices.size());
    std::vector<uint32_t> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        // int32 -1 → uint32 0xFFFFFFFF (batched sentinel; validated in C++).
        out.push_back(static_cast<uint32_t>(p[i]));
    }
    return out;
}

NB_MODULE(hiveclaw_mlx_ext, m) {
    // Ensure mlx.core is loaded first so NB_DOMAIN mlx recognizes Python mlx.core.array.
    (void)nb::module_::import_("mlx.core");

    nb::class_<SlabHandle>(m, "SlabHandle")
        .def(nb::init<uint32_t>(), nb::arg("surface_id"))
        .def("get_latent_dim", &SlabHandle::get_latent_dim)
        .def(
            "write",
            [](SlabHandle& self,
               size_t byte_offset,
               array scent_c,
               std::optional<array> dep) {
                return self.write(byte_offset, std::move(scent_c), std::move(dep));
            },
            nb::arg("byte_offset"),
            nb::arg("scent_c"),
            nb::arg("dep") = nb::none())
        .def(
            "read",
            [](SlabHandle& self, size_t byte_offset, nb::list shape_list) {
                Shape sh = shape_from_seq(shape_list);
                return self.read(byte_offset, sh, std::nullopt);
            },
            nb::arg("byte_offset"),
            nb::arg("shape"))
        .def(
            "read",
            [](SlabHandle& self,
               size_t byte_offset,
               nb::list shape_list,
               array dep) {
                Shape sh = shape_from_seq(shape_list);
                return self.read(byte_offset, sh, std::move(dep));
            },
            nb::arg("byte_offset"),
            nb::arg("shape"),
            nb::arg("dep"))
        .def(
            "read",
            [](SlabHandle& self, size_t byte_offset, nb::tuple shape_tuple) {
                Shape sh = shape_from_seq(shape_tuple);
                return self.read(byte_offset, sh, std::nullopt);
            },
            nb::arg("byte_offset"),
            nb::arg("shape"))
        .def(
            "read",
            [](SlabHandle& self,
               size_t byte_offset,
               nb::tuple shape_tuple,
               array dep) {
                Shape sh = shape_from_seq(shape_tuple);
                return self.read(byte_offset, sh, std::move(dep));
            },
            nb::arg("byte_offset"),
            nb::arg("shape"),
            nb::arg("dep"))
        .def(
            "write_slot",
            [](SlabHandle& self,
               uint32_t slot_index,
               array scent_c,
               std::optional<array> dep) {
                return self.write_slot(slot_index, std::move(scent_c), std::move(dep));
            },
            nb::arg("slot_index"),
            nb::arg("scent_c"),
            nb::arg("dep") = nb::none())
        .def(
            "write_slot_v5",
            [](SlabHandle& self,
               uint32_t slot_index,
               array latent,
               std::optional<array> dep) {
                return self.write_slot_v5(slot_index, std::move(latent), std::move(dep));
            },
            nb::arg("slot_index"),
            nb::arg("latent"),
            nb::arg("dep") = nb::none())
        .def(
            "read_slot",
            [](SlabHandle& self, uint32_t slot_index, nb::list shape_list) {
                Shape sh = shape_from_seq(shape_list);
                return self.read_slot(slot_index, sh, std::nullopt);
            },
            nb::arg("slot_index"),
            nb::arg("shape"))
        .def(
            "read_slot",
            [](SlabHandle& self,
               uint32_t slot_index,
               nb::list shape_list,
               array dep) {
                Shape sh = shape_from_seq(shape_list);
                return self.read_slot(slot_index, sh, std::move(dep));
            },
            nb::arg("slot_index"),
            nb::arg("shape"),
            nb::arg("dep"))
        .def(
            "read_slot",
            [](SlabHandle& self, uint32_t slot_index, nb::tuple shape_tuple) {
                Shape sh = shape_from_seq(shape_tuple);
                return self.read_slot(slot_index, sh, std::nullopt);
            },
            nb::arg("slot_index"),
            nb::arg("shape"))
        .def(
            "read_slot",
            [](SlabHandle& self,
               uint32_t slot_index,
               nb::tuple shape_tuple,
               array dep) {
                Shape sh = shape_from_seq(shape_tuple);
                return self.read_slot(slot_index, sh, std::move(dep));
            },
            nb::arg("slot_index"),
            nb::arg("shape"),
            nb::arg("dep"))
        .def(
            "read_slot_v5",
            [](SlabHandle& self, uint32_t slot_index) {
                return self.read_slot_v5(slot_index, std::nullopt);
            },
            nb::arg("slot_index"))
        .def(
            "read_slot_v5",
            [](SlabHandle& self, uint32_t slot_index, array dep) {
                return self.read_slot_v5(slot_index, std::move(dep));
            },
            nb::arg("slot_index"),
            nb::arg("dep"))
        .def(
            "read_slots_v5",
            [](SlabHandle& self, array slot_indices) {
                return self.read_slots_v5(
                    slots_array_to_indices(slot_indices), std::nullopt);
            },
            nb::arg("slot_indices"))
        .def(
            "read_slots_v5",
            [](SlabHandle& self, array slot_indices, array dep) {
                return self.read_slots_v5(
                    slots_array_to_indices(slot_indices), std::move(dep));
            },
            nb::arg("slot_indices"),
            nb::arg("dep"))
        .def(
            "write_slots_v5",
            [](SlabHandle& self, array slot_indices, array latents) {
                return self.write_slots_v5(
                    slots_array_to_indices(slot_indices),
                    std::move(latents),
                    std::nullopt);
            },
            nb::arg("slot_indices"),
            nb::arg("latents"))
        .def(
            "write_slots_v5",
            [](SlabHandle& self,
               array slot_indices,
               array latents,
               array dep) {
                return self.write_slots_v5(
                    slots_array_to_indices(slot_indices),
                    std::move(latents),
                    std::move(dep));
            },
            nb::arg("slot_indices"),
            nb::arg("latents"),
            nb::arg("dep"))
        .def(
            "claim",
            [](SlabHandle& self, array candidates, uint32_t agent_id) {
                return self.claim(std::move(candidates), agent_id, std::nullopt);
            },
            nb::arg("candidates"),
            nb::arg("agent_id"))
        .def(
            "claim",
            [](SlabHandle& self, array candidates, uint32_t agent_id, array dep) {
                return self.claim(std::move(candidates), agent_id, std::move(dep));
            },
            nb::arg("candidates"),
            nb::arg("agent_id"),
            nb::arg("dep"))
        .def(
            "inhibit",
            [](SlabHandle& self, uint32_t slot_index, uint32_t agent_id) {
                return self.inhibit(slot_index, agent_id, std::nullopt);
            },
            nb::arg("slot_index"),
            nb::arg("agent_id"))
        .def(
            "inhibit",
            [](SlabHandle& self, uint32_t slot_index, uint32_t agent_id, array dep) {
                return self.inhibit(slot_index, agent_id, std::move(dep));
            },
            nb::arg("slot_index"),
            nb::arg("agent_id"),
            nb::arg("dep"))
        .def("get_slot_states", [](SlabHandle& self) { return self.get_slot_states(); })
        .def("release_slot", &SlabHandle::release_slot, nb::arg("slot_index"));
}
