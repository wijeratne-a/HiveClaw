#include <nanobind/nanobind.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <mlx/array.h>

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

NB_MODULE(hiveclaw_mlx_ext, m) {
    // Ensure mlx.core is loaded first so NB_DOMAIN mlx recognizes Python mlx.core.array.
    (void)nb::module_::import_("mlx.core");

    m.def("get_latent_dim", []() { return static_cast<int>(HCLW_SCENT_ELEMS); });

    nb::class_<SlabHandle>(m, "SlabHandle")
        .def(nb::init<uint32_t>(), nb::arg("surface_id"))
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
