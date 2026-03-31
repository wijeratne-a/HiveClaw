#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <mlx/array.h>

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

NB_MODULE(hiveclaw_mlx_ext, m) {
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
            [](SlabHandle& self,
               size_t byte_offset,
               std::vector<int> shape,
               array like,
               std::optional<array> dep) {
                return self.read(
                    byte_offset,
                    list_to_shape(shape),
                    std::move(like),
                    std::move(dep));
            },
            nb::arg("byte_offset"),
            nb::arg("shape"),
            nb::arg("like"),
            nb::arg("dep") = nb::none())
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
            "read_slot",
            [](SlabHandle& self,
               uint32_t slot_index,
               std::vector<int> shape,
               array like,
               std::optional<array> dep) {
                return self.read_slot(
                    slot_index,
                    list_to_shape(shape),
                    std::move(like),
                    std::move(dep));
            },
            nb::arg("slot_index"),
            nb::arg("shape"),
            nb::arg("like"),
            nb::arg("dep") = nb::none())
        .def(
            "claim",
            [](SlabHandle& self,
               array candidates,
               uint32_t agent_id,
               std::optional<array> dep) {
                return self.claim(std::move(candidates), agent_id, std::move(dep));
            },
            nb::arg("candidates"),
            nb::arg("agent_id"),
            nb::arg("dep") = nb::none())
        .def(
            "inhibit",
            [](SlabHandle& self,
               uint32_t slot_index,
               uint32_t agent_id,
               std::optional<array> dep) {
                return self.inhibit(slot_index, agent_id, std::move(dep));
            },
            nb::arg("slot_index"),
            nb::arg("agent_id"),
            nb::arg("dep") = nb::none())
        .def("release_slot", &SlabHandle::release_slot, nb::arg("slot_index"));
}
