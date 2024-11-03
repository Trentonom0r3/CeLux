#include "Hflip_bindings.hpp"

namespace py = pybind11;

void bind_Hflip(py::module_ &m) {
    py::class_<Hflip, FilterBase, std::shared_ptr<Hflip>>(m, "Hflip")
        .def(py::init<>())
        ;
}