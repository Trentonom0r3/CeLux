#include "Vflip_bindings.hpp"

namespace py = pybind11;

void bind_Vflip(py::module_ &m) {
    py::class_<Vflip, FilterBase, std::shared_ptr<Vflip>>(m, "Vflip")
        .def(py::init<>())
        ;
}