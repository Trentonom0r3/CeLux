#include "Ashowinfo_bindings.hpp"

namespace py = pybind11;

void bind_Ashowinfo(py::module_ &m) {
    py::class_<Ashowinfo, FilterBase, std::shared_ptr<Ashowinfo>>(m, "Ashowinfo")
        .def(py::init<>())
        ;
}