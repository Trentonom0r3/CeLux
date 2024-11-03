#include "Extractplanes_bindings.hpp"

namespace py = pybind11;

void bind_Extractplanes(py::module_ &m) {
    py::class_<Extractplanes, FilterBase, std::shared_ptr<Extractplanes>>(m, "Extractplanes")
        .def(py::init<int>(),
             py::arg("planes") = 1)
        .def("setPlanes", &Extractplanes::setPlanes)
        .def("getPlanes", &Extractplanes::getPlanes)
        ;
}