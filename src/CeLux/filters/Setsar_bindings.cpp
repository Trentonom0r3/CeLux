#include "Setsar_bindings.hpp"

namespace py = pybind11;

void bind_Setsar(py::module_ &m) {
    py::class_<Setsar, FilterBase, std::shared_ptr<Setsar>>(m, "Setsar")
        .def(py::init<std::string, int>(),
             py::arg("ratio") = "0",
             py::arg("max") = 100)
        .def("setRatio", &Setsar::setRatio)
        .def("getRatio", &Setsar::getRatio)
        .def("setMax", &Setsar::setMax)
        .def("getMax", &Setsar::getMax)
        ;
}