#include "Yadif_bindings.hpp"

namespace py = pybind11;

void bind_Yadif(py::module_ &m) {
    py::class_<Yadif, FilterBase, std::shared_ptr<Yadif>>(m, "Yadif")
        .def(py::init<int, int, int>(),
             py::arg("mode") = 0,
             py::arg("parity") = -1,
             py::arg("deint") = 0)
        .def("setMode", &Yadif::setMode)
        .def("getMode", &Yadif::getMode)
        .def("setParity", &Yadif::setParity)
        .def("getParity", &Yadif::getParity)
        .def("setDeint", &Yadif::setDeint)
        .def("getDeint", &Yadif::getDeint)
        ;
}