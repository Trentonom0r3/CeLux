#include "Scdet_bindings.hpp"

namespace py = pybind11;

void bind_Scdet(py::module_ &m) {
    py::class_<Scdet, FilterBase, std::shared_ptr<Scdet>>(m, "Scdet")
        .def(py::init<double, bool>(),
             py::arg("threshold") = 10.00,
             py::arg("sc_pass") = false)
        .def("setThreshold", &Scdet::setThreshold)
        .def("getThreshold", &Scdet::getThreshold)
        .def("setSc_pass", &Scdet::setSc_pass)
        .def("getSc_pass", &Scdet::getSc_pass)
        ;
}