#include "Dedot_bindings.hpp"

namespace py = pybind11;

void bind_Dedot(py::module_ &m) {
    py::class_<Dedot, FilterBase, std::shared_ptr<Dedot>>(m, "Dedot")
        .def(py::init<int, float, float, float, float>(),
             py::arg("filteringMode") = 3,
             py::arg("lt") = 0.08,
             py::arg("tl") = 0.08,
             py::arg("tc") = 0.06,
             py::arg("ct") = 0.02)
        .def("setFilteringMode", &Dedot::setFilteringMode)
        .def("getFilteringMode", &Dedot::getFilteringMode)
        .def("setLt", &Dedot::setLt)
        .def("getLt", &Dedot::getLt)
        .def("setTl", &Dedot::setTl)
        .def("getTl", &Dedot::getTl)
        .def("setTc", &Dedot::setTc)
        .def("getTc", &Dedot::getTc)
        .def("setCt", &Dedot::setCt)
        .def("getCt", &Dedot::getCt)
        ;
}