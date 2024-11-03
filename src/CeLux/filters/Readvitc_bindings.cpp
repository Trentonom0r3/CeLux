#include "Readvitc_bindings.hpp"

namespace py = pybind11;

void bind_Readvitc(py::module_ &m) {
    py::class_<Readvitc, FilterBase, std::shared_ptr<Readvitc>>(m, "Readvitc")
        .def(py::init<int, double, double>(),
             py::arg("scan_max") = 45,
             py::arg("thr_b") = 0.20,
             py::arg("thr_w") = 0.60)
        .def("setScan_max", &Readvitc::setScan_max)
        .def("getScan_max", &Readvitc::getScan_max)
        .def("setThr_b", &Readvitc::setThr_b)
        .def("getThr_b", &Readvitc::getThr_b)
        .def("setThr_w", &Readvitc::setThr_w)
        .def("getThr_w", &Readvitc::getThr_w)
        ;
}