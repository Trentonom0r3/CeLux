#include "Lumakey_bindings.hpp"

namespace py = pybind11;

void bind_Lumakey(py::module_ &m) {
    py::class_<Lumakey, FilterBase, std::shared_ptr<Lumakey>>(m, "Lumakey")
        .def(py::init<double, double, double>(),
             py::arg("threshold") = 0.00,
             py::arg("tolerance") = 0.01,
             py::arg("softness") = 0.00)
        .def("setThreshold", &Lumakey::setThreshold)
        .def("getThreshold", &Lumakey::getThreshold)
        .def("setTolerance", &Lumakey::setTolerance)
        .def("getTolerance", &Lumakey::getTolerance)
        .def("setSoftness", &Lumakey::setSoftness)
        .def("getSoftness", &Lumakey::getSoftness)
        ;
}