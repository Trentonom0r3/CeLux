#include "Exposure_bindings.hpp"

namespace py = pybind11;

void bind_Exposure(py::module_ &m) {
    py::class_<Exposure, FilterBase, std::shared_ptr<Exposure>>(m, "Exposure")
        .def(py::init<float, float>(),
             py::arg("exposure") = 0.00,
             py::arg("black") = 0.00)
        .def("setExposure", &Exposure::setExposure)
        .def("getExposure", &Exposure::getExposure)
        .def("setBlack", &Exposure::setBlack)
        .def("getBlack", &Exposure::getBlack)
        ;
}