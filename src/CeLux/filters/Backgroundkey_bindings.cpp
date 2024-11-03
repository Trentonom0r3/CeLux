#include "Backgroundkey_bindings.hpp"

namespace py = pybind11;

void bind_Backgroundkey(py::module_ &m) {
    py::class_<Backgroundkey, FilterBase, std::shared_ptr<Backgroundkey>>(m, "Backgroundkey")
        .def(py::init<float, float, float>(),
             py::arg("threshold") = 0.08,
             py::arg("similarity") = 0.10,
             py::arg("blend") = 0.00)
        .def("setThreshold", &Backgroundkey::setThreshold)
        .def("getThreshold", &Backgroundkey::getThreshold)
        .def("setSimilarity", &Backgroundkey::setSimilarity)
        .def("getSimilarity", &Backgroundkey::getSimilarity)
        .def("setBlend", &Backgroundkey::setBlend)
        .def("getBlend", &Backgroundkey::getBlend)
        ;
}