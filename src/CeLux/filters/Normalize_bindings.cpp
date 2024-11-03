#include "Normalize_bindings.hpp"

namespace py = pybind11;

void bind_Normalize(py::module_ &m) {
    py::class_<Normalize, FilterBase, std::shared_ptr<Normalize>>(m, "Normalize")
        .def(py::init<std::string, std::string, int, float, float>(),
             py::arg("blackpt") = "black",
             py::arg("whitept") = "white",
             py::arg("smoothing") = 0,
             py::arg("independence") = 1.00,
             py::arg("strength") = 1.00)
        .def("setBlackpt", &Normalize::setBlackpt)
        .def("getBlackpt", &Normalize::getBlackpt)
        .def("setWhitept", &Normalize::setWhitept)
        .def("getWhitept", &Normalize::getWhitept)
        .def("setSmoothing", &Normalize::setSmoothing)
        .def("getSmoothing", &Normalize::getSmoothing)
        .def("setIndependence", &Normalize::setIndependence)
        .def("getIndependence", &Normalize::getIndependence)
        .def("setStrength", &Normalize::setStrength)
        .def("getStrength", &Normalize::getStrength)
        ;
}