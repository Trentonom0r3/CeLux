#include "Lenscorrection_bindings.hpp"

namespace py = pybind11;

void bind_Lenscorrection(py::module_ &m) {
    py::class_<Lenscorrection, FilterBase, std::shared_ptr<Lenscorrection>>(m, "Lenscorrection")
        .def(py::init<double, double, double, double, int, std::string>(),
             py::arg("cx") = 0.50,
             py::arg("cy") = 0.50,
             py::arg("k1") = 0.00,
             py::arg("k2") = 0.00,
             py::arg("interpolationType") = 0,
             py::arg("fc") = "black@0")
        .def("setCx", &Lenscorrection::setCx)
        .def("getCx", &Lenscorrection::getCx)
        .def("setCy", &Lenscorrection::setCy)
        .def("getCy", &Lenscorrection::getCy)
        .def("setK1", &Lenscorrection::setK1)
        .def("getK1", &Lenscorrection::getK1)
        .def("setK2", &Lenscorrection::setK2)
        .def("getK2", &Lenscorrection::getK2)
        .def("setInterpolationType", &Lenscorrection::setInterpolationType)
        .def("getInterpolationType", &Lenscorrection::getInterpolationType)
        .def("setFc", &Lenscorrection::setFc)
        .def("getFc", &Lenscorrection::getFc)
        ;
}