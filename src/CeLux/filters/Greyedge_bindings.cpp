#include "Greyedge_bindings.hpp"

namespace py = pybind11;

void bind_Greyedge(py::module_ &m) {
    py::class_<Greyedge, FilterBase, std::shared_ptr<Greyedge>>(m, "Greyedge")
        .def(py::init<int, int, double>(),
             py::arg("difford") = 1,
             py::arg("minknorm") = 1,
             py::arg("sigma") = 1.00)
        .def("setDifford", &Greyedge::setDifford)
        .def("getDifford", &Greyedge::getDifford)
        .def("setMinknorm", &Greyedge::setMinknorm)
        .def("getMinknorm", &Greyedge::getMinknorm)
        .def("setSigma", &Greyedge::setSigma)
        .def("getSigma", &Greyedge::getSigma)
        ;
}