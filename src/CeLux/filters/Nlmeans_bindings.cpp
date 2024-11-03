#include "Nlmeans_bindings.hpp"

namespace py = pybind11;

void bind_Nlmeans(py::module_ &m) {
    py::class_<Nlmeans, FilterBase, std::shared_ptr<Nlmeans>>(m, "Nlmeans")
        .def(py::init<double, int, int, int, int>(),
             py::arg("denoisingStrength") = 1.00,
             py::arg("patchSize") = 7,
             py::arg("pc") = 0,
             py::arg("researchWindow") = 15,
             py::arg("rc") = 0)
        .def("setDenoisingStrength", &Nlmeans::setDenoisingStrength)
        .def("getDenoisingStrength", &Nlmeans::getDenoisingStrength)
        .def("setPatchSize", &Nlmeans::setPatchSize)
        .def("getPatchSize", &Nlmeans::getPatchSize)
        .def("setPc", &Nlmeans::setPc)
        .def("getPc", &Nlmeans::getPc)
        .def("setResearchWindow", &Nlmeans::setResearchWindow)
        .def("getResearchWindow", &Nlmeans::getResearchWindow)
        .def("setRc", &Nlmeans::setRc)
        .def("getRc", &Nlmeans::getRc)
        ;
}