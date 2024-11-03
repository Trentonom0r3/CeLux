#include "Xstack_bindings.hpp"

namespace py = pybind11;

void bind_Xstack(py::module_ &m) {
    py::class_<Xstack, FilterBase, std::shared_ptr<Xstack>>(m, "Xstack")
        .def(py::init<int, std::string, std::pair<int, int>, bool, std::string>(),
             py::arg("inputs") = 2,
             py::arg("layout") = "",
             py::arg("grid") = std::make_pair<int, int>(0, 1),
             py::arg("shortest") = false,
             py::arg("fill") = "none")
        .def("setInputs", &Xstack::setInputs)
        .def("getInputs", &Xstack::getInputs)
        .def("setLayout", &Xstack::setLayout)
        .def("getLayout", &Xstack::getLayout)
        .def("setGrid", &Xstack::setGrid)
        .def("getGrid", &Xstack::getGrid)
        .def("setShortest", &Xstack::setShortest)
        .def("getShortest", &Xstack::getShortest)
        .def("setFill", &Xstack::setFill)
        .def("getFill", &Xstack::getFill)
        ;
}