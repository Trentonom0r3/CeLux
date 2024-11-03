#include "Mix_bindings.hpp"

namespace py = pybind11;

void bind_Mix(py::module_ &m) {
    py::class_<Mix, FilterBase, std::shared_ptr<Mix>>(m, "Mix")
        .def(py::init<int, std::string, float, int, int>(),
             py::arg("inputs") = 2,
             py::arg("weights") = "1 1",
             py::arg("scale") = 0.00,
             py::arg("planes") = 15,
             py::arg("duration") = 0)
        .def("setInputs", &Mix::setInputs)
        .def("getInputs", &Mix::getInputs)
        .def("setWeights", &Mix::setWeights)
        .def("getWeights", &Mix::getWeights)
        .def("setScale", &Mix::setScale)
        .def("getScale", &Mix::getScale)
        .def("setPlanes", &Mix::setPlanes)
        .def("getPlanes", &Mix::getPlanes)
        .def("setDuration", &Mix::setDuration)
        .def("getDuration", &Mix::getDuration)
        ;
}