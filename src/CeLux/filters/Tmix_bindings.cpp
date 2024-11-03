#include "Tmix_bindings.hpp"

namespace py = pybind11;

void bind_Tmix(py::module_ &m) {
    py::class_<Tmix, FilterBase, std::shared_ptr<Tmix>>(m, "Tmix")
        .def(py::init<int, std::string, float, int>(),
             py::arg("frames") = 3,
             py::arg("weights") = "1 1 1",
             py::arg("scale") = 0.00,
             py::arg("planes") = 15)
        .def("setFrames", &Tmix::setFrames)
        .def("getFrames", &Tmix::getFrames)
        .def("setWeights", &Tmix::setWeights)
        .def("getWeights", &Tmix::getWeights)
        .def("setScale", &Tmix::setScale)
        .def("getScale", &Tmix::getScale)
        .def("setPlanes", &Tmix::setPlanes)
        .def("getPlanes", &Tmix::getPlanes)
        ;
}