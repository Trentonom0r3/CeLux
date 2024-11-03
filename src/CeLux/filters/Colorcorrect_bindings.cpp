#include "Colorcorrect_bindings.hpp"

namespace py = pybind11;

void bind_Colorcorrect(py::module_ &m) {
    py::class_<Colorcorrect, FilterBase, std::shared_ptr<Colorcorrect>>(m, "Colorcorrect")
        .def(py::init<float, float, float, float, float, int>(),
             py::arg("rl") = 0.00,
             py::arg("bl") = 0.00,
             py::arg("rh") = 0.00,
             py::arg("bh") = 0.00,
             py::arg("saturation") = 1.00,
             py::arg("analyze") = 0)
        .def("setRl", &Colorcorrect::setRl)
        .def("getRl", &Colorcorrect::getRl)
        .def("setBl", &Colorcorrect::setBl)
        .def("getBl", &Colorcorrect::getBl)
        .def("setRh", &Colorcorrect::setRh)
        .def("getRh", &Colorcorrect::getRh)
        .def("setBh", &Colorcorrect::setBh)
        .def("getBh", &Colorcorrect::getBh)
        .def("setSaturation", &Colorcorrect::setSaturation)
        .def("getSaturation", &Colorcorrect::getSaturation)
        .def("setAnalyze", &Colorcorrect::setAnalyze)
        .def("getAnalyze", &Colorcorrect::getAnalyze)
        ;
}