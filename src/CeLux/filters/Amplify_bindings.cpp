#include "Amplify_bindings.hpp"

namespace py = pybind11;

void bind_Amplify(py::module_ &m) {
    py::class_<Amplify, FilterBase, std::shared_ptr<Amplify>>(m, "Amplify")
        .def(py::init<int, float, float, float, float, float, int>(),
             py::arg("radius") = 2,
             py::arg("factor") = 2.00,
             py::arg("threshold") = 10.00,
             py::arg("tolerance") = 0.00,
             py::arg("low") = 65535.00,
             py::arg("high") = 65535.00,
             py::arg("planes") = 7)
        .def("setRadius", &Amplify::setRadius)
        .def("getRadius", &Amplify::getRadius)
        .def("setFactor", &Amplify::setFactor)
        .def("getFactor", &Amplify::getFactor)
        .def("setThreshold", &Amplify::setThreshold)
        .def("getThreshold", &Amplify::getThreshold)
        .def("setTolerance", &Amplify::setTolerance)
        .def("getTolerance", &Amplify::getTolerance)
        .def("setLow", &Amplify::setLow)
        .def("getLow", &Amplify::getLow)
        .def("setHigh", &Amplify::setHigh)
        .def("getHigh", &Amplify::getHigh)
        .def("setPlanes", &Amplify::setPlanes)
        .def("getPlanes", &Amplify::getPlanes)
        ;
}