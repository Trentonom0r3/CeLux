#include "Gblur_bindings.hpp"

namespace py = pybind11;

void bind_Gblur(py::module_ &m) {
    py::class_<Gblur, FilterBase, std::shared_ptr<Gblur>>(m, "Gblur")
        .def(py::init<float, int, int, float>(),
             py::arg("sigma") = 0.50,
             py::arg("steps") = 1,
             py::arg("planes") = 15,
             py::arg("sigmaV") = -1.00)
        .def("setSigma", &Gblur::setSigma)
        .def("getSigma", &Gblur::getSigma)
        .def("setSteps", &Gblur::setSteps)
        .def("getSteps", &Gblur::getSteps)
        .def("setPlanes", &Gblur::setPlanes)
        .def("getPlanes", &Gblur::getPlanes)
        .def("setSigmaV", &Gblur::setSigmaV)
        .def("getSigmaV", &Gblur::getSigmaV)
        ;
}