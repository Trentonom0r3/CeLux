#include "Shear_bindings.hpp"

namespace py = pybind11;

void bind_Shear(py::module_ &m) {
    py::class_<Shear, FilterBase, std::shared_ptr<Shear>>(m, "Shear")
        .def(py::init<float, float, std::string, int>(),
             py::arg("shx") = 0.00,
             py::arg("shy") = 0.00,
             py::arg("fillcolor") = "black",
             py::arg("interp") = 1)
        .def("setShx", &Shear::setShx)
        .def("getShx", &Shear::getShx)
        .def("setShy", &Shear::setShy)
        .def("getShy", &Shear::getShy)
        .def("setFillcolor", &Shear::setFillcolor)
        .def("getFillcolor", &Shear::getFillcolor)
        .def("setInterp", &Shear::setInterp)
        .def("getInterp", &Shear::getInterp)
        ;
}