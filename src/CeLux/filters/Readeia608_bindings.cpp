#include "Readeia608_bindings.hpp"

namespace py = pybind11;

void bind_Readeia608(py::module_ &m) {
    py::class_<Readeia608, FilterBase, std::shared_ptr<Readeia608>>(m, "Readeia608")
        .def(py::init<int, int, float, bool, bool>(),
             py::arg("scan_min") = 0,
             py::arg("scan_max") = 29,
             py::arg("spw") = 0.27,
             py::arg("chp") = false,
             py::arg("lp") = true)
        .def("setScan_min", &Readeia608::setScan_min)
        .def("getScan_min", &Readeia608::getScan_min)
        .def("setScan_max", &Readeia608::setScan_max)
        .def("getScan_max", &Readeia608::getScan_max)
        .def("setSpw", &Readeia608::setSpw)
        .def("getSpw", &Readeia608::getSpw)
        .def("setChp", &Readeia608::setChp)
        .def("getChp", &Readeia608::getChp)
        .def("setLp", &Readeia608::setLp)
        .def("getLp", &Readeia608::getLp)
        ;
}