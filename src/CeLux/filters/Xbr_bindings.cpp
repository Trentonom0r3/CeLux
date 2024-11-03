#include "Xbr_bindings.hpp"

namespace py = pybind11;

void bind_Xbr(py::module_ &m) {
    py::class_<Xbr, FilterBase, std::shared_ptr<Xbr>>(m, "Xbr")
        .def(py::init<int>(),
             py::arg("scaleFactor") = 3)
        .def("setScaleFactor", &Xbr::setScaleFactor)
        .def("getScaleFactor", &Xbr::getScaleFactor)
        ;
}