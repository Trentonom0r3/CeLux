#include "Dctdnoiz_bindings.hpp"

namespace py = pybind11;

void bind_Dctdnoiz(py::module_ &m) {
    py::class_<Dctdnoiz, FilterBase, std::shared_ptr<Dctdnoiz>>(m, "Dctdnoiz")
        .def(py::init<float, int, std::string, int>(),
             py::arg("sigma") = 0.00,
             py::arg("overlap") = -1,
             py::arg("expr") = "",
             py::arg("blockSizeExpressedInBits") = 3)
        .def("setSigma", &Dctdnoiz::setSigma)
        .def("getSigma", &Dctdnoiz::getSigma)
        .def("setOverlap", &Dctdnoiz::setOverlap)
        .def("getOverlap", &Dctdnoiz::getOverlap)
        .def("setExpr", &Dctdnoiz::setExpr)
        .def("getExpr", &Dctdnoiz::getExpr)
        .def("setBlockSizeExpressedInBits", &Dctdnoiz::setBlockSizeExpressedInBits)
        .def("getBlockSizeExpressedInBits", &Dctdnoiz::getBlockSizeExpressedInBits)
        ;
}