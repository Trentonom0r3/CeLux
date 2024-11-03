#include "Bm3d_bindings.hpp"

namespace py = pybind11;

void bind_Bm3d(py::module_ &m) {
    py::class_<Bm3d, FilterBase, std::shared_ptr<Bm3d>>(m, "Bm3d")
        .def(py::init<float, int, int, int, int, int, float, float, int, bool, int>(),
             py::arg("sigma") = 1.00,
             py::arg("block") = 16,
             py::arg("bstep") = 4,
             py::arg("group") = 1,
             py::arg("range") = 9,
             py::arg("mstep") = 1,
             py::arg("thmse") = 0.00,
             py::arg("hdthr") = 2.70,
             py::arg("estim") = 0,
             py::arg("ref") = false,
             py::arg("planes") = 7)
        .def("setSigma", &Bm3d::setSigma)
        .def("getSigma", &Bm3d::getSigma)
        .def("setBlock", &Bm3d::setBlock)
        .def("getBlock", &Bm3d::getBlock)
        .def("setBstep", &Bm3d::setBstep)
        .def("getBstep", &Bm3d::getBstep)
        .def("setGroup", &Bm3d::setGroup)
        .def("getGroup", &Bm3d::getGroup)
        .def("setRange", &Bm3d::setRange)
        .def("getRange", &Bm3d::getRange)
        .def("setMstep", &Bm3d::setMstep)
        .def("getMstep", &Bm3d::getMstep)
        .def("setThmse", &Bm3d::setThmse)
        .def("getThmse", &Bm3d::getThmse)
        .def("setHdthr", &Bm3d::setHdthr)
        .def("getHdthr", &Bm3d::getHdthr)
        .def("setEstim", &Bm3d::setEstim)
        .def("getEstim", &Bm3d::getEstim)
        .def("setRef", &Bm3d::setRef)
        .def("getRef", &Bm3d::getRef)
        .def("setPlanes", &Bm3d::setPlanes)
        .def("getPlanes", &Bm3d::getPlanes)
        ;
}