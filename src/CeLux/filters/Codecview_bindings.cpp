#include "Codecview_bindings.hpp"

namespace py = pybind11;

void bind_Codecview(py::module_ &m) {
    py::class_<Codecview, FilterBase, std::shared_ptr<Codecview>>(m, "Codecview")
        .def(py::init<int, bool, int, int, bool>(),
             py::arg("mv") = 0,
             py::arg("qp") = false,
             py::arg("mv_type") = 0,
             py::arg("frame_type") = 0,
             py::arg("block") = false)
        .def("setMv", &Codecview::setMv)
        .def("getMv", &Codecview::getMv)
        .def("setQp", &Codecview::setQp)
        .def("getQp", &Codecview::getQp)
        .def("setMv_type", &Codecview::setMv_type)
        .def("getMv_type", &Codecview::getMv_type)
        .def("setFrame_type", &Codecview::setFrame_type)
        .def("getFrame_type", &Codecview::getFrame_type)
        .def("setBlock", &Codecview::setBlock)
        .def("getBlock", &Codecview::getBlock)
        ;
}