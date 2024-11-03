#include "Fieldmatch_bindings.hpp"

namespace py = pybind11;

void bind_Fieldmatch(py::module_ &m) {
    py::class_<Fieldmatch, FilterBase, std::shared_ptr<Fieldmatch>>(m, "Fieldmatch")
        .def(py::init<int, int, bool, int, bool, int, double, int, int, int, bool, int, int, int>(),
             py::arg("order") = -1,
             py::arg("mode") = 1,
             py::arg("ppsrc") = false,
             py::arg("field") = -1,
             py::arg("mchroma") = true,
             py::arg("y1") = 0,
             py::arg("scthresh") = 12.00,
             py::arg("combmatch") = 1,
             py::arg("combdbg") = 0,
             py::arg("cthresh") = 9,
             py::arg("chroma") = false,
             py::arg("blockx") = 16,
             py::arg("blocky") = 16,
             py::arg("combpel") = 80)
        .def("setOrder", &Fieldmatch::setOrder)
        .def("getOrder", &Fieldmatch::getOrder)
        .def("setMode", &Fieldmatch::setMode)
        .def("getMode", &Fieldmatch::getMode)
        .def("setPpsrc", &Fieldmatch::setPpsrc)
        .def("getPpsrc", &Fieldmatch::getPpsrc)
        .def("setField", &Fieldmatch::setField)
        .def("getField", &Fieldmatch::getField)
        .def("setMchroma", &Fieldmatch::setMchroma)
        .def("getMchroma", &Fieldmatch::getMchroma)
        .def("setY1", &Fieldmatch::setY1)
        .def("getY1", &Fieldmatch::getY1)
        .def("setScthresh", &Fieldmatch::setScthresh)
        .def("getScthresh", &Fieldmatch::getScthresh)
        .def("setCombmatch", &Fieldmatch::setCombmatch)
        .def("getCombmatch", &Fieldmatch::getCombmatch)
        .def("setCombdbg", &Fieldmatch::setCombdbg)
        .def("getCombdbg", &Fieldmatch::getCombdbg)
        .def("setCthresh", &Fieldmatch::setCthresh)
        .def("getCthresh", &Fieldmatch::getCthresh)
        .def("setChroma", &Fieldmatch::setChroma)
        .def("getChroma", &Fieldmatch::getChroma)
        .def("setBlockx", &Fieldmatch::setBlockx)
        .def("getBlockx", &Fieldmatch::getBlockx)
        .def("setBlocky", &Fieldmatch::setBlocky)
        .def("getBlocky", &Fieldmatch::getBlocky)
        .def("setCombpel", &Fieldmatch::setCombpel)
        .def("getCombpel", &Fieldmatch::getCombpel)
        ;
}