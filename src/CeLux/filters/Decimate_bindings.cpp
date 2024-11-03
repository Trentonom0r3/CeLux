#include "Decimate_bindings.hpp"

namespace py = pybind11;

void bind_Decimate(py::module_ &m) {
    py::class_<Decimate, FilterBase, std::shared_ptr<Decimate>>(m, "Decimate")
        .def(py::init<int, double, double, int, int, bool, bool, bool>(),
             py::arg("cycle") = 5,
             py::arg("dupthresh") = 1.10,
             py::arg("scthresh") = 15.00,
             py::arg("blockx") = 32,
             py::arg("blocky") = 32,
             py::arg("ppsrc") = false,
             py::arg("chroma") = true,
             py::arg("mixed") = false)
        .def("setCycle", &Decimate::setCycle)
        .def("getCycle", &Decimate::getCycle)
        .def("setDupthresh", &Decimate::setDupthresh)
        .def("getDupthresh", &Decimate::getDupthresh)
        .def("setScthresh", &Decimate::setScthresh)
        .def("getScthresh", &Decimate::getScthresh)
        .def("setBlockx", &Decimate::setBlockx)
        .def("getBlockx", &Decimate::getBlockx)
        .def("setBlocky", &Decimate::setBlocky)
        .def("getBlocky", &Decimate::getBlocky)
        .def("setPpsrc", &Decimate::setPpsrc)
        .def("getPpsrc", &Decimate::getPpsrc)
        .def("setChroma", &Decimate::setChroma)
        .def("getChroma", &Decimate::getChroma)
        .def("setMixed", &Decimate::setMixed)
        .def("getMixed", &Decimate::getMixed)
        ;
}