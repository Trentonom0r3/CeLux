#include "Showfreqs_bindings.hpp"

namespace py = pybind11;

void bind_Showfreqs(py::module_ &m) {
    py::class_<Showfreqs, FilterBase, std::shared_ptr<Showfreqs>>(m, "Showfreqs")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int, int, int, int, int, float, int, std::string, int, float, int, std::string>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("mode") = 1,
             py::arg("ascale") = 3,
             py::arg("fscale") = 0,
             py::arg("win_size") = 2048,
             py::arg("win_func") = 1,
             py::arg("overlap") = 1.00,
             py::arg("averaging") = 1,
             py::arg("colors") = "red|green|blue|yellow|orange|lime|pink|magenta|brown",
             py::arg("cmode") = 0,
             py::arg("minamp") = 0.00,
             py::arg("data") = 0,
             py::arg("channels") = "all")
        .def("setSize", &Showfreqs::setSize)
        .def("getSize", &Showfreqs::getSize)
        .def("setRate", &Showfreqs::setRate)
        .def("getRate", &Showfreqs::getRate)
        .def("setMode", &Showfreqs::setMode)
        .def("getMode", &Showfreqs::getMode)
        .def("setAscale", &Showfreqs::setAscale)
        .def("getAscale", &Showfreqs::getAscale)
        .def("setFscale", &Showfreqs::setFscale)
        .def("getFscale", &Showfreqs::getFscale)
        .def("setWin_size", &Showfreqs::setWin_size)
        .def("getWin_size", &Showfreqs::getWin_size)
        .def("setWin_func", &Showfreqs::setWin_func)
        .def("getWin_func", &Showfreqs::getWin_func)
        .def("setOverlap", &Showfreqs::setOverlap)
        .def("getOverlap", &Showfreqs::getOverlap)
        .def("setAveraging", &Showfreqs::setAveraging)
        .def("getAveraging", &Showfreqs::getAveraging)
        .def("setColors", &Showfreqs::setColors)
        .def("getColors", &Showfreqs::getColors)
        .def("setCmode", &Showfreqs::setCmode)
        .def("getCmode", &Showfreqs::getCmode)
        .def("setMinamp", &Showfreqs::setMinamp)
        .def("getMinamp", &Showfreqs::getMinamp)
        .def("setData", &Showfreqs::setData)
        .def("getData", &Showfreqs::getData)
        .def("setChannels", &Showfreqs::setChannels)
        .def("getChannels", &Showfreqs::getChannels)
        ;
}