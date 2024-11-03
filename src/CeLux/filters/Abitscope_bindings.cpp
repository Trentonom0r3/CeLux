#include "Abitscope_bindings.hpp"

namespace py = pybind11;

void bind_Abitscope(py::module_ &m) {
    py::class_<Abitscope, FilterBase, std::shared_ptr<Abitscope>>(m, "Abitscope")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, std::string, int>(),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("colors") = "red|green|blue|yellow|orange|lime|pink|magenta|brown",
             py::arg("mode") = 0)
        .def("setRate", &Abitscope::setRate)
        .def("getRate", &Abitscope::getRate)
        .def("setSize", &Abitscope::setSize)
        .def("getSize", &Abitscope::getSize)
        .def("setColors", &Abitscope::setColors)
        .def("getColors", &Abitscope::getColors)
        .def("setMode", &Abitscope::setMode)
        .def("getMode", &Abitscope::getMode)
        ;
}