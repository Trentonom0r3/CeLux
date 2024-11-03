#include "Fftfilt_bindings.hpp"

namespace py = pybind11;

void bind_Fftfilt(py::module_ &m) {
    py::class_<Fftfilt, FilterBase, std::shared_ptr<Fftfilt>>(m, "Fftfilt")
        .def(py::init<int, int, int, std::string, std::string, std::string, int>(),
             py::arg("dc_Y") = 0,
             py::arg("dc_U") = 0,
             py::arg("dc_V") = 0,
             py::arg("weight_Y") = "1",
             py::arg("weight_U") = "",
             py::arg("weight_V") = "",
             py::arg("eval") = 0)
        .def("setDc_Y", &Fftfilt::setDc_Y)
        .def("getDc_Y", &Fftfilt::getDc_Y)
        .def("setDc_U", &Fftfilt::setDc_U)
        .def("getDc_U", &Fftfilt::getDc_U)
        .def("setDc_V", &Fftfilt::setDc_V)
        .def("getDc_V", &Fftfilt::getDc_V)
        .def("setWeight_Y", &Fftfilt::setWeight_Y)
        .def("getWeight_Y", &Fftfilt::getWeight_Y)
        .def("setWeight_U", &Fftfilt::setWeight_U)
        .def("getWeight_U", &Fftfilt::getWeight_U)
        .def("setWeight_V", &Fftfilt::setWeight_V)
        .def("getWeight_V", &Fftfilt::getWeight_V)
        .def("setEval", &Fftfilt::setEval)
        .def("getEval", &Fftfilt::getEval)
        ;
}