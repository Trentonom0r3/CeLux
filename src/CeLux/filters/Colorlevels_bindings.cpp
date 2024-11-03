#include "Colorlevels_bindings.hpp"

namespace py = pybind11;

void bind_Colorlevels(py::module_ &m) {
    py::class_<Colorlevels, FilterBase, std::shared_ptr<Colorlevels>>(m, "Colorlevels")
        .def(py::init<double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, int>(),
             py::arg("rimin") = 0.00,
             py::arg("gimin") = 0.00,
             py::arg("bimin") = 0.00,
             py::arg("aimin") = 0.00,
             py::arg("rimax") = 1.00,
             py::arg("gimax") = 1.00,
             py::arg("bimax") = 1.00,
             py::arg("aimax") = 1.00,
             py::arg("romin") = 0.00,
             py::arg("gomin") = 0.00,
             py::arg("bomin") = 0.00,
             py::arg("aomin") = 0.00,
             py::arg("romax") = 1.00,
             py::arg("gomax") = 1.00,
             py::arg("bomax") = 1.00,
             py::arg("aomax") = 1.00,
             py::arg("preserve") = 0)
        .def("setRimin", &Colorlevels::setRimin)
        .def("getRimin", &Colorlevels::getRimin)
        .def("setGimin", &Colorlevels::setGimin)
        .def("getGimin", &Colorlevels::getGimin)
        .def("setBimin", &Colorlevels::setBimin)
        .def("getBimin", &Colorlevels::getBimin)
        .def("setAimin", &Colorlevels::setAimin)
        .def("getAimin", &Colorlevels::getAimin)
        .def("setRimax", &Colorlevels::setRimax)
        .def("getRimax", &Colorlevels::getRimax)
        .def("setGimax", &Colorlevels::setGimax)
        .def("getGimax", &Colorlevels::getGimax)
        .def("setBimax", &Colorlevels::setBimax)
        .def("getBimax", &Colorlevels::getBimax)
        .def("setAimax", &Colorlevels::setAimax)
        .def("getAimax", &Colorlevels::getAimax)
        .def("setRomin", &Colorlevels::setRomin)
        .def("getRomin", &Colorlevels::getRomin)
        .def("setGomin", &Colorlevels::setGomin)
        .def("getGomin", &Colorlevels::getGomin)
        .def("setBomin", &Colorlevels::setBomin)
        .def("getBomin", &Colorlevels::getBomin)
        .def("setAomin", &Colorlevels::setAomin)
        .def("getAomin", &Colorlevels::getAomin)
        .def("setRomax", &Colorlevels::setRomax)
        .def("getRomax", &Colorlevels::getRomax)
        .def("setGomax", &Colorlevels::setGomax)
        .def("getGomax", &Colorlevels::getGomax)
        .def("setBomax", &Colorlevels::setBomax)
        .def("getBomax", &Colorlevels::getBomax)
        .def("setAomax", &Colorlevels::setAomax)
        .def("getAomax", &Colorlevels::getAomax)
        .def("setPreserve", &Colorlevels::setPreserve)
        .def("getPreserve", &Colorlevels::getPreserve)
        ;
}