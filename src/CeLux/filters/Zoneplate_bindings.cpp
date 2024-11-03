#include "Zoneplate_bindings.hpp"

namespace py = pybind11;

void bind_Zoneplate(py::module_ &m) {
    py::class_<Zoneplate, FilterBase, std::shared_ptr<Zoneplate>>(m, "Zoneplate")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, int64_t, std::pair<int, int>, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int>(),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("duration") = 0,
             py::arg("sar") = std::make_pair<int, int>(0, 1),
             py::arg("precision") = 10,
             py::arg("xo") = 0,
             py::arg("yo") = 0,
             py::arg("to") = 0,
             py::arg("k0") = 0,
             py::arg("kx") = 0,
             py::arg("ky") = 0,
             py::arg("kt") = 0,
             py::arg("kxt") = 0,
             py::arg("kyt") = 0,
             py::arg("kxy") = 0,
             py::arg("kx2") = 0,
             py::arg("ky2") = 0,
             py::arg("kt2") = 0,
             py::arg("ku") = 0,
             py::arg("kv") = 0)
        .def("setSize", &Zoneplate::setSize)
        .def("getSize", &Zoneplate::getSize)
        .def("setRate", &Zoneplate::setRate)
        .def("getRate", &Zoneplate::getRate)
        .def("setDuration", &Zoneplate::setDuration)
        .def("getDuration", &Zoneplate::getDuration)
        .def("setSar", &Zoneplate::setSar)
        .def("getSar", &Zoneplate::getSar)
        .def("setPrecision", &Zoneplate::setPrecision)
        .def("getPrecision", &Zoneplate::getPrecision)
        .def("setXo", &Zoneplate::setXo)
        .def("getXo", &Zoneplate::getXo)
        .def("setYo", &Zoneplate::setYo)
        .def("getYo", &Zoneplate::getYo)
        .def("setTo", &Zoneplate::setTo)
        .def("getTo", &Zoneplate::getTo)
        .def("setK0", &Zoneplate::setK0)
        .def("getK0", &Zoneplate::getK0)
        .def("setKx", &Zoneplate::setKx)
        .def("getKx", &Zoneplate::getKx)
        .def("setKy", &Zoneplate::setKy)
        .def("getKy", &Zoneplate::getKy)
        .def("setKt", &Zoneplate::setKt)
        .def("getKt", &Zoneplate::getKt)
        .def("setKxt", &Zoneplate::setKxt)
        .def("getKxt", &Zoneplate::getKxt)
        .def("setKyt", &Zoneplate::setKyt)
        .def("getKyt", &Zoneplate::getKyt)
        .def("setKxy", &Zoneplate::setKxy)
        .def("getKxy", &Zoneplate::getKxy)
        .def("setKx2", &Zoneplate::setKx2)
        .def("getKx2", &Zoneplate::getKx2)
        .def("setKy2", &Zoneplate::setKy2)
        .def("getKy2", &Zoneplate::getKy2)
        .def("setKt2", &Zoneplate::setKt2)
        .def("getKt2", &Zoneplate::getKt2)
        .def("setKu", &Zoneplate::setKu)
        .def("getKu", &Zoneplate::getKu)
        .def("setKv", &Zoneplate::setKv)
        .def("getKv", &Zoneplate::getKv)
        ;
}