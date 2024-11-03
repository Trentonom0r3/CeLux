#include "A3dscope_bindings.hpp"

namespace py = pybind11;

void bind_A3dscope(py::module_ &m) {
    py::class_<A3dscope, FilterBase, std::shared_ptr<A3dscope>>(m, "A3dscope")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, float, float, float, float, float, float, int>(),
             py::arg("rate") = std::make_pair<int, int>(0, 1),
             py::arg("size") = std::make_pair<int, int>(0, 1),
             py::arg("fov") = 90.00,
             py::arg("roll") = 0.00,
             py::arg("pitch") = 0.00,
             py::arg("yaw") = 0.00,
             py::arg("zzoom") = 1.00,
             py::arg("zpos") = 0.00,
             py::arg("length") = 15)
        .def("setRate", &A3dscope::setRate)
        .def("getRate", &A3dscope::getRate)
        .def("setSize", &A3dscope::setSize)
        .def("getSize", &A3dscope::getSize)
        .def("setFov", &A3dscope::setFov)
        .def("getFov", &A3dscope::getFov)
        .def("setRoll", &A3dscope::setRoll)
        .def("getRoll", &A3dscope::getRoll)
        .def("setPitch", &A3dscope::setPitch)
        .def("getPitch", &A3dscope::getPitch)
        .def("setYaw", &A3dscope::setYaw)
        .def("getYaw", &A3dscope::getYaw)
        .def("setZzoom", &A3dscope::setZzoom)
        .def("getZzoom", &A3dscope::getZzoom)
        .def("setZpos", &A3dscope::setZpos)
        .def("getZpos", &A3dscope::getZpos)
        .def("setLength", &A3dscope::setLength)
        .def("getLength", &A3dscope::getLength)
        ;
}