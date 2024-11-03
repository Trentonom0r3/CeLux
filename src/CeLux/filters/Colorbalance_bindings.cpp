#include "Colorbalance_bindings.hpp"

namespace py = pybind11;

void bind_Colorbalance(py::module_ &m) {
    py::class_<Colorbalance, FilterBase, std::shared_ptr<Colorbalance>>(m, "Colorbalance")
        .def(py::init<float, float, float, float, float, float, float, float, float, bool>(),
             py::arg("rs") = 0.00,
             py::arg("gs") = 0.00,
             py::arg("bs") = 0.00,
             py::arg("rm") = 0.00,
             py::arg("gm") = 0.00,
             py::arg("bm") = 0.00,
             py::arg("rh") = 0.00,
             py::arg("gh") = 0.00,
             py::arg("bh") = 0.00,
             py::arg("pl") = false)
        .def("setRs", &Colorbalance::setRs)
        .def("getRs", &Colorbalance::getRs)
        .def("setGs", &Colorbalance::setGs)
        .def("getGs", &Colorbalance::getGs)
        .def("setBs", &Colorbalance::setBs)
        .def("getBs", &Colorbalance::getBs)
        .def("setRm", &Colorbalance::setRm)
        .def("getRm", &Colorbalance::getRm)
        .def("setGm", &Colorbalance::setGm)
        .def("getGm", &Colorbalance::getGm)
        .def("setBm", &Colorbalance::setBm)
        .def("getBm", &Colorbalance::getBm)
        .def("setRh", &Colorbalance::setRh)
        .def("getRh", &Colorbalance::getRh)
        .def("setGh", &Colorbalance::setGh)
        .def("getGh", &Colorbalance::getGh)
        .def("setBh", &Colorbalance::setBh)
        .def("getBh", &Colorbalance::getBh)
        .def("setPl", &Colorbalance::setPl)
        .def("getPl", &Colorbalance::getPl)
        ;
}