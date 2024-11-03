#include "Sr_bindings.hpp"

namespace py = pybind11;

void bind_Sr(py::module_ &m) {
    py::class_<Sr, FilterBase, std::shared_ptr<Sr>>(m, "Sr")
        .def(py::init<int, int, std::string, std::string, std::string>(),
             py::arg("dnn_backend") = 1,
             py::arg("scale_factor") = 2,
             py::arg("model") = "",
             py::arg("input") = "x",
             py::arg("output") = "y")
        .def("setDnn_backend", &Sr::setDnn_backend)
        .def("getDnn_backend", &Sr::getDnn_backend)
        .def("setScale_factor", &Sr::setScale_factor)
        .def("getScale_factor", &Sr::getScale_factor)
        .def("setModel", &Sr::setModel)
        .def("getModel", &Sr::getModel)
        .def("setInput", &Sr::setInput)
        .def("getInput", &Sr::getInput)
        .def("setOutput", &Sr::setOutput)
        .def("getOutput", &Sr::getOutput)
        ;
}