#include "Dnn_processing_bindings.hpp"

namespace py = pybind11;

void bind_Dnn_processing(py::module_ &m) {
    py::class_<Dnn_processing, FilterBase, std::shared_ptr<Dnn_processing>>(m, "Dnn_processing")
        .def(py::init<int, std::string, std::string, std::string, std::string, bool>(),
             py::arg("dnn_backend") = 1,
             py::arg("model") = "",
             py::arg("input") = "",
             py::arg("output") = "",
             py::arg("backend_configs") = "",
             py::arg("async") = true)
        .def("setDnn_backend", &Dnn_processing::setDnn_backend)
        .def("getDnn_backend", &Dnn_processing::getDnn_backend)
        .def("setModel", &Dnn_processing::setModel)
        .def("getModel", &Dnn_processing::getModel)
        .def("setInput", &Dnn_processing::setInput)
        .def("getInput", &Dnn_processing::getInput)
        .def("setOutput", &Dnn_processing::setOutput)
        .def("getOutput", &Dnn_processing::getOutput)
        .def("setBackend_configs", &Dnn_processing::setBackend_configs)
        .def("getBackend_configs", &Dnn_processing::getBackend_configs)
        .def("setAsync", &Dnn_processing::setAsync)
        .def("getAsync", &Dnn_processing::getAsync)
        ;
}