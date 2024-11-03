#include "Dnn_classify_bindings.hpp"

namespace py = pybind11;

void bind_Dnn_classify(py::module_ &m) {
    py::class_<Dnn_classify, FilterBase, std::shared_ptr<Dnn_classify>>(m, "Dnn_classify")
        .def(py::init<int, std::string, std::string, std::string, std::string, bool, float, std::string, std::string>(),
             py::arg("dnn_backend") = 2,
             py::arg("model") = "",
             py::arg("input") = "",
             py::arg("output") = "",
             py::arg("backend_configs") = "",
             py::arg("async") = true,
             py::arg("confidence") = 0.50,
             py::arg("labels") = "",
             py::arg("target") = "")
        .def("setDnn_backend", &Dnn_classify::setDnn_backend)
        .def("getDnn_backend", &Dnn_classify::getDnn_backend)
        .def("setModel", &Dnn_classify::setModel)
        .def("getModel", &Dnn_classify::getModel)
        .def("setInput", &Dnn_classify::setInput)
        .def("getInput", &Dnn_classify::getInput)
        .def("setOutput", &Dnn_classify::setOutput)
        .def("getOutput", &Dnn_classify::getOutput)
        .def("setBackend_configs", &Dnn_classify::setBackend_configs)
        .def("getBackend_configs", &Dnn_classify::getBackend_configs)
        .def("setAsync", &Dnn_classify::setAsync)
        .def("getAsync", &Dnn_classify::getAsync)
        .def("setConfidence", &Dnn_classify::setConfidence)
        .def("getConfidence", &Dnn_classify::getConfidence)
        .def("setLabels", &Dnn_classify::setLabels)
        .def("getLabels", &Dnn_classify::getLabels)
        .def("setTarget", &Dnn_classify::setTarget)
        .def("getTarget", &Dnn_classify::getTarget)
        ;
}