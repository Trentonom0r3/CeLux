#include "Dnn_detect_bindings.hpp"

namespace py = pybind11;

void bind_Dnn_detect(py::module_ &m) {
    py::class_<Dnn_detect, FilterBase, std::shared_ptr<Dnn_detect>>(m, "Dnn_detect")
        .def(py::init<int, std::string, std::string, std::string, std::string, bool, float, std::string, int, int, int, int, std::string>(),
             py::arg("dnn_backend") = 2,
             py::arg("model") = "",
             py::arg("input") = "",
             py::arg("output") = "",
             py::arg("backend_configs") = "",
             py::arg("async") = true,
             py::arg("confidence") = 0.50,
             py::arg("labels") = "",
             py::arg("model_type") = 0,
             py::arg("cell_w") = 0,
             py::arg("cell_h") = 0,
             py::arg("nb_classes") = 0,
             py::arg("anchors") = "")
        .def("setDnn_backend", &Dnn_detect::setDnn_backend)
        .def("getDnn_backend", &Dnn_detect::getDnn_backend)
        .def("setModel", &Dnn_detect::setModel)
        .def("getModel", &Dnn_detect::getModel)
        .def("setInput", &Dnn_detect::setInput)
        .def("getInput", &Dnn_detect::getInput)
        .def("setOutput", &Dnn_detect::setOutput)
        .def("getOutput", &Dnn_detect::getOutput)
        .def("setBackend_configs", &Dnn_detect::setBackend_configs)
        .def("getBackend_configs", &Dnn_detect::getBackend_configs)
        .def("setAsync", &Dnn_detect::setAsync)
        .def("getAsync", &Dnn_detect::getAsync)
        .def("setConfidence", &Dnn_detect::setConfidence)
        .def("getConfidence", &Dnn_detect::getConfidence)
        .def("setLabels", &Dnn_detect::setLabels)
        .def("getLabels", &Dnn_detect::getLabels)
        .def("setModel_type", &Dnn_detect::setModel_type)
        .def("getModel_type", &Dnn_detect::getModel_type)
        .def("setCell_w", &Dnn_detect::setCell_w)
        .def("getCell_w", &Dnn_detect::getCell_w)
        .def("setCell_h", &Dnn_detect::setCell_h)
        .def("getCell_h", &Dnn_detect::getCell_h)
        .def("setNb_classes", &Dnn_detect::setNb_classes)
        .def("getNb_classes", &Dnn_detect::getNb_classes)
        .def("setAnchors", &Dnn_detect::setAnchors)
        .def("getAnchors", &Dnn_detect::getAnchors)
        ;
}