#include "filter_bindings_chunk_6.hpp"
// Include filter binding headers for this chunk
#include "Showwaves_bindings.hpp"
#include "Showwavespic_bindings.hpp"
#include "Buffer_bindings.hpp"
#include "Buffersink_bindings.hpp"

namespace py = pybind11;

void register_filter_bindings_chunk_6(py::module_ &m) {
    bind_Showwaves(m);
    bind_Showwavespic(m);
    bind_Buffer(m);
    bind_Buffersink(m);
}
