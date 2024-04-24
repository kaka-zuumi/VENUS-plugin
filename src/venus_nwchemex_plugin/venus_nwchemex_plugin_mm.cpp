#include "venus_nwchemex_plugin/venus_nwchemex_plugin.hpp"
//#include "venus_nwchemex_plugin/venus_nwchemex_plugin_mm.hpp"
#include "simde/types.hpp"
#include <pluginplay/property_type/property_type.hpp>

namespace venus_nwchemex_plugin {

inline void set_defaults(pluginplay::ModuleManager& mm) {
    // Default submodules between collections can be set here
}

DECLARE_PLUGIN(venus_nwchemex_plugin) {
    // Add subcollection load calls here

    // Assign default submodules
    set_defaults(mm);
}

// Kazuumi addition:
PYBIND11_MODULE(venus_nwchemex_plugin, m) {
    m.doc() =
      "PySimDE: Python bindings for the Simulation development environment";
    export_sampling(m);
}

} // namespace venus_nwchemex_plugin
