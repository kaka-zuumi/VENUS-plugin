#include <catch2/catch.hpp>
#include <venus_nwchemex_plugin/venus_nwchemex_plugin.hpp>

TEST_CASE("load_modules") {
    pluginplay::ModuleManager mm;
    venus_nwchemex_plugin::load_modules(mm);
}
