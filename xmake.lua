add_rules("mode.debug", "mode.release")

package("april_src")
    add_deps("cmake")
    set_sourcedir(path.join(os.scriptdir(), "apriltag"))
    on_install(function (package)
        local configs = {}
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"))
        table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"))
        import("package.tools.cmake").install(package, configs)
    end)
package_end()


add_requires("opencv")
add_requires("april_src")
add_requires("cmake::ZED 3",{alias = "zed", system = true})
add_requires("cmake::CUDA $(ZED_CUDA_VERSION)",{alias = "cuda", system = true})

target("calibrate")
    set_kind("binary")
    add_files("src/calibrate.cpp")
	add_packages("opencv")

target("apritag")
    set_kind("binary")
    add_files("src/main.cpp")
	add_packages("opencv")
	add_packages("april_src")

target("zed_test")
    set_kind("binary")
    add_files("src/zed_test.cpp")
    add_packages("opencv")
    add_packages("april_src")
    add_packages("zed")
    add_packages("cuda")

target("zed_apriltag")
    set_kind("binary")
    add_files("src/zed_apriltag.cpp")
    add_packages("april_src")
    add_packages("opencv")
    add_packages("zed")
    add_packages("cuda")

--
-- If you want to known more usage about xmake, please see https://xmake.io
--
-- ## FAQ
--
-- You can enter the project directory firstly before building project.
--
--   $ cd projectdir
--
-- 1. How to build project?
--
--   $ xmake
--
-- 2. How to configure project?
--
--   $ xmake f -p [macosx|linux|iphoneos ..] -a [x86_64|i386|arm64 ..] -m [debug|release]
--
-- 3. Where is the build output directory?
--
--   The default output directory is `./build` and you can configure the output directory.
--
--   $ xmake f -o outputdir
--   $ xmake
--
-- 4. How to run and debug target after building project?
--
--   $ xmake run [targetname]
--   $ xmake run -d [targetname]
--
-- 5. How to install target to the system directory or other output directory?
--
--   $ xmake install
--   $ xmake install -o installdir
--
-- 6. Add some frequently-used compilation flags in xmake.lua
--
-- @code
--    -- add debug and release modes
--    add_rules("mode.debug", "mode.release")
--
--    -- add macro defination
--    add_defines("NDEBUG", "_GNU_SOURCE=1")
--
--    -- set warning all as error
--    set_warnings("all", "error")
--
--    -- set language: c99, c++11
--    set_languages("c99", "c++11")
--
--    -- set optimization: none, faster, fastest, smallest
--    set_optimize("fastest")
--
--    -- add include search directories
--    add_includedirs("/usr/include", "/usr/local/include")
--
--    -- add link libraries and search directories
--    add_links("tbox")
--    add_linkdirs("/usr/local/lib", "/usr/lib")
--
--    -- add system link libraries
--    add_syslinks("z", "pthread")
--
--    -- add compilation and link flags
--    add_cxflags("-stdnolib", "-fno-strict-aliasing")
--    add_ldflags("-L/usr/local/lib", "-lpthread", {force = true})
--
-- @endcode
--

