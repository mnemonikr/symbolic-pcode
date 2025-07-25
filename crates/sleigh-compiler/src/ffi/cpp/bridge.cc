#include <mutex>

#include "bridge.hh"
#include "rust/cxx.h"
#include "sleigh-compiler/src/ffi/ffi.rs.h"

void SleighCompileProxy::setAllOptionsProxy(
        std::unique_ptr<DefineOptions> defines, bool unnecessaryPcodeWarning,
        bool lenientConflict, bool allCollisionWarning,
        bool allNopWarning,bool deadTempWarning,bool enforceLocalKeyWord,
        bool largeTemporaryWarning, bool caseSensitiveRegisterNames) {
    setAllOptions(defines->options(), unnecessaryPcodeWarning,
        lenientConflict, allCollisionWarning,
        allNopWarning, deadTempWarning, enforceLocalKeyWord,
        largeTemporaryWarning, caseSensitiveRegisterNames);
}

int4 SleighCompileProxy::run_compilation_proxy(const std::string &filein, const std::string &fileout) {
    // The run_compilation call sets the global variable `slgh` in slgh_compile.cc. Guard with lock
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock(mutex);
    return run_compilation(filein, fileout);
}

std::unique_ptr<DefineOptions> construct_new_define_options() {
    return std::make_unique<DefineOptions>();
}

std::unique_ptr<SleighCompileProxy> construct_new_sleigh_compile() {
    return std::make_unique<SleighCompileProxy>();
}
