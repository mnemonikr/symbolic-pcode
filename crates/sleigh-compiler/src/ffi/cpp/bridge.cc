#include <mutex>

#include "bridge.hh"
#include "rust/cxx.h"
#include "sleigh-compiler/src/ffi/sys.rs.h"

void SleighCompileProxy::setAllOptionsProxy(
        const std::vector<PreprocessorDefine> &defines, bool unnecessaryPcodeWarning,
        bool lenientConflict, bool allCollisionWarning,
        bool allNopWarning,bool deadTempWarning,bool enforceLocalKeyWord,
        bool largeTemporaryWarning, bool caseSensitiveRegisterNames) {

    std::map<std::string, std::string> definesMap;
    for (auto &d : defines) {
        definesMap.emplace(d.name, d.value);
    }

    setAllOptions(definesMap, unnecessaryPcodeWarning,
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

std::unique_ptr<SleighCompileProxy> construct_new_sleigh_compile() {
    return std::make_unique<SleighCompileProxy>();
}
