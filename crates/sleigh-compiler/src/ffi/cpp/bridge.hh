#pragma once

#include <memory>

#include "error_handling.hh"
#include "slgh_compile.hh"

// Defined in cxx bridge in sys.rs
struct PreprocessorDefine;

class SleighCompileProxy : public SleighCompile {
    public:
        void setAllOptionsProxy(const std::vector<PreprocessorDefine> &defines, bool unnecessaryPcodeWarning,
                bool lenientConflict, bool allCollisionWarning,
                bool allNopWarning, bool deadTempWarning, bool enforceLocalKeyWord,
                bool largeTemporaryWarning, bool caseSensitiveRegisterNames);
        int4 run_compilation_proxy(const std::string &filein, const std::string &fileout);
};

std::unique_ptr<SleighCompileProxy> construct_new_sleigh_compile();
