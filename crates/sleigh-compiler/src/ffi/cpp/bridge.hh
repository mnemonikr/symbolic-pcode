#pragma once

#include <memory>

#include "error_handling.hh"
#include "slgh_compile.hh"

class DefineOptions {
    std::map<std::string, std::string> opts;

    public:
        void defineOption(std::string &name, std::string &value) {
            this->opts[std::move(name)] = std::move(value);
        }
        const std::map<std::string, std::string>& options() const { return opts; }
};

std::unique_ptr<DefineOptions> construct_new_define_options();

class SleighCompileProxy : public SleighCompile {
    public:
        void setAllOptionsProxy(std::unique_ptr<DefineOptions> defines, bool unnecessaryPcodeWarning,
                bool lenientConflict, bool allCollisionWarning,
                bool allNopWarning, bool deadTempWarning, bool enforceLocalKeyWord,
                bool largeTemporaryWarning, bool caseSensitiveRegisterNames);
        int4 run_compilation_proxy(const std::string &filein, const std::string &fileout);
};

std::unique_ptr<SleighCompileProxy> construct_new_sleigh_compile();
