#pragma once

using namespace std;

#include <memory>

#include "error_handling.hh"

#include "sleigh.hh"
#include "loadimage.hh"
#include "translate.hh"
#include "marshal.hh"

using namespace ghidra;

class RustAssemblyEmit;

class RustAssemblyEmitProxy : public AssemblyEmit {
    RustAssemblyEmit &inner;

    public:
        RustAssemblyEmitProxy(RustAssemblyEmit &emit);
        void dump(const Address &addr, const std::string &mnemonic, const std::string &body) override;
};

class RustLoadImage;

class RustLoadImageProxy : public LoadImage {
    const RustLoadImage *inner;

    public:
        RustLoadImageProxy();
        void loadFill(uint1 *ptr,int4 size,const Address &addr) override;
        string getArchType() const override;
        void adjustVma(long adjust) override;
        void setInner(const RustLoadImage &inner) { this->inner = &inner; }
        void resetInner() { this->inner = nullptr; }
};

/// This class is responsible for managing the load image callback. When the disassembly function
/// is called the callback is set. On completion of the disassembly the callback is cleared.
class RustLoadImageManager final {
    RustLoadImageProxy &proxy;

    public:
        RustLoadImageManager(RustLoadImageProxy &proxy, const RustLoadImage &loadImage);
        ~RustLoadImageManager();
};

class RustPcodeEmit;

class RustPcodeEmitProxy : public PcodeEmit {
    RustPcodeEmit &inner;

    public:
        RustPcodeEmitProxy(RustPcodeEmit &emit);
        void dump(const Address &addr,OpCode opc,VarnodeData *outvar,VarnodeData *vars,int4 isize) override;
};

class RegisterVarnodeName {
    std::pair<VarnodeData, std::string> pair;

    public:
        RegisterVarnodeName(std::pair<VarnodeData, std::string> pair);
        const VarnodeData& getVarnode() const;
        const std::string& getName() const;
};

class SleighProxy : public Sleigh {
    unique_ptr<RustLoadImageProxy> loader;
    unique_ptr<ContextDatabase> context;

    public:
        SleighProxy(unique_ptr<RustLoadImageProxy> loader, unique_ptr<ContextDatabase> context);
        void parseProcessorConfig(const DocumentStorage &store);
        int4 disassemblePcode(const RustLoadImage &loadImage, RustPcodeEmit &emit, const Address &baseaddr) const;
        int4 disassembleNative(const RustLoadImage &loadImage, RustAssemblyEmit &emit, const Address &baseaddr) const;
        std::unique_ptr<std::string> getRegisterNameProxy(AddrSpace *base, uintb off, int4 size) const;
        unique_ptr<vector<RegisterVarnodeName>> getAllRegistersProxy() const;
};

unique_ptr<SleighProxy> construct_new_sleigh(unique_ptr<ContextDatabase> context);
unique_ptr<ContextDatabase> construct_new_context();

template<typename T, typename... Args>
T construct(Args... args) {
    return T(args...);
}

template<typename T, typename... Args>
unique_ptr<T> construct_new(Args... args) {
    return make_unique<T>(args...);
}

void initialize_element_id();
void initialize_attribute_id();

unique_ptr<Address> getAddress(const VarnodeData &data);
uint4 getSize(const VarnodeData &data);
const Document& parseDocumentIntoStore(DocumentStorage &store, const std::string &data);
const Element& getDocumentRoot(const Document& document);

void parseDocumentAndRegisterRootElement(DocumentStorage &store, const std::string &data);
