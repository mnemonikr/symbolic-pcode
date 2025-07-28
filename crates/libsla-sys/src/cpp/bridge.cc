#include <mutex>
#include <stdexcept>

#include "bridge.hh"
#include "rust/cxx.h"
#include "libsla-sys/src/sys.rs.h"

// Copied from architecture.cc
ElementId ELEM_PROCESSOR_SPEC = ElementId("processor_spec",147);

RustAssemblyEmitProxy::RustAssemblyEmitProxy(RustAssemblyEmit &emit)
    : inner(emit) {
}

void RustAssemblyEmitProxy::dump(
    const Address &addr,
    const std::string &mnemonic,
    const std::string &body
) {
    inner.dump(addr, mnemonic, body);
}

void RustPcodeEmitProxy::dump(
    const Address &addr,
    OpCode op_code,
    VarnodeData *outvar,
    VarnodeData *vars,
    int4 len
) {
    std::vector<VarnodeData> inputs(vars, vars + len);
    inner.dump(addr, op_code, outvar, inputs);
}

RustLoadImageProxy::RustLoadImageProxy() : LoadImage("undefined") { }

void RustLoadImageProxy::loadFill(uint1 *output_buf, int4 size, const Address &addr) {
    if (inner == nullptr) {
        throw std::runtime_error(std::string("inner image loader is null"));
    }

    try {
        inner->load_fill(output_buf, size, addr);
    } catch (const rust::Error& e) {
        throw DataUnavailError(string(e.what()));
    }
}

string RustLoadImageProxy::getArchType() const {
    return "undefined";
}

void RustLoadImageProxy::adjustVma(long adjust) {
    // TODO
}

RegisterVarnodeName::RegisterVarnodeName(std::pair<VarnodeData, std::string> pair) : pair(pair) {}
const VarnodeData& RegisterVarnodeName::getVarnode() const {
    return std::get<0>(pair);
}
const std::string& RegisterVarnodeName::getName() const {
    return std::get<1>(pair);
}

SleighProxy::SleighProxy(unique_ptr<RustLoadImageProxy> loader, unique_ptr<ContextDatabase> context)
    : Sleigh(loader.get(), context.get()), loader(move(loader)), context(move(context)) {
}

unique_ptr<SleighProxy> construct_new_sleigh(unique_ptr<ContextDatabase> context) {
    auto loader = make_unique<RustLoadImageProxy>();
    return make_unique<SleighProxy>(move(loader), move(context));
}

unique_ptr<ContextDatabase> construct_new_context() {
    return make_unique<ContextInternal>();
}

RustPcodeEmitProxy::RustPcodeEmitProxy(RustPcodeEmit &emit)
    : inner(emit) {
}

void initialize_element_id() {
    ElementId::initialize();
}

void initialize_attribute_id() {
    AttributeId::initialize();
}

unique_ptr<Address> getAddress(const VarnodeData &data) {
    return make_unique<Address>(data.space, data.offset);
}

uint4 getSize(const VarnodeData &data) {
    return data.size;
}

const Document& parseDocumentIntoStore(DocumentStorage &store, const std::string &data) {
    std::stringstream ss;
    ss << data;
    
    // The XML parser references global state. Must guard this with a lock
    static std::mutex parseDocumentMutex;
    const std::lock_guard<std::mutex> lock(parseDocumentMutex);
    return *store.parseDocument(ss);
}

const Element& getDocumentRoot(const Document& document) {
    return *document.getRoot();
}

void parseDocumentAndRegisterRootElement(DocumentStorage &store, const std::string &data) {
    auto& doc = parseDocumentIntoStore(store, data);
    store.registerTag(doc.getRoot());
}

void SleighProxy::parseProcessorConfig(const DocumentStorage &store) {
    // This logic lives in architecture.cc which is not exposed in libsla.
    const Element *element = store.getTag("processor_spec");
    if (element == (const Element*)0) {
        throw LowlevelError("No processor_spec tag found");
    }

    XmlDecode decoder(this, element);
    uint4 elementId = decoder.openElement(ELEM_PROCESSOR_SPEC);
    for (;;) {
        uint4 subId = decoder.peekElement();
        if (subId == 0) {
            break;
        } else if (subId != ELEM_CONTEXT_DATA) {
            // Intentionally skipping non-context elements since this does not fully implement the processor spec
            // TODO There is not explicit logic for opening and closing an unknown element.
            // Build should include architecture.cc to at least ensure the elements are understood.
            decoder.openElement();
            decoder.closeElementSkipping(subId);
        } else {
            context->decodeFromSpec(decoder);
        }
    }
}

std::unique_ptr<std::string> SleighProxy::getRegisterNameProxy(AddrSpace *base, uintb off, int4 size) const {
    return std::make_unique<std::string>(getRegisterName(base, off, size));
}


std::unique_ptr<std::vector<RegisterVarnodeName>> SleighProxy::getAllRegistersProxy() const {
    std::map<VarnodeData, std::string> regmap;
    getAllRegisters(regmap);

    auto reglist = std::make_unique<std::vector<RegisterVarnodeName>>();
    for (auto &regmapEntry : regmap) {
        reglist->push_back(RegisterVarnodeName(regmapEntry));
    }

    return reglist;
}

int4 SleighProxy::disassemblePcode(const RustLoadImage &loadImage, RustPcodeEmit &emit, const Address &baseaddr) const {
    // The loader is stored in the loader proxy only for as long as this manager lives
    RustLoadImageManager manager { *loader.get(), loadImage };

    auto proxy = RustPcodeEmitProxy(emit);
    return Sleigh::oneInstruction(proxy, baseaddr);
}

int4 SleighProxy::disassembleNative(const RustLoadImage &loadImage, RustAssemblyEmit &emit, const Address &baseaddr) const {
    // The loader is stored in the loader proxy only for as long as this manager lives
    RustLoadImageManager manager { *loader.get(), loadImage };

    auto proxy = RustAssemblyEmitProxy(emit);
    return Sleigh::printAssembly(proxy, baseaddr);
}

RustLoadImageManager::RustLoadImageManager(RustLoadImageProxy &proxy, const RustLoadImage &loadImage)
    : proxy(proxy) {
    this->proxy.setInner(loadImage);
}

RustLoadImageManager::~RustLoadImageManager() {
    this->proxy.resetInner();
}
