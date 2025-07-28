#pragma once

#include "error.hh"
#include "loadimage.hh"
#include "translate.hh"
#include "xml.hh"

// The following are errors thrown from the decompiler codebase.
// Almost all of the errors are children of LowlevelError
//
// BadDataError : LowlevelError
// DataUnavailError : LowlevelError
// DuplicateFunctionError : RecovError
// EvaluationError : LowlevelError
// JavaError : LowlevelError
// JumptableNotReachableError : LowlevelError
// JumptableThunkError : LowlevelError
// LowlevelError (error.hh)
// ParamUnassignedError : LowlevelError
// ParseError : LowlevelError
// RecovError : LowlevelError
// SleighError : LowlevelError
// UnimplError : LowlevelError
//
// IfaceError (interface.hh)
// IfaceExecutionError : IfaceError
// IfaceParseError : IfaceError
//
// DecoderError (xml.hh)

namespace rust {
    namespace behavior {
        template <typename Try, typename Fail>
        static void trycatch(Try &&func, Fail &&fail) noexcept {
            try {
                func();
            } catch (const ghidra::UnimplError &e) {
                // Defined in translate.hh, type of LowLevelError
                fail("UnimplError: " + e.explain);
            } catch (const ghidra::DataUnavailError &e) {
                // Defined in loadimage.hh, type of LowLevelError
                fail("DataUnavailError: " + e.explain);
            } catch (const ghidra::LowlevelError &e) {
                // Top-level error
                fail("LowlevelError: " + e.explain);
            } catch (const ghidra::DecoderError &e) {
                // Top-level error thrown by XML parser
                fail("DecoderError: " + e.explain);
            } catch (const std::exception &e) {
                fail(e.what());
            }
        }
    } // namespace behavior
} // namespace rust
