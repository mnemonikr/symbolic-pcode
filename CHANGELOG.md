## Unreleased

Nothing here yet.

## libsla-0.3.1

### Changed

* Updated documentation regarding compilation of `.sla` files. Can now build `.sla` files from Rust using [sleigh-compiler](https://crates.io/crates/sleigh-compiler) crate.
* Upgraded `thiserror` from `1` to `2`

## libsla-0.3.0

### Added

* `Sleigh::register_name`: Get the name for a register identified by `VarnodeData`.
* `Sleigh::register_name_map`: Get a mapping of all registers as `VarnodeData` to their respective
names.
* Implemented `PartialOrd` and `Ord` on `VarnodeData` and dependent types to support ordering in
`register_name_map`.

### Changed

* `DependencyError::source` must now implement `Send` and `Sync`. This is required to convert
`Error` to the error reporting type of other reporting frameworks such as `eyre`.
* `Debug` implementations for `Address` and `AddressSpaceId` to use hex values. For Ghidra the
internal `AddressSpaceId` is actually the virtual address of the `AddressSpace` C++ structure.

## libsla-0.2.0

### Changed

* Replaced `GhidraSleigh::new` with `GhidraSleigh::builder` to improve construction ergonomics. The
necessary objects required to construct `GhidraSleigh` (`.pspec` and `.sla` files) must be provided
before it is possible to instantiate the object.

### Fixed

* Various Rust clippy lints

## libsla-0.1.3

### Added

* README.md

### Fixed

* Crate publishing

## libsla-0.1.0

Initial release!
