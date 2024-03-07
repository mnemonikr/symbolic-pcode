//! Convert symbolic bits into And-Inverter Graph (AIG) format

use std::collections::{BTreeSet, HashMap};

use crate::SymbolicBit;

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
enum AigEntry {
    Gate(usize, usize, usize),
}

fn create_circuit(outputs: impl Iterator<Item = SymbolicBit>) {
    let outputs = outputs.collect::<Vec<_>>();
    let mut indexes = BTreeSet::new();
    for bit in outputs.iter() {
        generate_indexes(bit, &mut indexes);
    }

    let index_lookup = indexes
        .iter()
        .enumerate()
        .map(|(index, &bit)| (bit, 2 * (index + 1)))
        .collect::<HashMap<_, _>>();

    let mut entries = BTreeSet::new();
    for bit in outputs.iter() {
        aig_entries(bit, &index_lookup, &mut entries);
    }
}

fn generate_indexes<'a>(bit: &'a SymbolicBit, indexes: &mut BTreeSet<&'a SymbolicBit>) {
    match bit {
        SymbolicBit::Variable(_) => {
            indexes.insert(bit);
        }
        SymbolicBit::And(ref x, ref y) => {
            generate_indexes(x.as_ref(), indexes);
            generate_indexes(y.as_ref(), indexes);
            indexes.insert(bit);
        }
        _ => (),
    };
}

fn generate_aig(bit: &SymbolicBit) {
    let mut indexes = BTreeSet::new();
    generate_indexes(bit, &mut indexes);

    let index_lookup = indexes
        .into_iter()
        .enumerate()
        .map(|(index, bit)| (bit, 2 * (index + 1)))
        .collect::<HashMap<_, _>>();

    let mut entries = Vec::new();
    aig_entries(bit, &index_lookup, &mut entries);
}

fn index(bit: &SymbolicBit, index_lookup: &HashMap<&SymbolicBit, usize>) -> usize {
    match bit {
        SymbolicBit::Not(x) => index_lookup.get(x.as_ref()).unwrap() + 1,
        _ => *index_lookup.get(bit).unwrap(),
    }
}

fn aig_entries(
    bit: &SymbolicBit,
    index_lookup: &HashMap<&SymbolicBit, usize>,
    entries: &mut BTreeSet<AigEntry>,
) {
    let (bit, negation_offset) = if let SymbolicBit::Not(ref x) = bit {
        (x.as_ref(), 1)
    } else {
        (bit, 0)
    };

    match bit {
        SymbolicBit::And(ref x, ref y) => {
            // The ordering in which the indexes were constructed ensures that the index
            // of x is less than or equal to the index of y
            aig_entries(x.as_ref(), index_lookup, entries);
            aig_entries(y.as_ref(), index_lookup, entries);
            entries.insert(AigEntry::Gate(
                index(&bit, &index_lookup) + negation_offset,
                index(&x, &index_lookup),
                index(&y, &index_lookup),
            ));
        }
        _ => (),
    };
}
