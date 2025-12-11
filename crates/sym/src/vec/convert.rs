use super::SymbolicBitVec;
use crate::bit::SymbolicBit;
use crate::buf::SymbolicByte;

impl TryInto<Vec<SymbolicByte>> for SymbolicBitVec {
    type Error = String;

    fn try_into(self) -> Result<Vec<SymbolicByte>, Self::Error> {
        if self.bits.len().is_multiple_of(8) {
            Ok(self.into_bytes())
        } else {
            Err(format!(
                "invalid number of bits: {len}",
                len = self.bits.len()
            ))
        }
    }
}

impl IntoIterator for SymbolicBitVec {
    type Item = SymbolicBit;
    type IntoIter = std::collections::vec_deque::IntoIter<SymbolicBit>;

    fn into_iter(self) -> Self::IntoIter {
        self.bits.into_iter()
    }
}

impl FromIterator<SymbolicBit> for SymbolicBitVec {
    fn from_iter<T: IntoIterator<Item = SymbolicBit>>(iter: T) -> Self {
        Self {
            bits: iter.into_iter().collect(),
        }
    }
}
