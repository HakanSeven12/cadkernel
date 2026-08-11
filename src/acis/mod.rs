//! Conversion between ACIS records and kernel B-rep topology.
//!
//! Provenance preserves untouched source records during lowering.

mod append;
mod history;
mod lift;
mod lower;

pub use append::{append, Unappendable, Written};
pub use history::{rebuild_body, HistoryRebuildError};
pub use lift::{lift, lift_body, Loss};
pub use lower::{lower, pending, Unwritable};
