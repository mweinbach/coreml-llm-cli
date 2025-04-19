import Foundation
import Tokenizers

/// Back‑compat shim: resolve a single token string to its ID.
/// Falls back to `encode(…)` and returns nil if the token
/// expands to more than one ID.
extension Tokenizer {
    /// Returns the ID for `token` if it is recognised as a single, special token.
    /// Otherwise returns `nil`.
    public func tokenToID(token: String) -> Int? {
        let ids = self.encode(text: token)
        guard ids.count == 1 else { return nil }
        return ids.first
    }
}