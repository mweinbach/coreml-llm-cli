import Foundation
import Tokenizers

/// Lightweight resolver for all special/control tokens used by
/// Meta‑Llama‑3 / 3.2 Instruct checkpoints.  Resolution is tolerant of
/// tokenizer variations so the program never aborts due to missing IDs.
struct Llama3SpecialTokens {
    /// End‑of‑turn  («&lt;|eot_id|&gt;»)
    let eotID: Int
    /// Begin‑of‑text («&lt;|begin_of_text|&gt;») – optional.
    let bosID: Int?
    /// «&lt;|start_header_id|&gt;»
    let startHeaderID: Int
    /// «&lt;|end_header_id|&gt;»
    let endHeaderID: Int

    /// Set of non‑printable / control token ids.
    let controlTokenIDs: Set<Int>

    // MARK: – Resolution helper ------------------------------------------------

    /// Tries multiple strategies to map a token string to its single ID.
    /// 1. `tokenToID(…)`
    /// 2. `encode(text:)` (must return exactly one id)
    /// 3. Optional fallback constant supplied by the caller
    private static func resolveID(
        _ token: String,
        tokenizer: Tokenizer,
        fallback: Int? = nil
    ) -> Int {
        if let single = tokenizer.tokenToID(token: token) { return single }
        let ids = tokenizer.encode(text: token)
        if ids.count == 1 { return ids[0] }
        if let fallback { return fallback }
        fatalError("Token “\(token)” missing from tokenizer vocabulary and no fallback available.")
    }

    // MARK: – Public API -------------------------------------------------------

    /// Resolve all ids up‑front using the supplied tokenizer.
    /// Falls back to well‑known default IDs (Meta‑Llama‑3 v0.1) where safe.
    static func resolve(using tokenizer: Tokenizer) -> Llama3SpecialTokens {
        // Defaults taken from Meta‑Llama‑3‑8B tokenizer (April 2025 release).
        let eot        = resolveID("<|eot_id|>",          tokenizer: tokenizer, fallback: 128009)
        let eom        = resolveID("<|eom_id|>",          tokenizer: tokenizer, fallback: 128008) // new
        let startHdr   = resolveID("<|start_header_id|>", tokenizer: tokenizer, fallback: 128006)
        let endHdr     = resolveID("<|end_header_id|>",   tokenizer: tokenizer, fallback: 128007)
        let bosOpt     = tokenizer.tokenToID(token: "<|begin_of_text|>") ??
                         (tokenizer.encode(text: "<|begin_of_text|>").count == 1
                           ? tokenizer.encode(text: "<|begin_of_text|>")[0]
                           : nil)

        var all: Set<Int> = [eot, eom, startHdr, endHdr]
        if let bosOpt { all.insert(bosOpt) }

        return Llama3SpecialTokens(
            eotID: eot,
            bosID: bosOpt,
            startHeaderID: startHdr,
            endHeaderID: endHdr,
            controlTokenIDs: all
        )
    }
}