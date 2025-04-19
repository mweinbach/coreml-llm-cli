// ChatCLI.swift
// Standâ€‘alone CLI for interactive chat with a CoreÂ MLâ€‘converted LLM
// Drop this file into the coremlâ€‘llmâ€‘cli/Sources folder (next to CLI.swift)
// and run:  swift run -c release ChatCLI --repo-id coreml-projects/Llama-2-7b-chat-coreml

import Foundation
import Foundation
import ArgumentParser
import Tokenizers
import Hub
import CoreML
import LLMKit
// import LlamaKit   // (no actual module, so delete if compilation fails)

// Utility: test whether `self` is a prefix of another array.
private extension Array where Element: Equatable {
    func isPrefix(of other: [Element]) -> Bool {
        guard self.count <= other.count else { return false }
        return self.elementsEqual(other.prefix(self.count))
    }
}

// MARK: - Prompt formatting helpers

enum Role: String { case system, user, assistant }

struct ChatMessage {
    var role: Role
    var content: String
}

/// Builds a Llamaâ€‘2â€‘chat style prompt from a list of messages.
struct Llama2Prompt {
    static func build(_ history: [ChatMessage]) -> String {
        var s = ""
        for (i, msg) in history.enumerated() {
            switch msg.role {
            case .system where i == 0:
                s += "<s>[INST] <<SYS>>\n\(msg.content)\n<</SYS>>\n\n"
            case .user:
                s += "\(msg.content) [/INST]"
            case .assistant:
                s += " \(msg.content)</s><s>[INST] "
            default:
                break
            }
        }
        return s
    }
}

// MARK: - CLI definition

@main
struct ChatCLI: AsyncParsableCommand {
    @Option(help: "HuggingFace repo ID for a *chat* CoreML package (mlmodelc chunks + processors).")
    var repoID: String = "coreml-projects/Llama-2-7b-chat-coreml"

    @Option(help: "System prompt injected at the start of every session.")
    var systemPrompt: String = "You are a helpful assistant."

    @Option(help: "Maximum tokens the model can emit for one assistant turn.")
    var maxNewTokens: Int = 512

    mutating func run() async throws {
        // 1. Download or locate model
        let modelURL = try await downloadModel(repoID: repoID)

        // 2. Load tokenizer that matches the model
        let tokenizerName = inferTokenizer(from: repoID)
        let tokenizer = try await AutoTokenizer.from(cached: tokenizerName,
                                                     hubApi: .init(hfToken: HubApi.defaultToken()))
        
        // ---------------------------------------------------------------------
        // Model‑specific stop‑token handling
        // ---------------------------------------------------------------------
        let isLlama3Family = repoID.lowercased().contains("llama-3")
        
        // Tokens that should immediately terminate generation:
//   – 2       : </s>            (Llama‑2 / generic EOS)
        var stopTokenIDs: Set<Int> = [2]
        // Llama‑3 family stop tokens are added dynamically once the tokenizer
        // resolves them, avoiding hard‑coded IDs that may drift between
        // checkpoints/releases.
        
        let llama3Special = isLlama3Family ? Llama3SpecialTokens.resolve(using: tokenizer) : nil

        if let special = llama3Special {
            stopTokenIDs.insert(special.eotID)
        }

        // Preâ€‘compute possible endâ€‘ofâ€‘response token sequences.
        //  â€“ Llamaâ€‘2 chat stops on </s> (idÂ 2)
        //  â€“ Some instruct models (e.g. Metaâ€‘Llamaâ€‘3â€‘Instruct) use â€ś[/RESP]â€ť
        let endSequences: [[Int]] = [
            tokenizer.encode(text: " [/RESP]"),
            tokenizer.encode(text: "[/RESP]")
        ].filter { !$0.isEmpty }

        // Tokens that should immediately terminate generation:
        //   â€“Â 2       : </s>            (Llamaâ€‘2Â / generic EOS)
        //   â€“Â 128003  : <|eot_id|>      (Llamaâ€‘3Â /Â Llamaâ€‘3.2 Instruct EOS)
// (stopTokenIDs already declared above)

        // Detect Llamaâ€‘3â€‘family models (covers â€śllamaâ€‘3.2â€ť as well) and
        // unconditionally register the reserved <|eot_id|> token ID 128003.
        // Some tokenizer configs omit this ID from their specialâ€‘tokens list,
        // so relying on encode(â€¦) alone is not always sufficient.
// (isLlama3Family already calculated above)
// Duplicate removed – handled by tokenizer‑derived ID

        // Fallback: capture whatever ID the tokenizer maps <|eot_id|> to, if any,
        // so the logic stays futureâ€‘proof should the mapping ever change.
        if let eotID = tokenizer.encode(text: "<|eot_id|>").first {
            stopTokenIDs.insert(eotID)
        }

        // 3. Build the model pipeline (loads just once!)
        let pipeline = try ModelPipeline.from(
            folder: modelURL,
            modelPrefix: nil,
            cacheProcessorModelName: "cache-processor.mlmodelc",
            logitProcessorModelName: "logit-processor.mlmodelc"
        )
        try pipeline.load()
        print("âś… Model loaded. Type your message and press return (empty line to quit).\n")

        // 4. Chat loop -----------------------------------------------------------
        var history: [ChatMessage] = [.init(role: .system, content: systemPrompt)]

        while true {
            print("🧑‍💻 ", terminator: "")
            fflush(stdout)
            guard let userInput = readLine(strippingNewline: true), !userInput.isEmpty else {
                print("👋  Bye!")
                break
            }
            history.append(.init(role: .user, content: userInput))

            // Build prompt and encode (choose correct template)
            let prompt: String
            if repoID.lowercased().contains("llama-3.2") || repoID.lowercased().contains("llama-3") {
                prompt = Llama3Prompt.build(history)
            } else {
                prompt = Llama2Prompt.build(history)
            }
            let promptTokens = tokenizer.encode(text: prompt)

            // Stream inference
            print("🤖 ", terminator: "")
            fflush(stdout)
            var newTokenIds: [Int] = []
            var pending: [Int] = [] // tokens not yet printed
            var assistantReplyBuffer = ""
            
            streamLoop: for try await p in try pipeline.predict(tokens: promptTokens,
                                                                maxNewTokens: maxNewTokens) {
            
                // Stop generation if a designated stop token is produced
                if stopTokenIDs.contains(p.newToken) { break streamLoop }
            
                newTokenIds.append(p.newToken)
                pending.append(p.newToken)
            
                // 1. Check if pending *exactly* matches an end sequence -> stop, drop marker.
                if endSequences.contains(where: { $0 == pending }) {
                    newTokenIds.removeLast(pending.count) // strip marker
                    break streamLoop
                }
            
                // 2. Flush tokens that cannot be part of a future end sequence.
                while !pending.isEmpty {
                    // If current pending tokens are a prefix of ANY end sequence, we must wait.
                    let isPrefixOfEnd = endSequences.contains { pending.isPrefix(of: $0) }
                    if isPrefixOfEnd { break }
            
                    // Safe to print the first token.
                    let tok = pending.removeFirst()
                    let isControl = llama3Special?.controlTokenIDs.contains(tok) ?? false
                    if !isControl {
                        let fragment = tokenizer.decode(tokens: [tok])
                        assistantReplyBuffer.append(fragment)
                        print(fragment, terminator: "")
                        fflush(stdout)
                    }
                }
            }
            // Flush whatever remains (safe: cannot be marker, otherwise loop would have broken)
            pending.forEach { tok in
                let isControl = llama3Special?.controlTokenIDs.contains(tok) ?? false
                if !isControl {
                    let fragment = tokenizer.decode(tokens: [tok])
                    assistantReplyBuffer.append(fragment)
                    print(fragment, terminator: "")
                    fflush(stdout)
                }
            }
            print("\n")
            
            let assistantReply = assistantReplyBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
            history.append(.init(role: .assistant, content: assistantReply))
        }
    }

    // MARK: - Helpers -----------------------------------------------------------

    /// Deduce the correct tokenizer repo for a given CoreML package.
    /// We match the most specific strings first so that broader
    /// patterns do not incorrectly capture later variants.
    func inferTokenizer(from repoID: String) -> String {
        let id = repoID.lowercased()

        // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Llamaâ€‘2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if id.contains("llama-2-7b") {
            return "pcuenq/Llama-2-7b-chat-coreml"
        }

        // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Llamaâ€‘3 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        // All Llamaâ€‘3 model sizes share the same tokenizer that lives in
        // `meta-llama/Meta-Llama-3-8B`, so just point everything there.
        if id.contains("llama-3.2-1b") ||
           id.contains("llama-3.2-3b") ||
           id.contains("llama-3.2-8b") ||
           id.contains("llama-3.8b") {
            return "meta-llama/Meta-Llama-3-8B"
        }

        // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        // Assume the tokenizer lives in the same repo.
        return repoID
    }

    /// Minimal downloader (mirrors logic in CLI.swift but without the progress stats).
    func downloadModel(repoID: String) async throws -> URL {
        let hub = HubApi(hfToken: HubApi.defaultToken())
        let repo = Hub.Repo(id: repoID, type: .models)
        let patterns = ["*.mlmodelc/*", "logit*", "cache*"]
        let filenames = try await hub.getFilenames(from: repo, matching: patterns)
        let localURL = hub.localRepoLocation(repo)

        let needsDownload = filenames.contains { !FileManager.default.fileExists(atPath: localURL.appending(component: $0).path(percentEncoded: false)) }
        guard needsDownload else { return localURL }

        print("â¬‡ď¸Ź  Downloading model (~this can take a few minutes)â€¦")
        _ = try await hub.snapshot(from: repo, matching: patterns) { progress in
            if !progress.isFinished {
                fputs("\r\(Int(progress.fractionCompleted * 100))%", stderr)
            }
        }
        print("\nâś… Download complete.")
        return localURL
    }
}